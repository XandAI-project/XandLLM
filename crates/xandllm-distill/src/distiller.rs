//! Distillation orchestrator.
//!
//! Runs the two-phase sequence-level distillation pipeline:
//!
//! **Phase 1** — Teacher generation
//! For every prompt in the dataset the teacher model generates a completion.
//! Results are saved incrementally to an intermediate JSONL file in the output
//! directory.  If the process is interrupted, the next run resumes from where
//! it left off.
//!
//! **Phase 2** — Student training
//! The student model is trained with cross-entropy loss to predict the
//! teacher-generated completions, using AdamW as the optimiser.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use xandllm_core::chat_template;

use crate::dataset::{DataLoader, DataPoint};
use crate::student::TrainableStudent;
use crate::teacher::Teacher;

// ── Config ────────────────────────────────────────────────────────────────────

/// Hyper-parameters for the distillation run.
#[derive(Debug, Clone)]
pub struct DistillConfig {
    /// Number of full passes over the teacher-output dataset.
    pub epochs: usize,
    /// Number of examples per training batch.
    pub batch_size: usize,
    /// AdamW learning rate.
    pub learning_rate: f64,
    /// Maximum token-sequence length (prompt + completion).
    pub max_seq_len: usize,
    /// Maximum new tokens the teacher may generate per prompt.
    pub teacher_max_tokens: usize,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 4,
            learning_rate: 1e-4,
            max_seq_len: 2048,
            teacher_max_tokens: 512,
        }
    }
}

// ── Checkpoint format ─────────────────────────────────────────────────────────

/// One record in the intermediate teacher-output JSONL file.
#[derive(Serialize, Deserialize)]
struct TeacherRecord {
    /// The formatted prompt sent to the teacher.
    prompt: String,
    /// The teacher's completion (stop tokens stripped).
    completion: String,
}

// ── Training statistics ───────────────────────────────────────────────────────

/// Summary returned after training completes.
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_steps: usize,
    pub final_loss: f32,
    pub elapsed_secs: f64,
    pub tokens_per_sec: f64,
}

// ── Distiller ─────────────────────────────────────────────────────────────────

/// Orchestrates Phase 1 (teacher generation) and Phase 2 (student training).
pub struct Distiller {
    teacher: Teacher,
    student: TrainableStudent,
    config: DistillConfig,
    output_dir: PathBuf,
}

impl Distiller {
    /// Create a new `Distiller`.
    ///
    /// `output_dir` is used for the intermediate `teacher_outputs.jsonl` file
    /// and the final model weights.  It is created if it does not exist.
    pub fn new(
        teacher: Teacher,
        student: TrainableStudent,
        config: DistillConfig,
        output_dir: PathBuf,
    ) -> Self {
        Self { teacher, student, config, output_dir }
    }

    /// Consume the `Distiller` and return the trained student.
    pub fn into_student(self) -> TrainableStudent {
        self.student
    }

    /// Run the full distillation pipeline.
    ///
    /// 1. Generate teacher completions for every prompt (or reuse / resume a
    ///    previous run's `teacher_outputs.jsonl`).
    /// 2. Train the student.
    pub fn run(&mut self, dataset: &DataLoader) -> Result<TrainingStats> {
        std::fs::create_dir_all(&self.output_dir)
            .with_context(|| format!("Cannot create output dir: {}", self.output_dir.display()))?;

        let teacher_cache = self.output_dir.join("teacher_outputs.jsonl");

        let teacher_data = self.generate_teacher_outputs_resumable(dataset, &teacher_cache)?;

        info!(
            examples = teacher_data.len(),
            epochs = self.config.epochs,
            batch_size = self.config.batch_size,
            lr = self.config.learning_rate,
            "Phase 2: training student"
        );
        self.train(&teacher_data)
    }

    // ── Phase 1 ───────────────────────────────────────────────────────────────

    /// Generate teacher completions, resuming from a partial cache file.
    ///
    /// Already-completed records are loaded from `cache_path`.  New records are
    /// appended one-by-one so progress survives crashes.
    fn generate_teacher_outputs_resumable(
        &mut self,
        dataset: &DataLoader,
        cache_path: &Path,
    ) -> Result<Vec<DataPoint>> {
        let total = dataset.len();
        let chat_fmt = self.teacher.chat_format().to_string();

        // Load any previously completed records.
        let mut results: Vec<DataPoint> = if cache_path.exists() {
            let existing = load_teacher_cache(cache_path)?;
            if existing.len() >= total {
                info!(
                    path = %cache_path.display(),
                    count = existing.len(),
                    "Phase 1 already complete — reusing teacher_outputs.jsonl"
                );
                return Ok(existing);
            }
            info!(
                path = %cache_path.display(),
                completed = existing.len(),
                remaining = total - existing.len(),
                "Resuming Phase 1 from existing teacher_outputs.jsonl"
            );
            existing
        } else {
            Vec::with_capacity(total)
        };

        let already_done = results.len();
        let remaining = total - already_done;

        if remaining == 0 {
            return Ok(results);
        }

        info!(
            total,
            already_done,
            remaining,
            "Phase 1: generating teacher completions"
        );

        // Open the cache file in append mode for incremental writes.
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(cache_path)
            .with_context(|| format!("Cannot open {} for appending", cache_path.display()))?;
        let mut writer = std::io::BufWriter::new(file);

        let pb = phase1_progress_bar(total as u64);
        pb.set_position(already_done as u64);

        let start = Instant::now();
        let mut total_tokens: usize = 0;

        for (i, dp) in dataset.as_slice().iter().enumerate() {
            if i < already_done {
                continue;
            }

            let formatted = chat_template::format_prompt(
                &chat_fmt,
                &dp.prompt,
                "You are a helpful assistant.",
            );

            let (completion, gen_tokens) = match self.teacher.generate_completion(
                &formatted,
                self.config.teacher_max_tokens,
            ) {
                Ok(r) => r,
                Err(e) => {
                    warn!(error = %e, prompt = %dp.prompt, "Teacher generation failed; using original completion");
                    (dp.completion.clone(), 0)
                }
            };

            total_tokens += gen_tokens;

            // Append to cache immediately.
            let rec = TeacherRecord {
                prompt: formatted.clone(),
                completion: completion.clone(),
            };
            let line = serde_json::to_string(&rec)?;
            writeln!(writer, "{line}")?;
            writer.flush()?;

            results.push(DataPoint { prompt: formatted, completion });

            // Update progress bar with tok/s and ETA.
            let done = (i + 1 - already_done) as f64;
            let elapsed = start.elapsed().as_secs_f64().max(0.001);
            let tps = total_tokens as f64 / elapsed;
            let prompts_remaining = (total - i - 1) as f64;
            let secs_per_prompt = elapsed / done;
            let eta_secs = (prompts_remaining * secs_per_prompt) as u64;

            pb.set_message(format!(
                "{:.1} tok/s | ETA {}",
                tps,
                format_duration(eta_secs),
            ));
            pb.set_position((i + 1) as u64);
        }

        let elapsed = start.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 { total_tokens as f64 / elapsed } else { 0.0 };
        pb.finish_with_message(format!(
            "Phase 1 complete — {remaining} prompts in {} ({:.1} tok/s)",
            format_duration(elapsed as u64),
            tps,
        ));

        info!(
            path = %cache_path.display(),
            count = results.len(),
            "Teacher outputs saved"
        );

        Ok(results)
    }

    // ── Phase 2 ───────────────────────────────────────────────────────────────

    /// Train the student on `teacher_data` using AdamW + masked cross-entropy.
    pub fn train(&mut self, teacher_data: &[DataPoint]) -> Result<TrainingStats> {
        let tokenizer = self.teacher.tokenizer();

        let params = ParamsAdamW {
            lr: self.config.learning_rate,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(self.student.trainable_vars(), params)
            .context("Failed to create AdamW optimiser")?;

        let total_batches_per_epoch =
            (teacher_data.len() + self.config.batch_size - 1) / self.config.batch_size;
        let total_steps = total_batches_per_epoch * self.config.epochs;

        let pb = progress_bar(total_steps as u64, "Student training");

        let mut step = 0usize;
        let mut last_loss = 0.0f32;
        let mut total_tokens: usize = 0;
        let start = Instant::now();

        for epoch in 0..self.config.epochs {
            for batch in teacher_data.chunks(self.config.batch_size) {
                let tok_batch = DataLoader::tokenize_batch(
                    batch,
                    &tokenizer,
                    self.config.max_seq_len,
                )
                .context("Tokenisation error")?;

                let device = &self.student.device.clone();

                let input_ids = tensor_2d(&tok_batch.input_ids, device)?;
                let labels = tensor_2d_u32(&tok_batch.labels, device)?;
                let mask = tensor_2d_f32(&tok_batch.completion_mask, device)?;

                let logits = self.student.forward(&input_ids)
                    .context("Student forward pass failed")?;

                let loss = TrainableStudent::masked_ce_loss(&logits, &labels, &mask)
                    .context("Loss computation failed")?;

                last_loss = loss.to_scalar::<f32>().unwrap_or(f32::NAN);

                let batch_tokens: usize = tok_batch.input_ids.iter().map(|r| r.len()).sum();
                total_tokens += batch_tokens;

                optimizer.backward_step(&loss)
                    .context("Backward/optimizer step failed")?;

                step += 1;
                pb.set_message(format!("epoch {}/{} loss {:.4}", epoch + 1, self.config.epochs, last_loss));
                pb.inc(1);
            }

            info!(epoch = epoch + 1, loss = last_loss, "Epoch complete");
        }

        pb.finish_with_message(format!("Training complete — final loss {last_loss:.4}"));

        let elapsed = start.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 { total_tokens as f64 / elapsed } else { 0.0 };

        Ok(TrainingStats {
            total_steps: step,
            final_loss: last_loss,
            elapsed_secs: elapsed,
            tokens_per_sec: tps,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn phase1_progress_bar(total: u64) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({percent}%) | {msg}",
        )
        .unwrap()
        .progress_chars("█▓░"),
    );
    pb.set_message("starting...");
    pb
}

fn progress_bar(total: u64, label: &str) -> ProgressBar {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("█▓░"),
    );
    pb.set_message(label.to_string());
    pb
}

fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {:02}m {:02}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    }
}

fn load_teacher_cache(path: &Path) -> Result<Vec<DataPoint>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open {}", path.display()))?;
    let mut data = Vec::new();
    for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let rec: TeacherRecord = serde_json::from_str(trimmed)
            .with_context(|| format!("Parse error at {}:{}", path.display(), i + 1))?;
        data.push(DataPoint { prompt: rec.prompt, completion: rec.completion });
    }
    Ok(data)
}

// ── Tensor construction helpers ───────────────────────────────────────────────

fn tensor_2d(rows: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let batch = rows.len();
    let seq = rows.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (batch, seq), device)
        .context("Failed to build u32 tensor")
}

fn tensor_2d_u32(rows: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    tensor_2d(rows, device)
}

fn tensor_2d_f32(rows: &[Vec<f32>], device: &Device) -> Result<Tensor> {
    let batch = rows.len();
    let seq = rows.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (batch, seq), device)
        .context("Failed to build f32 tensor")
}
