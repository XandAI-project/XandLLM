//! Distillation orchestrator.
//!
//! Runs the two-phase sequence-level distillation pipeline:
//!
//! **Phase 1** — Teacher generation
//! For every prompt in the dataset the teacher model generates a completion.
//! Results are saved to an intermediate JSONL file in the output directory so
//! the (expensive) teacher pass can be skipped if it already ran.
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
    /// 1. Generate teacher completions for every prompt (or reuse a previous
    ///    run's `teacher_outputs.jsonl` if it already exists).
    /// 2. Train the student.
    pub fn run(&mut self, dataset: &DataLoader) -> Result<TrainingStats> {
        std::fs::create_dir_all(&self.output_dir)
            .with_context(|| format!("Cannot create output dir: {}", self.output_dir.display()))?;

        let teacher_cache = self.output_dir.join("teacher_outputs.jsonl");

        let teacher_data: Vec<DataPoint> = if teacher_cache.exists() {
            info!(
                path = %teacher_cache.display(),
                "Reusing existing teacher_outputs.jsonl (delete the file to regenerate)"
            );
            load_teacher_cache(&teacher_cache)?
        } else {
            info!(examples = dataset.len(), "Phase 1: generating teacher completions");
            let data = self.generate_teacher_outputs(dataset)?;
            save_teacher_cache(&teacher_cache, &data)?;
            info!(
                path = %teacher_cache.display(),
                count = data.len(),
                "Teacher outputs saved"
            );
            data
        };

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

    /// Run the teacher model over every prompt in `dataset` and return the
    /// resulting `(prompt, completion)` pairs.
    pub fn generate_teacher_outputs(&mut self, dataset: &DataLoader) -> Result<Vec<DataPoint>> {
        let n = dataset.len();
        let pb = progress_bar(n as u64, "Teacher generation");
        let chat_fmt = self.teacher.chat_format().to_string();

        let mut results = Vec::with_capacity(n);

        for dp in dataset.as_slice() {
            // Format the prompt with the teacher's chat template so the
            // teacher sees the proper instruction wrapper.
            let formatted = chat_template::format_prompt(&chat_fmt, &dp.prompt, "You are a helpful assistant.");

            let completion = match self.teacher.generate_completion(
                &formatted,
                self.config.teacher_max_tokens,
            ) {
                Ok(c) => c,
                Err(e) => {
                    warn!(error = %e, prompt = %dp.prompt, "Teacher generation failed; using original completion");
                    dp.completion.clone()
                }
            };

            results.push(DataPoint {
                prompt: formatted,
                completion,
            });
            pb.inc(1);
        }

        pb.finish_with_message("Teacher generation complete");
        Ok(results)
    }

    // ── Phase 2 ───────────────────────────────────────────────────────────────

    /// Train the student on `teacher_data` using AdamW + masked cross-entropy.
    pub fn train(&mut self, teacher_data: &[DataPoint]) -> Result<TrainingStats> {
        let tokenizer = self.teacher.tokenizer();

        // Build the optimiser from all trainable parameters in the VarMap.
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

                // Build Tensors from the tokenised batch
                let input_ids = tensor_2d(&tok_batch.input_ids, device)?;
                let labels = tensor_2d_u32(&tok_batch.labels, device)?;
                let mask = tensor_2d_f32(&tok_batch.completion_mask, device)?;

                // Forward pass
                let logits = self.student.forward(&input_ids)
                    .context("Student forward pass failed")?;

                // Masked cross-entropy loss
                let loss = TrainableStudent::masked_ce_loss(&logits, &labels, &mask)
                    .context("Loss computation failed")?;

                last_loss = loss.to_scalar::<f32>().unwrap_or(f32::NAN);

                // Count tokens contributing to this step for throughput stats
                let batch_tokens: usize = tok_batch.input_ids.iter().map(|r| r.len()).sum();
                total_tokens += batch_tokens;

                // Backward + optimizer step
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

fn save_teacher_cache(path: &Path, data: &[DataPoint]) -> Result<()> {
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create {}", path.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    for dp in data {
        let rec = TeacherRecord { prompt: dp.prompt.clone(), completion: dp.completion.clone() };
        let line = serde_json::to_string(&rec)?;
        writeln!(writer, "{line}")?;
    }
    Ok(())
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

/// Build a 2D `[batch, seq_len]` U32 tensor from a batch of equal-length rows.
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

/// Build a 2D `[batch, seq_len]` F32 tensor from a batch of equal-length rows.
fn tensor_2d_f32(rows: &[Vec<f32>], device: &Device) -> Result<Tensor> {
    let batch = rows.len();
    let seq = rows.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (batch, seq), device)
        .context("Failed to build f32 tensor")
}
