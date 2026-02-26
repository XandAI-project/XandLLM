//! Distillation orchestrator — pipelined two-phase design.
//!
//! **Phase 1** (`Phase1Runner`) — Teacher generates completions.
//! The teacher is the **only** model in memory.  When `Phase1Runner::run`
//! returns, the teacher is **dropped**, freeing all VRAM before the student
//! is ever allocated.
//!
//! **Phase 2** (`Phase2Runner`) — Student trains.
//! After Phase 1 the full GPU memory budget is available for the student.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Tensor;
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

// Phase 1 uses indicatif for its rich progress bar (it runs in foreground with a TTY
// from the distill-docker.sh wrapper).  Phase 2 replaces the bar with explicit info!
// logs because `docker run` without `-t` suppresses indicatif output entirely.

use xandllm_core::{chat_template, Tokenizer};

use crate::dataset::{DataLoader, DataPoint};
use crate::student::TrainableStudent;
use crate::teacher::Teacher;

// ── Config ────────────────────────────────────────────────────────────────────

/// Hyper-parameters for the distillation run.
#[derive(Debug, Clone)]
pub struct DistillConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_seq_len: usize,
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

#[derive(Serialize, Deserialize)]
struct TeacherRecord {
    prompt: String,
    completion: String,
}

// ── Training statistics ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_steps: usize,
    pub final_loss: f32,
    pub elapsed_secs: f64,
    pub tokens_per_sec: f64,
}

// ── Phase 1 Runner ────────────────────────────────────────────────────────────

/// Runs Phase 1 only — teacher inference.
///
/// Holds the teacher exclusively.  Calling [`run`][Phase1Runner::run] consumes
/// `self`, so the teacher is dropped (VRAM freed) before Phase 2 begins.
pub struct Phase1Runner {
    teacher: Teacher,
    config: DistillConfig,
    output_dir: PathBuf,
}

impl Phase1Runner {
    pub fn new(teacher: Teacher, config: DistillConfig, output_dir: PathBuf) -> Self {
        Self { teacher, config, output_dir }
    }

    /// Generate teacher completions for every prompt in `dataset`.
    ///
    /// On return `self` (and the teacher) is dropped, freeing VRAM.
    /// Returns `(teacher_data, tokenizer)` for use in Phase 2.
    pub fn run(mut self, dataset: &DataLoader) -> Result<(Vec<DataPoint>, Arc<Tokenizer>)> {
        std::fs::create_dir_all(&self.output_dir)
            .with_context(|| format!("Cannot create output dir: {}", self.output_dir.display()))?;

        let tokenizer = self.teacher.tokenizer();
        let cache_path = self.output_dir.join("teacher_outputs.jsonl");

        let data = generate_resumable(
            &mut self.teacher,
            &self.config,
            dataset,
            &cache_path,
        )?;

        info!("Phase 1 complete — releasing teacher from memory");
        drop(self.teacher);
        Ok((data, tokenizer))
    }
}

// ── Phase 2 Runner ────────────────────────────────────────────────────────────

/// Runs Phase 2 only — student training.
///
/// Holds the student exclusively (teacher has already been freed by Phase 1).
pub struct Phase2Runner {
    student: TrainableStudent,
    config: DistillConfig,
}

impl Phase2Runner {
    pub fn new(student: TrainableStudent, config: DistillConfig) -> Self {
        Self { student, config }
    }

    /// Train the student and return `(stats, trained_student)`.
    pub fn run(
        mut self,
        teacher_data: &[DataPoint],
        tokenizer: Arc<Tokenizer>,
    ) -> Result<(TrainingStats, TrainableStudent)> {
        let total_batches_per_epoch =
            (teacher_data.len() + self.config.batch_size - 1) / self.config.batch_size;
        let total_steps = total_batches_per_epoch * self.config.epochs;

        info!(
            examples = teacher_data.len(),
            epochs = self.config.epochs,
            batch_size = self.config.batch_size,
            lr = self.config.learning_rate,
            total_steps,
            "Phase 2: starting student training"
        );

        // Phase 2 uses explicit info! logs instead of indicatif because Docker
        // runs without a TTY and indicatif renders nothing in that environment.
        // Every step emits a log line so the user can confirm the process is alive.
        info!("Phase 2: initializing AdamW optimizer (may take a moment for large models)");

        let params = ParamsAdamW {
            lr: self.config.learning_rate,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(self.student.trainable_vars(), params)
            .context("Failed to create AdamW optimiser")?;

        info!(total_steps, "Phase 2: optimizer ready — entering training loop");

        let mut step = 0usize;
        let mut last_loss = 0.0f32;
        let mut total_tokens: usize = 0;
        let start = Instant::now();

        for epoch in 0..self.config.epochs {
            info!(
                epoch = epoch + 1,
                total_epochs = self.config.epochs,
                "Phase 2: epoch start"
            );
            let epoch_start = Instant::now();

            for batch in teacher_data.chunks(self.config.batch_size) {
                let tok_batch = DataLoader::tokenize_batch(
                    batch,
                    &tokenizer,
                    self.config.max_seq_len,
                )
                .context("Tokenisation error")?;

                let seq_len = tok_batch.input_ids.first().map(|r| r.len()).unwrap_or(0);
                let device = &self.student.device.clone();

                info!(
                    step = step + 1,
                    total_steps,
                    epoch = epoch + 1,
                    seq_len,
                    "Phase 2: step start (forward+backward may take several minutes on CPU)"
                );

                let step_start = Instant::now();

                // Build input tensor [batch, seq_len].
                let input_ids = tensor_2d(&tok_batch.input_ids, device)?;

                // Derive per-sample training targets from the tokenized batch.
                //
                // `candle_transformers::Llama::forward` always returns logits for the
                // LAST position only ([batch, vocab]).  We therefore run two forward
                // passes per step and average their losses:
                //
                //   Pass 1 — full sequence  → predicts last completion token
                //   Pass 2 — prompt prefix  → predicts first completion token
                //
                // Targets are extracted from `labels` (next-token targets) using
                // `completion_mask` to locate the first/last completion positions.
                let mut prompt_lengths = Vec::with_capacity(tok_batch.input_ids.len());
                let mut last_labels   = Vec::with_capacity(tok_batch.input_ids.len());
                let mut first_labels  = Vec::with_capacity(tok_batch.input_ids.len());

                for (i, (labels_row, mask_row)) in tok_batch.labels
                    .iter()
                    .zip(&tok_batch.completion_mask)
                    .enumerate()
                {
                    // First completion index: first mask position that is 1.0.
                    let first_comp = mask_row.iter().position(|&m| m > 0.5).unwrap_or(0);
                    // Number of tokens to feed to predict the first completion token.
                    // input_ids[0..first_comp+1] = full[0..first_comp+1] ends with p_{P-1},
                    // so the model predicts full[first_comp+1] = c_0.
                    let prompt_len = (first_comp + 1).min(tok_batch.input_ids[i].len());
                    prompt_lengths.push(prompt_len);

                    first_labels.push(
                        labels_row.get(first_comp).copied().unwrap_or(0)
                    );

                    // Last completion index: last mask position that is 1.0.
                    let last_comp = mask_row.iter().rposition(|&m| m > 0.5).unwrap_or(first_comp);
                    last_labels.push(
                        labels_row.get(last_comp).copied().unwrap_or(0)
                    );
                }

                let loss = self.student
                    .train_step(&input_ids, &prompt_lengths, &last_labels, &first_labels)
                    .context("Student train_step failed")?;

                last_loss = loss.to_scalar::<f32>().unwrap_or(f32::NAN);

                let batch_tokens: usize = tok_batch.input_ids.iter().map(|r| r.len()).sum();
                total_tokens += batch_tokens;

                optimizer.backward_step(&loss)
                    .context("Backward/optimizer step failed")?;

                step += 1;

                let step_secs = step_start.elapsed().as_secs_f64();
                let elapsed = start.elapsed().as_secs_f64();
                let steps_remaining = total_steps.saturating_sub(step);
                let avg_secs_per_step = if step > 0 { elapsed / step as f64 } else { step_secs };
                let eta_secs = (avg_secs_per_step * steps_remaining as f64) as u64;

                info!(
                    step,
                    total_steps,
                    epoch = epoch + 1,
                    loss = format!("{:.4}", last_loss),
                    step_secs = format!("{:.1}s", step_secs),
                    eta = format_duration(eta_secs),
                    "Phase 2: step complete"
                );
            }

            let epoch_secs = epoch_start.elapsed().as_secs_f64();
            info!(
                epoch = epoch + 1,
                loss = format!("{:.4}", last_loss),
                epoch_secs = format!("{:.1}s", epoch_secs),
                "Phase 2: epoch complete"
            );
        }

        info!(
            final_loss = format!("{:.4}", last_loss),
            total_steps = step,
            "Phase 2: training complete"
        );

        let elapsed = start.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 { total_tokens as f64 / elapsed } else { 0.0 };

        let stats = TrainingStats {
            total_steps: step,
            final_loss: last_loss,
            elapsed_secs: elapsed,
            tokens_per_sec: tps,
        };

        Ok((stats, self.student))
    }
}

// ── Phase 1 generation logic (free function, shared by runner) ────────────────

fn generate_resumable(
    teacher: &mut Teacher,
    config: &DistillConfig,
    dataset: &DataLoader,
    cache_path: &Path,
) -> Result<Vec<DataPoint>> {
    let total = dataset.len();
    let chat_fmt = teacher.chat_format().to_string();

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

    info!(total, already_done, remaining, "Phase 1: generating teacher completions");

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(cache_path)
        .with_context(|| format!("Cannot open {} for appending", cache_path.display()))?;
    let mut writer = std::io::BufWriter::new(file);

    let pb = phase1_progress_bar(total as u64);
    pb.set_position(already_done as u64);
    let non_tty = pb.is_hidden();

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

        let (completion, gen_tokens) = match teacher.generate_completion(
            &formatted,
            config.teacher_max_tokens,
        ) {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, prompt = %dp.prompt, "Teacher generation failed; using original completion");
                (dp.completion.clone(), 0)
            }
        };

        total_tokens += gen_tokens;

        let rec = TeacherRecord { prompt: formatted.clone(), completion: completion.clone() };
        writeln!(writer, "{}", serde_json::to_string(&rec)?)?;
        writer.flush()?;

        results.push(DataPoint { prompt: formatted, completion });

        let done = (i + 1 - already_done) as f64;
        let elapsed = start.elapsed().as_secs_f64().max(0.001);
        let tps = total_tokens as f64 / elapsed;
        let eta_secs = ((total - i - 1) as f64 * (elapsed / done)) as u64;

        if non_tty {
            let done_count = i + 1 - already_done;
            if done_count % 50 == 0 || i + 1 == total {
                let pct = (i + 1) as f64 / total as f64 * 100.0;
                info!(
                    completed = i + 1,
                    total,
                    pct = format!("{:.1}%", pct),
                    tok_per_sec = format!("{:.1}", tps),
                    eta = format_duration(eta_secs),
                    "Phase 1 progress"
                );
            }
        } else {
            pb.set_message(format!("{:.1} tok/s | ETA {}", tps, format_duration(eta_secs)));
            pb.set_position((i + 1) as u64);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let tps = if elapsed > 0.0 { total_tokens as f64 / elapsed } else { 0.0 };
    pb.finish_with_message(format!(
        "Phase 1 complete — {remaining} prompts in {} ({:.1} tok/s)",
        format_duration(elapsed as u64),
        tps,
    ));

    info!(path = %cache_path.display(), count = results.len(), "Teacher outputs saved");
    Ok(results)
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

// ── Tensor helpers ────────────────────────────────────────────────────────────

fn tensor_2d(rows: &[Vec<u32>], device: &candle_core::Device) -> Result<Tensor> {
    let batch = rows.len();
    let seq = rows.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<u32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (batch, seq), device).context("Failed to build u32 tensor")
}
