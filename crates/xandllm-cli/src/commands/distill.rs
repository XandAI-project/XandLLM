//! `xandllm distill` — distil a teacher model into a smaller student.
//!
//! Models that are not already cached are **automatically downloaded** from
//! Hugging Face before training starts — no separate `xandllm pull` step is
//! required.
//!
//! ## Pipeline
//!
//! 1. **Phase 1** — Teacher is loaded (GPU if VRAM allows), generates all
//!    completions, then is **dropped** before Phase 2 begins, freeing VRAM.
//! 2. **Phase 2** — Student is loaded on GPU (if VRAM estimate allows) or CPU,
//!    then trained with AdamW.
//!
//! This sequential loading ensures neither phase is memory-limited by the
//! other model.
//!
//! ## Modes
//!
//! **Fresh student** (random weights, architecture from size preset):
//! ```text
//! xandllm distill \
//!   --model-from  Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
//!   --dataset     ./my_data \
//!   --model-to    ./output/XandLM-1B \
//!   --size        1b
//! ```
//!
//! **Fine-tune an existing smaller model**:
//! ```text
//! xandllm distill \
//!   --model-from  Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
//!   --dataset     ./my_data \
//!   --model-to    ./output/MyFineTuned \
//!   --student-base Qwen/Qwen2.5-1.5B
//! ```
//!
//! ## Dataset format
//!
//! Each line of the `.jsonl` files in `--dataset` must be:
//! ```json
//! {"prompt": "...", "completion": "..."}
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use indicatif::MultiProgress;
use tracing::info;

use xandllm_core::Tokenizer;
use xandllm_distill::{
    dataset::DataLoader,
    distiller::{DistillConfig, Phase1Runner, Phase2Runner},
    export::{export, OutputFormat},
    presets::SizePreset,
    student::TrainableStudent,
    teacher::{estimate_model_size_mb, query_free_vram_mb, Teacher},
};
use xandllm_hub::{ModelCache, ModelDownloader};

use crate::config::{expand_cache_dir, AppConfig};

// ── Main entry point ──────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub async fn run(
    model_from: &str,
    dataset: Option<&PathBuf>,
    model_to: &PathBuf,
    output_type: &str,
    size: Option<&str>,
    student_base: Option<&str>,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    max_seq_len: Option<usize>,
    teacher_max_tokens: Option<usize>,
    prefer_gpu: bool,
    model_name: Option<&str>,
    config: &AppConfig,
) -> Result<()> {
    // ── Validate mutually-exclusive flags ─────────────────────────────────────
    match (size, student_base) {
        (None, None) => bail!(
            "Specify either --size (e.g. 1b, 3b, 7b) for a fresh student, \
             or --student-base for fine-tuning an existing model."
        ),
        (Some(_), Some(_)) => bail!(
            "--size and --student-base are mutually exclusive. \
             Use --size for fresh weights or --student-base for fine-tuning."
        ),
        _ => {}
    }

    let output_format = OutputFormat::parse(output_type)
        .with_context(|| format!("Invalid --type '{output_type}'"))?;

    let cache_dir = expand_cache_dir(&config.model.cache_dir);
    let max_seq = max_seq_len.unwrap_or(config.inference.max_sequence_length);
    let cuda_id = config.device.cuda_device_id;
    let use_gpu = prefer_gpu || config.device.prefer_gpu;

    // ── Resolve dataset path ──────────────────────────────────────────────────
    let dataset_path: PathBuf = match dataset {
        Some(p) => p.clone(),
        None => {
            let fallback = PathBuf::from("internal/dataset");
            println!("No --dataset provided, using default: {}", fallback.display());
            fallback
        }
    };

    // ── Auto-download teacher if not cached ───────────────────────────────────
    ensure_cached(model_from, &cache_dir).await
        .with_context(|| format!("Failed to download teacher model '{model_from}'"))?;

    // ── Auto-download student base if not cached ──────────────────────────────
    if let Some(base_id) = student_base {
        ensure_cached(base_id, &cache_dir).await
            .with_context(|| format!("Failed to download student base model '{base_id}'"))?;
    }

    // ── Load dataset ──────────────────────────────────────────────────────────
    info!(path = %dataset_path.display(), "Loading dataset");
    let loader = DataLoader::from_dir(&dataset_path)
        .with_context(|| format!("Failed to load dataset from {}", dataset_path.display()))?;
    info!(examples = loader.len(), "Dataset ready");

    // ── Build distill config ──────────────────────────────────────────────────
    let distill_config = DistillConfig {
        epochs,
        batch_size,
        learning_rate,
        max_seq_len: max_seq,
        teacher_max_tokens: teacher_max_tokens.unwrap_or(512),
    };

    // ── PHASE 1: Teacher inference ────────────────────────────────────────────
    //
    // Load the teacher (GPU if VRAM allows), generate all completions, then
    // DROP the teacher so VRAM is fully available for the student in Phase 2.

    info!("=== Phase 1: Teacher inference ===");
    let teacher = Teacher::load(model_from, &cache_dir, use_gpu, cuda_id, max_seq)?;
    let teacher_model_dir = model_dir_for(model_from, &cache_dir)?;
    let vocab_size = teacher.vocab_size();

    let phase1 = Phase1Runner::new(teacher, distill_config.clone(), model_to.clone());
    // `run` consumes `phase1` — the teacher is dropped at the end of this call.
    let (teacher_data, tokenizer) = phase1.run(&loader)?;

    // ── PHASE 2: Student training ─────────────────────────────────────────────
    //
    // Teacher has been dropped.  Decide student device based on current free VRAM.

    info!("=== Phase 2: Student training ===");
    let student_device = choose_student_device(use_gpu, cuda_id, size, student_base, &cache_dir)?;

    let student = build_student(
        size,
        student_base,
        vocab_size,
        Arc::clone(&tokenizer),
        &student_device,
        &cache_dir,
    )?;

    let phase2 = Phase2Runner::new(student, distill_config);
    let (stats, trained_student) = phase2.run(&teacher_data, Arc::clone(&tokenizer))?;

    info!(
        steps   = stats.total_steps,
        loss    = stats.final_loss,
        elapsed = format!("{:.1}s", stats.elapsed_secs),
        tok_s   = format!("{:.0}", stats.tokens_per_sec),
        "Distillation complete"
    );

    // ── Export ────────────────────────────────────────────────────────────────
    info!(format = ?output_format, output = %model_to.display(), "Exporting model");
    export(
        &trained_student,
        &tokenizer,
        &teacher_model_dir,
        model_to,
        output_format,
        model_name,
    )?;

    println!(
        "\nDistillation complete!\n\
         Output: {}\n\
         Final loss: {:.4}\n\
         Elapsed: {:.1} s ({:.0} tok/s)\n\
         \n\
         Run the distilled model with:\n  xandllm serve --model {}",
        model_to.display(),
        stats.final_loss,
        stats.elapsed_secs,
        stats.tokens_per_sec,
        model_to.display(),
    );

    Ok(())
}

// ── Student device selection ──────────────────────────────────────────────────

/// Proactively decide which device to use for the student, based on a VRAM
/// estimate.  No surprise OOM panics — either we know it fits and use GPU, or
/// we pick CPU upfront.
///
/// Strategy:
/// - Estimate the student's full training memory footprint (weights + grads +
///   AdamW first & second moments ≈ 4× weight size).
/// - Compare against current free VRAM (teacher has already been dropped).
/// - If `training_est ≤ free_vram × 2` → try GPU (covers cases where the
///   estimate is conservative, e.g. 1 B on a 16 GB card).
/// - Otherwise → CPU, no point attempting GPU.
fn choose_student_device(
    use_gpu: bool,
    cuda_id: usize,
    size: Option<&str>,
    student_base: Option<&str>,
    cache_dir: &Path,
) -> Result<Device> {
    if !use_gpu {
        info!("Student: GPU not requested — using CPU");
        return Ok(Device::Cpu);
    }

    // Get a CUDA device; if CUDA feature is off this returns CPU.
    let gpu_device = xandllm_core::select_device(true, cuda_id)?;
    if matches!(gpu_device, Device::Cpu) {
        return Ok(Device::Cpu);
    }

    let free_mb = match query_free_vram_mb() {
        Some(v) => v,
        None => {
            info!("Student: cannot query VRAM — loading on GPU optimistically");
            return Ok(gpu_device);
        }
    };

    let training_est_mb = estimate_student_training_mb(size, student_base, cache_dir);

    // If the training footprint is within 2× of available VRAM it is worth
    // trying GPU.  The factor-of-2 tolerance handles cases where our estimate
    // is conservative (quantisation, GQA, etc.) and lets 1B models attempt
    // 16 GB cards.  A genuine OOM will not occur in practice because the
    // threshold is generous.
    let try_gpu = training_est_mb <= free_mb.saturating_mul(2);

    if try_gpu {
        info!(
            free_vram_mb = free_mb,
            training_est_mb,
            "Student: fits on GPU — using GPU"
        );
        Ok(gpu_device)
    } else {
        info!(
            free_vram_mb = free_mb,
            training_est_mb,
            "Student: too large for GPU — using CPU (system RAM)"
        );
        Ok(Device::Cpu)
    }
}

/// Estimate the full training memory footprint of the student in MiB.
///
/// Training = weights + gradients + AdamW first moment + AdamW second moment
///          ≈ 4 × weight size (all in F32).
///
/// For `--size` presets the weight size comes from the known parameter counts.
/// For `--student-base` fine-tunes we read the safetensors files off disk.
fn estimate_student_training_mb(
    size: Option<&str>,
    student_base: Option<&str>,
    cache_dir: &Path,
) -> u64 {
    // AdamW training uses 4× the weight memory (weights + grad + m1 + m2).
    const ADAMW_FACTOR: u64 = 4;

    let weights_mb: u64 = if let Some(preset_str) = size {
        match preset_str.to_lowercase().as_str() {
            "1b" => 4_400,   // ~1.1 B params × 4 bytes
            "3b" => 12_288,  // ~3.0 B params × 4 bytes
            "7b" => 28_000,  // ~7.0 B params × 4 bytes
            _    => 8_192,   // safe default
        }
    } else if let Some(base_id) = student_base {
        // Parse base_id into repo path and read disk size.
        let (repo_id, _) = parse_model_id(base_id);
        if let Ok(cache) = ModelCache::new(cache_dir) {
            let base_dir = cache.model_dir(repo_id, "main");
            estimate_model_size_mb(&base_dir)
        } else {
            8_192
        }
    } else {
        8_192
    };

    weights_mb * ADAMW_FACTOR
}

// ── Auto-download helper ──────────────────────────────────────────────────────

/// Ensure a model is present in the local cache.
async fn ensure_cached(raw_model_id: &str, cache_dir: &Path) -> Result<()> {
    let (repo_id, quant_tag) = parse_model_id(raw_model_id);
    let cache = ModelCache::new(cache_dir)?;
    let model_dir = cache.model_dir(repo_id, "main");

    if has_weight_files(&model_dir) {
        info!(model_id = repo_id, "Model already cached — skipping download");
        return Ok(());
    }

    println!(
        "Model '{}' not found in cache — downloading from Hugging Face …",
        repo_id
    );

    let downloader = ModelDownloader::new(cache)
        .context("Failed to create model downloader")?;
    let mp = MultiProgress::new();

    let use_gguf = quant_tag.is_some()
        || repo_id.contains("GGUF")
        || repo_id.contains("gguf");

    let paths = if use_gguf {
        downloader
            .pull_gguf(repo_id, "main", None, quant_tag, Some(&mp))
            .await
            .with_context(|| format!("GGUF download failed for '{repo_id}'"))?
    } else {
        downloader
            .pull(repo_id, "main", Some(&mp))
            .await
            .with_context(|| format!("Download failed for '{repo_id}'"))?
    };

    info!(model_id = repo_id, files = paths.len(), "Model downloaded and cached");
    Ok(())
}

fn has_weight_files(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let s = name.to_string_lossy();
        if s.ends_with(".gguf")
            || s == "model.safetensors"
            || s == "model.safetensors.index.json"
        {
            return true;
        }
    }
    false
}

// ── Model-ID parser ───────────────────────────────────────────────────────────

fn parse_model_id(raw: &str) -> (&str, Option<&str>) {
    let stripped = raw.strip_prefix("hf.co/").unwrap_or(raw);
    if let Some(colon) = stripped.rfind(':') {
        let repo = &stripped[..colon];
        let tag = &stripped[colon + 1..];
        if !tag.is_empty() && !tag.contains('/') && !tag.contains('\\') {
            return (repo, Some(tag));
        }
    }
    (stripped, None)
}

// ── Small utilities ───────────────────────────────────────────────────────────

fn model_dir_for(model_id: &str, cache_dir: &Path) -> Result<PathBuf> {
    let (repo_id, _) = parse_model_id(model_id);
    let cache = ModelCache::new(cache_dir)?;
    Ok(cache.model_dir(repo_id, "main"))
}

/// Build a `TrainableStudent` on `device`.
fn build_student(
    size: Option<&str>,
    student_base: Option<&str>,
    vocab_size: usize,
    tokenizer: Arc<Tokenizer>,
    device: &Device,
    cache_dir: &Path,
) -> Result<TrainableStudent> {
    if let Some(preset_str) = size {
        let preset = SizePreset::parse(preset_str)?;
        info!(preset = preset.label(), vocab_size, "Building student from size preset");
        TrainableStudent::from_preset(&preset, vocab_size, tokenizer, device)
    } else {
        let base_id = student_base.expect("validated: --size or --student-base required");
        info!(base = base_id, "Loading student base model for fine-tuning");
        let base_dir = model_dir_for(base_id, cache_dir)?;
        TrainableStudent::from_safetensors(&base_dir, tokenizer, device)
    }
}
