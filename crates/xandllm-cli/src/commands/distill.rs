//! `xandllm distill` — distil a teacher model into a smaller student.
//!
//! Models that are not already cached are **automatically downloaded** from
//! Hugging Face before training starts — no separate `xandllm pull` step is
//! required.
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

use anyhow::{bail, Context, Result};
use indicatif::MultiProgress;
use tracing::info;

use xandllm_distill::{
    dataset::DataLoader,
    distiller::{DistillConfig, Distiller},
    export::{export, OutputFormat},
    presets::SizePreset,
    student::TrainableStudent,
    teacher::Teacher,
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

    // ── Resolve dataset path (fall back to ./internal/dataset/) ──────────────
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

    // ── Load teacher ──────────────────────────────────────────────────────────
    let teacher = Teacher::load(
        model_from,
        &cache_dir,
        prefer_gpu || config.device.prefer_gpu,
        cuda_id,
        config.inference.max_sequence_length,
    )?;

    let tokenizer = teacher.tokenizer();
    let vocab_size = teacher.vocab_size();
    let teacher_model_dir = model_dir_for(model_from, &cache_dir)?;

    // ── Build student ─────────────────────────────────────────────────────────
    let device = xandllm_core::select_device(
        prefer_gpu || config.device.prefer_gpu,
        cuda_id,
    )?;

    let student: TrainableStudent = if let Some(preset_str) = size {
        let preset = SizePreset::parse(preset_str)?;
        info!(preset = preset.label(), vocab_size, "Building student from size preset");
        TrainableStudent::from_preset(&preset, vocab_size, tokenizer.clone(), &device)?
    } else {
        let base_id = student_base.unwrap(); // validated above
        info!(base = base_id, "Loading student base model for fine-tuning");
        let base_dir = model_dir_for(base_id, &cache_dir)?;
        TrainableStudent::from_safetensors(&base_dir, tokenizer.clone(), &device)?
    };

    // ── Configure and run distiller ───────────────────────────────────────────
    let distill_config = DistillConfig {
        epochs,
        batch_size,
        learning_rate,
        max_seq_len: max_seq,
        teacher_max_tokens: teacher_max_tokens.unwrap_or(512),
    };

    let mut distiller = Distiller::new(teacher, student, distill_config, model_to.clone());

    info!("Starting distillation");
    let stats = distiller.run(&loader)?;

    info!(
        steps   = stats.total_steps,
        loss    = stats.final_loss,
        elapsed = format!("{:.1}s", stats.elapsed_secs),
        tok_s   = format!("{:.0}", stats.tokens_per_sec),
        "Distillation complete"
    );

    // ── Export ────────────────────────────────────────────────────────────────
    let trained_student = distiller.into_student();

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

// ── Auto-download helper ──────────────────────────────────────────────────────

/// Ensure a model is present in the local cache.
///
/// If the model directory exists and already contains weight files (`.gguf` or
/// `.safetensors`), nothing is done.  Otherwise the model is downloaded from
/// Hugging Face Hub using the same logic as `xandllm pull`.
///
/// Accepts the same model-id formats as `pull`:
/// - `owner/repo`                  — plain HF repo id
/// - `owner/repo:Q4_0`             — GGUF with quantisation tag
/// - `hf.co/owner/repo:Q4_0`       — Ollama-style prefix (stripped)
async fn ensure_cached(raw_model_id: &str, cache_dir: &Path) -> Result<()> {
    let (repo_id, quant_tag) = parse_model_id(raw_model_id);
    let cache = ModelCache::new(cache_dir)?;
    let model_dir = cache.model_dir(repo_id, "main");

    if has_weight_files(&model_dir) {
        info!(
            model_id = repo_id,
            "Model already cached — skipping download"
        );
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

    info!(
        model_id = repo_id,
        files = paths.len(),
        "Model downloaded and cached"
    );
    Ok(())
}

/// Return `true` if `dir` contains at least one `.gguf` file or a
/// `model.safetensors` / `model.safetensors.index.json` file.
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

// ── Model-ID parser (mirrors pull.rs) ────────────────────────────────────────

/// Parse `[hf.co/]owner/repo[:quant_tag]` into `(repo_id, Option<quant_tag>)`.
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

// ── Small utility ─────────────────────────────────────────────────────────────

fn model_dir_for(model_id: &str, cache_dir: &Path) -> Result<PathBuf> {
    let (repo_id, _) = parse_model_id(model_id);
    let cache = ModelCache::new(cache_dir)?;
    Ok(cache.model_dir(repo_id, "main"))
}
