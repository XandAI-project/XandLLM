//! Model export utilities.
//!
//! After training the student, this module writes the weights and metadata to
//! a directory that can be used directly with `xandllm serve` or converted to
//! GGUF for quantised inference.
//!
//! ## SafeTensors output
//!
//! Produces:
//! ```text
//! <output_dir>/
//!   model.safetensors   — all trained weights
//!   config.json         — HuggingFace-compatible architecture config
//!   tokenizer.json      — copied from the teacher model directory
//!   tokenizer_config.json — (optional) copied from teacher directory
//! ```
//!
//! ## GGUF output
//!
//! First writes the SafeTensors output, then attempts to convert + quantise
//! via `llama.cpp`'s `convert_hf_to_gguf.py` / `llama-quantize` binaries.
//! If neither is found on PATH the command prints clear instructions.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_transformers::models::llama::Config as LlamaConfig;
use serde::Serialize;
use tracing::{info, warn};

use xandllm_core::Tokenizer;

use crate::student::TrainableStudent;

// ── OutputFormat ──────────────────────────────────────────────────────────────

/// The file format the distilled model should be saved in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Standard HuggingFace SafeTensors + `config.json`.
    SafeTensors,
    /// GGUF quantised format (requires `llama.cpp` on PATH).
    Gguf,
}

impl OutputFormat {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "safetensor" | "safetensors" => Ok(Self::SafeTensors),
            "gguf" => Ok(Self::Gguf),
            other => bail!("Unknown output format '{}'. Valid values: safetensor, gguf.", other),
        }
    }
}

// ── HuggingFace config.json ───────────────────────────────────────────────────

/// A minimal `config.json` that `transformers` and `xandllm serve` can read.
#[derive(Serialize)]
struct HfConfigJson {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    model_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    _name_or_path: Option<String>,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    torch_dtype: String,
    vocab_size: usize,
}

impl HfConfigJson {
    fn from_llama_config(cfg: &LlamaConfig, name: Option<&str>) -> Self {
        Self {
            architectures: vec!["LlamaForCausalLM".to_string()],
            hidden_size: cfg.hidden_size,
            intermediate_size: cfg.intermediate_size,
            max_position_embeddings: cfg.max_position_embeddings,
            model_type: "llama".to_string(),
            model_name: name.map(str::to_owned),
            _name_or_path: name.map(str::to_owned),
            num_attention_heads: cfg.num_attention_heads,
            num_hidden_layers: cfg.num_hidden_layers,
            num_key_value_heads: cfg.num_key_value_heads,
            rms_norm_eps: cfg.rms_norm_eps,
            rope_theta: cfg.rope_theta as f64,
            torch_dtype: "float32".to_string(),
            vocab_size: cfg.vocab_size,
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Export the trained `student` to `output_dir` in the requested `format`.
///
/// `teacher_model_dir` is the directory of the teacher model; tokenizer files
/// are copied from there so the student can be loaded by `xandllm serve`.
pub fn export(
    student: &TrainableStudent,
    _tokenizer: &Arc<Tokenizer>,
    teacher_model_dir: &Path,
    output_dir: &Path,
    format: OutputFormat,
    model_name: Option<&str>,
) -> Result<()> {
    // Always save SafeTensors first — GGUF conversion builds on top of it.
    export_safetensors(student, teacher_model_dir, output_dir, model_name)?;

    if format == OutputFormat::Gguf {
        export_gguf(output_dir)?;
    }

    Ok(())
}

// ── SafeTensors export ────────────────────────────────────────────────────────

fn export_safetensors(
    student: &TrainableStudent,
    teacher_model_dir: &Path,
    output_dir: &Path,
    model_name: Option<&str>,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Cannot create output dir: {}", output_dir.display()))?;

    // ── Weights ───────────────────────────────────────────────────────────────
    let weights_path = output_dir.join("model.safetensors");
    student.save(&weights_path)?;
    info!(path = %weights_path.display(), "Student weights saved");

    // ── config.json ───────────────────────────────────────────────────────────
    let hf_cfg = HfConfigJson::from_llama_config(student.llama_config(), model_name);
    let config_json = serde_json::to_string_pretty(&hf_cfg)?;
    let config_path = output_dir.join("config.json");
    std::fs::write(&config_path, config_json)
        .with_context(|| format!("Cannot write {}", config_path.display()))?;
    info!(path = %config_path.display(), "config.json written");

    // ── Tokenizer files ───────────────────────────────────────────────────────
    copy_tokenizer_files(teacher_model_dir, output_dir)?;

    info!(output_dir = %output_dir.display(), "SafeTensors export complete");
    Ok(())
}

/// Copy tokenizer-related files from the teacher's model directory so the
/// student can be loaded by `xandllm serve` without manual steps.
fn copy_tokenizer_files(src_dir: &Path, dst_dir: &Path) -> Result<()> {
    // These are the standard HuggingFace tokenizer artifacts
    let candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ];

    for name in &candidates {
        let src = src_dir.join(name);
        if src.exists() {
            let dst = dst_dir.join(name);
            std::fs::copy(&src, &dst)
                .with_context(|| format!("Failed to copy {name}"))?;
            info!(file = name, "Tokenizer file copied");
        }
    }

    Ok(())
}

// ── GGUF conversion ───────────────────────────────────────────────────────────

/// Attempt to convert the SafeTensors model in `dir` to GGUF using
/// `llama.cpp`'s converter.
///
/// Looks for the converter in the following order:
/// 1. `convert_hf_to_gguf.py` on PATH (run with Python 3)
/// 2. `convert-hf-to-gguf` CLI binary on PATH
///
/// After conversion, runs `llama-quantize` for Q4_0 quantisation if it is
/// available on PATH.  If no converter is found, prints installation
/// instructions and returns an error.
fn export_gguf(output_dir: &Path) -> Result<()> {
    // Try Python-based converter first
    if try_python_converter(output_dir)? {
        try_quantize(output_dir);
        return Ok(());
    }

    // Try CLI binary
    if try_cli_converter(output_dir)? {
        try_quantize(output_dir);
        return Ok(());
    }

    bail!(
        "GGUF conversion requires llama.cpp to be installed.\n\
         Please install it and ensure one of the following is on your PATH:\n\
         - convert_hf_to_gguf.py  (from https://github.com/ggerganov/llama.cpp)\n\
         - convert-hf-to-gguf     (llama.cpp CLI binary)\n\
         \n\
         Alternatively, load the SafeTensors output directly:\n\
         xandllm serve --model {dir}\n",
        dir = output_dir.display()
    )
}

fn try_python_converter(output_dir: &Path) -> Result<bool> {
    // Check if convert_hf_to_gguf.py is reachable
    let check = Command::new("python3").arg("-c").arg("import sys").output();
    if check.is_err() {
        return Ok(false);
    }

    let script = which_on_path("convert_hf_to_gguf.py");
    let Some(script) = script else {
        return Ok(false);
    };

    info!("Running convert_hf_to_gguf.py …");
    let gguf_out = output_dir.join("model.gguf");
    let status = Command::new("python3")
        .arg(&script)
        .arg(output_dir)
        .arg("--outfile")
        .arg(&gguf_out)
        .arg("--outtype")
        .arg("f16")
        .status()
        .context("Failed to run convert_hf_to_gguf.py")?;

    if !status.success() {
        bail!("convert_hf_to_gguf.py exited with {status}");
    }

    info!(path = %gguf_out.display(), "GGUF conversion complete");
    Ok(true)
}

fn try_cli_converter(output_dir: &Path) -> Result<bool> {
    if which_on_path("convert-hf-to-gguf").is_none() {
        return Ok(false);
    }

    info!("Running convert-hf-to-gguf …");
    let gguf_out = output_dir.join("model.gguf");
    let status = Command::new("convert-hf-to-gguf")
        .arg(output_dir)
        .arg("--outfile")
        .arg(&gguf_out)
        .arg("--outtype")
        .arg("f16")
        .status()
        .context("Failed to run convert-hf-to-gguf")?;

    if !status.success() {
        bail!("convert-hf-to-gguf exited with {status}");
    }

    info!(path = %gguf_out.display(), "GGUF conversion complete");
    Ok(true)
}

/// Run `llama-quantize` to produce a Q4_0 quantised GGUF from the F16 base.
/// Logs a warning instead of failing if `llama-quantize` is not found.
fn try_quantize(output_dir: &Path) {
    let f16 = output_dir.join("model.gguf");
    let q4 = output_dir.join("model-q4_0.gguf");

    if which_on_path("llama-quantize").is_none() {
        warn!(
            "llama-quantize not found on PATH; skipping Q4_0 quantisation.\n\
             Run manually: llama-quantize {f16} {q4} Q4_0",
            f16 = f16.display(),
            q4 = q4.display()
        );
        return;
    }

    info!("Running llama-quantize Q4_0 …");
    match Command::new("llama-quantize")
        .arg(&f16)
        .arg(&q4)
        .arg("Q4_0")
        .status()
    {
        Ok(s) if s.success() => info!(path = %q4.display(), "Q4_0 quantisation complete"),
        Ok(s) => warn!("llama-quantize exited with {s}"),
        Err(e) => warn!(error = %e, "llama-quantize failed"),
    }
}

// ── PATH search helper ────────────────────────────────────────────────────────

fn which_on_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let candidate = dir.join(name);
            if candidate.exists() { Some(candidate) } else { None }
        })
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_output_format() {
        assert_eq!(OutputFormat::parse("safetensor").unwrap(), OutputFormat::SafeTensors);
        assert_eq!(OutputFormat::parse("safetensors").unwrap(), OutputFormat::SafeTensors);
        assert_eq!(OutputFormat::parse("gguf").unwrap(), OutputFormat::Gguf);
        assert!(OutputFormat::parse("bin").is_err());
    }
}
