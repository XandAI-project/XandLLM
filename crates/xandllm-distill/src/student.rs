//! Trainable student model.
//!
//! The student is a standard Llama-family transformer loaded into a
//! [`candle_nn::VarMap`] so its parameters can receive gradients and be
//! updated by an optimiser.
//!
//! ## Two construction modes
//!
//! * **`from_preset`** — create a fresh model with randomly-initialised weights
//!   based on a [`SizePreset`].
//! * **`from_safetensors`** — load an existing smaller model from a
//!   `model.safetensors` file into the `VarMap` and fine-tune it.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, VarBuilder, VarMap};
use candle_transformers::models::llama::{
    Cache, Config as LlamaConfig, Llama as CandleLlama,
};
use tracing::info;

use xandllm_core::Tokenizer;

use crate::presets::SizePreset;

// ── TrainableStudent ──────────────────────────────────────────────────────────

/// A Llama-family student model whose parameters are held in a [`VarMap`]
/// so they can be trained with gradient descent.
pub struct TrainableStudent {
    pub(crate) model: CandleLlama,
    pub(crate) varmap: VarMap,
    /// Non-caching KV-cache (use_kv_cache=false): reused across training steps.
    cache: Cache,
    pub(crate) config: LlamaConfig,
    #[allow(dead_code)]
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) device: Device,
}

impl TrainableStudent {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Create a student with **random weights** matching `preset`.
    ///
    /// `vocab_size` must equal the teacher's vocabulary size so the student's
    /// embedding matrix and LM head have compatible dimensions.
    pub fn from_preset(
        preset: &SizePreset,
        vocab_size: usize,
        tokenizer: Arc<Tokenizer>,
        device: &Device,
    ) -> Result<Self> {
        let config = preset.llama_config(vocab_size);
        info!(
            preset = preset.label(),
            params = preset.approx_params(),
            vocab_size,
            "Initialising student from size preset (random weights)"
        );
        Self::build(config, tokenizer, device, &[])
    }

    /// Load an **existing smaller model** from a local directory into the
    /// `VarMap` for fine-tuning.
    ///
    /// The directory must contain either a single `model.safetensors` file or
    /// sharded files referenced by `model.safetensors.index.json`, plus a
    /// `config.json` in HuggingFace format.
    pub fn from_safetensors(
        model_dir: &Path,
        tokenizer: Arc<Tokenizer>,
        device: &Device,
    ) -> Result<Self> {
        let config = read_hf_config(model_dir)?;

        info!(
            model_dir = %model_dir.display(),
            num_hidden_layers = config.num_hidden_layers,
            hidden_size = config.hidden_size,
            vocab_size = config.vocab_size,
            "Loading student base model from safetensors"
        );

        let shard_paths = collect_safetensor_paths(model_dir)?;
        Self::build(config, tokenizer, device, &shard_paths)
    }

    // ── Internal builder ──────────────────────────────────────────────────────

    /// Shared construction path.
    ///
    /// 1. Create a `VarMap`.
    /// 2. Build the Llama model from the VarMap — this registers all parameter
    ///    names (with random initial values).
    /// 3. If `weight_paths` is non-empty, call `varmap.load(path)` for each
    ///    shard to overwrite the random values with pre-trained weights.
    ///
    /// ## Dtype strategy (mixed-precision lite)
    ///
    /// | Device | Weights dtype | Memory savings vs F32 |
    /// |--------|---------------|-----------------------|
    /// | CPU    | F32           | — (BF16 not supported on CPU) |
    /// | CUDA   | BF16          | ~50 % (1B: 4.4 GB → 2.2 GB) |
    ///
    /// The forward pass runs in BF16 on GPU.  The cross-entropy loss is cast to
    /// F32 before computation (`train_step`) to preserve numerical accuracy
    /// in the probability distribution — this is the standard "mixed-precision
    /// lite" approach used by llama.cpp and transformers training scripts.
    fn build(
        config: LlamaConfig,
        tokenizer: Arc<Tokenizer>,
        device: &Device,
        weight_paths: &[std::path::PathBuf],
    ) -> Result<Self> {
        // Use BF16 on CUDA for ~50 % memory savings; fall back to F32 on CPU
        // (candle does not support BF16 operations on CPU).
        let dtype = match device {
            Device::Cuda(_) => DType::BF16,
            _ => DType::F32,
        };

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);

        let model = CandleLlama::load(vb, &config)
            .context("Failed to construct student Llama model")?;

        // Overwrite random initialisation with pre-trained weights.
        // VarMap::load updates only keys that already exist in the map, so
        // any extra keys in the file are silently ignored.
        for path in weight_paths {
            varmap.load(path)
                .with_context(|| format!("Failed to load weights from {}", path.display()))?;
        }

        if !weight_paths.is_empty() {
            info!(shards = weight_paths.len(), "Pre-trained weights loaded into student VarMap");
        }

        // A non-caching Cache is reused across all training forward passes.
        // With use_kv_cache=false the Cache is never written, so reuse is safe.
        // Use the same dtype as the model weights.
        let cache = Cache::new(false, dtype, &config, device)
            .context("Failed to create training cache")?;

        Ok(Self { model, varmap, cache, config, tokenizer, device: device.clone() })
    }

    // ── Training step ─────────────────────────────────────────────────────────

    /// Compute the distillation loss for one batch and return a scalar tensor
    /// suitable for calling `.backward()` on.
    ///
    /// ## Why not a plain `forward()` returning full-sequence logits?
    ///
    /// `candle_transformers::Llama::forward` is built for *inference*: it
    /// slices the hidden states to the **last position** before the LM head,
    /// always returning `[batch, vocab]` regardless of input length.  There is
    /// no public API to recover per-position logits from the built-in model.
    ///
    /// ## What this does instead — multi-position teacher forcing
    ///
    /// For every sample we run **two** forward passes:
    ///
    /// 1. **Full sequence** (`input_ids`): model sees `[prompt ++ completion[:-1]]`
    ///    and predicts the **last completion token** — the most context-rich signal.
    ///
    /// 2. **Prompt only** (`input_ids[:, :prompt_end]`): model sees only the
    ///    prompt and predicts the **first completion token** — the hardest and
    ///    most informative step for generation quality.
    ///
    /// Both losses are averaged, giving two gradient signals per example with
    /// the same number of backward passes as a single-position approach.
    ///
    /// `prompt_lengths`: number of prompt tokens per sample (before completion).
    /// `last_labels`:    last-token label per sample (shape `[batch]`).
    /// `first_labels`:   first completion label per sample (shape `[batch]`).
    pub fn train_step(
        &mut self,
        input_ids: &Tensor,
        prompt_lengths: &[usize],
        last_labels: &[u32],
        first_labels: &[u32],
    ) -> Result<Tensor> {
        let (_batch, _seq_len) = input_ids.dims2()
            .context("Expected 2-D input_ids [batch, seq_len]")?;
        let device = &self.device.clone();

        // ── Loss 1: full-sequence prediction of last completion token ─────────
        let logits_full = self.model.forward(input_ids, 0, &mut self.cache)?;
        // logits_full: [batch, vocab]
        let logits_f32 = to_f32(&logits_full)?;
        let labels_last = Tensor::new(last_labels, device)?.to_dtype(DType::U32)?;
        let loss_last = loss::cross_entropy(&logits_f32, &labels_last)
            .context("cross_entropy (last token) failed")?
            .mean_all()
            .context("mean (last token) failed")?;

        // ── Loss 2: prompt-only prediction of first completion token ──────────
        // Build the shortest prefix across the batch (minimum prompt length).
        // This still covers all samples because we feed only up to prompt_end.
        let min_prompt = *prompt_lengths.iter().min().unwrap_or(&1);
        let prompt_only = input_ids.narrow(1, 0, min_prompt.max(1))?;

        let logits_prompt = self.model.forward(&prompt_only, 0, &mut self.cache)?;
        let logits_prompt_f32 = to_f32(&logits_prompt)?;
        let labels_first = Tensor::new(first_labels, device)?.to_dtype(DType::U32)?;
        let loss_first = loss::cross_entropy(&logits_prompt_f32, &labels_first)
            .context("cross_entropy (first token) failed")?
            .mean_all()
            .context("mean (first token) failed")?;

        // Average both losses — equal weight.
        ((loss_last + loss_first)? / 2.0).context("loss averaging failed")
    }

    // ── Forward (kept for export/evaluation — not used in training) ───────────

    /// Run one forward pass on `input_ids` and return `[batch, vocab]` logits
    /// for the **last position** of each sequence.
    ///
    /// > **Note:** `candle_transformers::Llama` only exposes last-token logits.
    /// > Use [`train_step`] for gradient-based training.
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids, 0, &mut self.cache)
            .context("Student forward pass failed")
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Return references to all trainable variables for the optimiser.
    pub fn trainable_vars(&self) -> Vec<candle_core::Var> {
        self.varmap.all_vars()
    }

    /// The student's vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// The student's Llama configuration (needed by the exporter).
    pub fn llama_config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Save all parameters to a SafeTensors file at `path`.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.varmap.save(path)
            .with_context(|| format!("Failed to save student weights to {}", path.display()))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Cast `t` to F32 if it is not already; no-op if already F32.
///
/// Cross-entropy / softmax over large vocabularies is numerically unstable in
/// BF16 — exp() overflows easily.  Always computing the loss in F32 is the
/// standard "mixed-precision lite" practice (used by llama.cpp, Transformers,
/// and nanoGPT) and adds negligible overhead.
fn to_f32(t: &Tensor) -> Result<Tensor> {
    if t.dtype() == DType::F32 {
        Ok(t.clone())
    } else {
        t.to_dtype(DType::F32).context("dtype cast to F32 failed")
    }
}

fn read_hf_config(model_dir: &Path) -> Result<LlamaConfig> {
    let path = model_dir.join("config.json");
    let json = std::fs::read_to_string(&path)
        .with_context(|| format!("Cannot read {}", path.display()))?;
    let hf: HfConfig = serde_json::from_str(&json)
        .with_context(|| format!("Cannot parse {}", path.display()))?;
    Ok(hf.into_llama_config())
}

fn collect_safetensor_paths(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)
            .with_context(|| format!("Cannot read {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&json)?;
        let mut shards: Vec<String> = index["weight_map"]
            .as_object()
            .map(|m| {
                m.values()
                    .filter_map(|v| v.as_str())
                    .map(String::from)
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect()
            })
            .unwrap_or_default();
        shards.sort();
        return Ok(shards.iter().map(|s| dir.join(s)).collect());
    }

    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    anyhow::bail!("No safetensors weights found in {}", dir.display());
}

// ── HF config deserialization (mirrors xandllm-core's private struct) ─────────

#[derive(serde::Deserialize)]
struct HfConfig {
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    vocab_size: usize,
}

impl HfConfig {
    fn into_llama_config(self) -> LlamaConfig {
        LlamaConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-5),
            rope_theta: self.rope_theta.unwrap_or(10_000.0) as f32,
            use_flash_attn: false,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: false,
        }
    }
}
