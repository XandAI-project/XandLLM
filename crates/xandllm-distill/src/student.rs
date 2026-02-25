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
use candle_core::{DType, Device, IndexOp, Tensor};
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
    fn build(
        config: LlamaConfig,
        tokenizer: Arc<Tokenizer>,
        device: &Device,
        weight_paths: &[std::path::PathBuf],
    ) -> Result<Self> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

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
        let cache = Cache::new(false, DType::F32, &config, device)
            .context("Failed to create training cache")?;

        Ok(Self { model, varmap, cache, config, tokenizer, device: device.clone() })
    }

    // ── Forward pass ──────────────────────────────────────────────────────────

    /// Run a forward pass and return the full logit tensor.
    ///
    /// `input_ids` shape: `[batch, seq_len]`.
    /// Returns flattened shape: `[batch * seq_len, vocab_size]`.
    ///
    /// Each sample in the batch is processed independently (the Candle Llama
    /// model does not natively support batched training), then stacked.
    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch, seq_len) = input_ids.dims2()
            .context("Expected 2-D input_ids [batch, seq_len]")?;

        let mut per_sample: Vec<Tensor> = Vec::with_capacity(batch);
        for b in 0..batch {
            let ids = input_ids.i(b)?;     // [seq_len]
            let ids = ids.unsqueeze(0)?;   // [1, seq_len]
            // Position 0: we process the full sequence in one shot (no KV cache).
            let logits = self.model.forward(&ids, 0, &mut self.cache)?; // [1, seq_len, vocab]
            let logits = logits.squeeze(0)?; // [seq_len, vocab]
            per_sample.push(logits);
        }

        // Stack → [batch, seq_len, vocab], then flatten → [batch*seq_len, vocab]
        let stacked = Tensor::stack(&per_sample, 0)?;
        let vocab = stacked.dim(2)?;
        stacked.reshape((batch * seq_len, vocab))
            .context("reshape failed in student forward")
    }

    // ── Loss ──────────────────────────────────────────────────────────────────

    /// Compute cross-entropy loss masked to completion tokens only.
    ///
    /// * `logits` — `[N, vocab_size]` (already flattened)
    /// * `labels` — `[batch, seq_len]` next-token ids (u32)
    /// * `mask`   — `[batch, seq_len]` 1.0 for completion tokens, 0.0 for prompt
    ///
    /// Returns the mean cross-entropy over the masked (completion) positions.
    pub fn masked_ce_loss(
        logits: &Tensor,
        labels: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let device = logits.device();
        let (batch, seq) = labels.dims2()?;
        let n = batch * seq;

        let labels_flat = labels.reshape(n)?.to_dtype(DType::U32)?;
        let mask_flat = mask.reshape(n)?.to_dtype(DType::F32)?;

        // Per-position cross-entropy → shape [n]
        let ce = loss::cross_entropy(logits, &labels_flat)
            .context("cross_entropy computation failed")?;

        // Zero-out prompt positions
        let masked = (ce * &mask_flat)?;

        // Mean over completion tokens; guard against empty mask
        let denom = mask_flat.sum_all()?.to_scalar::<f32>()?;
        if denom < 1e-6 {
            Tensor::zeros((), DType::F32, device).context("zero-loss tensor")
        } else {
            let total = masked.sum_all()?;
            let denom_t = Tensor::new(denom, device)?;
            (total / denom_t).context("loss mean computation failed")
        }
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
