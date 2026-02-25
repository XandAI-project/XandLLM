//! Teacher model wrapper.
//!
//! Wraps [`AnyModel`] so the distiller can generate completions from a teacher
//! without caring whether it is GGUF-quantized or safetensors.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;

use candle_core::Device;
use xandllm_core::{
    chat_template,
    AnyModel,
    GenerateInput,
    Model,
    ModelConfig,
    SamplingParams,
    Tokenizer,
};
use xandllm_hub::ModelCache;

/// Wraps a loaded teacher model and exposes a simple `generate_completion` API.
pub struct Teacher {
    model: AnyModel,
    tokenizer: Arc<Tokenizer>,
    chat_format: String,
    max_sequence_length: usize,
}

impl Teacher {
    /// Load a teacher from the local model cache.
    ///
    /// `model_id` accepts the same formats as `xandllm pull`:
    /// - `owner/repo`
    /// - `owner/repo:Q4_0`  (quant tag is stripped before building the cache path)
    /// - `hf.co/owner/repo:Q4_0`
    pub fn load(
        model_id: &str,
        cache_dir: &Path,
        prefer_gpu: bool,
        cuda_device_id: usize,
        max_sequence_length: usize,
    ) -> Result<Self> {
        // Strip optional quant tag (e.g. "repo:Q4_0" → "repo") before building
        // the cache path — the tag is never part of the on-disk directory name.
        let repo_id = strip_quant_tag(model_id);

        let cache = ModelCache::new(cache_dir)?;
        let model_dir = cache.model_dir(repo_id, "main");

        info!(
            model_id,
            model_dir = %model_dir.display(),
            "Loading teacher model"
        );

        // Teacher only runs forward inference — it never receives gradients.
        // Keeping it on CPU frees the full GPU memory budget for student training.
        // The 64 GB RAM easily holds even a 7B Q4_0 GGUF (~4.3 GB).
        let _ = (prefer_gpu, cuda_device_id); // params kept for API compatibility
        let device = Device::Cpu;

        let model_config = ModelConfig {
            repo_id: model_id.to_string(),
            revision: "main".to_string(),
            model_dir,
            max_sequence_length,
        };

        let model = AnyModel::load(&model_config, &device)
            .with_context(|| format!("Failed to load teacher model '{model_id}'"))?;

        let tokenizer = model.tokenizer_arc();
        let chat_format = model.chat_format().to_string();

        info!(
            arch = model.architecture(),
            chat_format = %chat_format,
            "Teacher model loaded"
        );

        Ok(Self { model, tokenizer, chat_format, max_sequence_length })
    }

    /// Return the teacher's tokenizer (shared reference).
    pub fn tokenizer(&self) -> Arc<Tokenizer> {
        Arc::clone(&self.tokenizer)
    }

    /// Return the chat format string (e.g. `"chatml"`, `"llama3"`).
    pub fn chat_format(&self) -> &str {
        &self.chat_format
    }

    /// Return the teacher's vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// Generate a completion for `prompt` using the teacher model.
    ///
    /// `prompt` should already be formatted with the appropriate chat template
    /// (or be a raw prompt for base models).  The generated text is returned
    /// as a `String` with stop tokens stripped.
    pub fn generate_completion(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<String> {
        let token_ids = self.tokenizer.encode(prompt, false)
            .with_context(|| "Teacher tokenisation failed")?;

        // Sanity: truncate prompt if it would exceed the context window.
        let prompt_len = token_ids.len();
        let available = self.max_sequence_length.saturating_sub(prompt_len);
        let max_tokens = max_new_tokens.min(available.max(1));

        // Collect stop token IDs for this model's chat format.
        let stop_token_ids: Vec<u32> = chat_template::stop_token_strings_for_format(&self.chat_format)
            .iter()
            .filter_map(|s| self.tokenizer.token_id(s))
            .chain(self.tokenizer.eos_token_id())
            .collect();

        let params = SamplingParams {
            max_new_tokens: max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            greedy: false,
            stop_token_ids,
            ..SamplingParams::default()
        };

        let input = GenerateInput { token_ids };

        // `generate` fills the channel synchronously — no spawn_blocking needed
        // here because the distiller drives generation in a blocking context.
        let mut rx = self.model.generate(input, params)
            .context("Teacher generation failed")?;

        let mut output = String::new();
        // Drain the channel (it's already filled when generate() returns)
        while let Ok(result) = rx.try_recv() {
            let token = result.context("Error receiving teacher token")?;
            if token.is_eos {
                break;
            }
            output.push_str(&token.text);
        }

        Ok(output)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Strip an optional quantisation tag from a model id.
///
/// `"owner/repo:Q4_0"` → `"owner/repo"`
/// `"owner/repo"`      → `"owner/repo"` (unchanged)
/// `"hf.co/owner/repo:Q4_0"` → `"owner/repo"`
fn strip_quant_tag(model_id: &str) -> &str {
    let s = model_id.strip_prefix("hf.co/").unwrap_or(model_id);
    if let Some(colon) = s.rfind(':') {
        let tag = &s[colon + 1..];
        // Only treat as quant tag if there are no path separators after ':'
        if !tag.is_empty() && !tag.contains('/') && !tag.contains('\\') {
            return &s[..colon];
        }
    }
    s
}
