//! Teacher model wrapper.
//!
//! Wraps [`AnyModel`] so the distiller can generate completions from a teacher
//! without caring whether it is GGUF-quantized or safetensors.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::{info, warn};

use candle_core::Device;
use xandllm_core::{
    chat_template,
    select_device,
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
    stop_token_ids: Vec<u32>,
}

impl Teacher {
    /// Load a teacher from the local model cache.
    ///
    /// `model_id` accepts the same formats as `xandllm pull`:
    /// - `owner/repo`
    /// - `owner/repo:Q4_0`  (quant tag is stripped before building the cache path)
    /// - `hf.co/owner/repo:Q4_0`
    ///
    /// The teacher is placed on GPU when there is enough free VRAM (with a 1 GB
    /// headroom buffer).  Otherwise it falls back to CPU.  This is decided
    /// automatically by querying `nvidia-smi`.
    pub fn load(
        model_id: &str,
        cache_dir: &Path,
        prefer_gpu: bool,
        cuda_device_id: usize,
        max_sequence_length: usize,
    ) -> Result<Self> {
        let repo_id = strip_quant_tag(model_id);

        let cache = ModelCache::new(cache_dir)?;
        let model_dir = cache.model_dir(repo_id, "main");

        info!(
            model_id,
            model_dir = %model_dir.display(),
            "Loading teacher model"
        );

        let device = choose_teacher_device(prefer_gpu, cuda_device_id, &model_dir)?;

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

        let stop_token_ids: Vec<u32> = chat_template::stop_token_strings_for_format(&chat_format)
            .iter()
            .filter_map(|s| tokenizer.token_id(s))
            .chain(tokenizer.eos_token_id())
            .collect();

        info!(
            arch = model.architecture(),
            chat_format = %chat_format,
            device = if matches!(device, Device::Cpu) { "CPU" } else { "GPU" },
            "Teacher model loaded"
        );

        Ok(Self { model, tokenizer, chat_format, max_sequence_length, stop_token_ids })
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
    /// Uses greedy decoding (argmax) for maximum throughput — diversity is not
    /// needed for distillation targets.
    pub fn generate_completion(
        &mut self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<(String, usize)> {
        let token_ids = self.tokenizer.encode(prompt, false)
            .with_context(|| "Teacher tokenisation failed")?;

        let prompt_len = token_ids.len();
        let available = self.max_sequence_length.saturating_sub(prompt_len);
        let max_tokens = max_new_tokens.min(available.max(1));

        let params = SamplingParams {
            max_new_tokens: max_tokens,
            greedy: true,
            stop_token_ids: self.stop_token_ids.clone(),
            ..SamplingParams::default()
        };

        let input = GenerateInput { token_ids };

        let mut rx = self.model.generate(input, params)
            .context("Teacher generation failed")?;

        let mut output = String::new();
        let mut generated_tokens = 0usize;
        while let Ok(result) = rx.try_recv() {
            let token = result.context("Error receiving teacher token")?;
            if token.is_eos {
                break;
            }
            output.push_str(&token.text);
            generated_tokens += 1;
        }

        Ok((output, generated_tokens))
    }
}

// ── Device selection ─────────────────────────────────────────────────────────

/// Pick the best device for the teacher model.
///
/// When `prefer_gpu` is true and CUDA is compiled in, queries `nvidia-smi` for
/// free VRAM.  If there is enough room for the model files plus a 1 GB buffer,
/// the teacher is placed on GPU (100-200 tok/s for GGUF).  Otherwise it falls
/// back to CPU.
fn choose_teacher_device(
    prefer_gpu: bool,
    cuda_device_id: usize,
    model_dir: &Path,
) -> Result<Device> {
    if !prefer_gpu {
        info!("Teacher: GPU not requested — using CPU");
        return Ok(Device::Cpu);
    }

    // Try to get a CUDA device first; if the feature is off this returns CPU.
    let device = select_device(true, cuda_device_id)?;
    if matches!(device, Device::Cpu) {
        return Ok(Device::Cpu);
    }

    let free_mb = match query_free_vram_mb() {
        Some(v) => v,
        None => {
            info!("Teacher: cannot query VRAM — loading on GPU optimistically");
            return Ok(device);
        }
    };

    let model_mb = estimate_model_size_mb(model_dir);
    // Require: model + 1 GB KV-cache headroom + 4 GB spare for student init/training.
    // Without the spare, teacher fills VRAM and student construction OOMs even
    // though the error looks unrelated.
    let needed_mb = model_mb + 1024 + 4096;

    if free_mb >= needed_mb {
        info!(
            free_vram_mb = free_mb,
            model_mb,
            needed_mb,
            "Teacher: enough VRAM — loading on GPU"
        );
        Ok(device)
    } else {
        info!(
            free_vram_mb = free_mb,
            needed_mb,
            "Teacher: insufficient VRAM — loading on CPU"
        );
        Ok(Device::Cpu)
    }
}

/// Query free VRAM in MiB via `nvidia-smi`.
///
/// Returns `None` if the command is not found or produces unexpected output.
fn query_free_vram_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    text.lines()
        .next()?
        .trim()
        .parse::<u64>()
        .ok()
}

/// Sum the sizes of model weight files (`.gguf`, `.safetensors`) in `dir`.
///
/// Returns the total in MiB.  Falls back to 0 if the directory is unreadable.
fn estimate_model_size_mb(dir: &Path) -> u64 {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    let total_bytes: u64 = entries
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let s = name.to_string_lossy();
            s.ends_with(".gguf") || s.ends_with(".safetensors")
        })
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum();

    total_bytes / (1024 * 1024)
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
        if !tag.is_empty() && !tag.contains('/') && !tag.contains('\\') {
            return &s[..colon];
        }
    }
    s
}
