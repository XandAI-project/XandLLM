use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config as LlamaConfig, Llama as CandleLlama};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, info};

use crate::{
    error::{CoreError, CoreResult},
    model::{GenerateInput, Model, ModelConfig, SamplingParams, Token},
    sampling::sample_token,
    tokenizer::Tokenizer,
};

/// HuggingFace `config.json` fields relevant to LLaMA/Mistral.
#[derive(Debug, Deserialize, Serialize)]
struct HfConfig {
    architectures: Option<Vec<String>>,
    hidden_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    vocab_size: usize,
    torch_dtype: Option<String>,
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

/// LLaMA / Mistral model wrapped for XandLLM.
pub struct LlamaModel {
    inner: CandleLlama,
    cache: Cache,
    tokenizer: Arc<Tokenizer>,
    chat_format: String,
    #[allow(dead_code)]
    config: LlamaConfig,
    device: Device,
    dtype: DType,
    max_sequence_length: usize,
}

impl std::fmt::Debug for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModel")
            .field("dtype", &self.dtype)
            .field("max_sequence_length", &self.max_sequence_length)
            .finish()
    }
}

impl LlamaModel {
    /// Return a cheap clone of the model's tokenizer Arc.
    pub fn tokenizer_arc(&self) -> Arc<Tokenizer> {
        Arc::clone(&self.tokenizer)
    }

    /// Architecture identifier for this model family.
    pub fn architecture(&self) -> &str {
        "llama"
    }

    /// The chat template format to use for this model.
    pub fn chat_format(&self) -> &str {
        &self.chat_format
    }

    fn load_weights(model_dir: &Path, device: &Device, dtype: DType) -> CoreResult<VarBuilder<'static>> {
        // Prefer sharded safetensors if present
        let index_path = model_dir.join("model.safetensors.index.json");
        if index_path.exists() {
            let index_str = std::fs::read_to_string(&index_path).map_err(CoreError::Io)?;
            let index: serde_json::Value = serde_json::from_str(&index_str)?;
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
            let shard_paths: Vec<_> = shards.iter().map(|s| model_dir.join(s)).collect();
            return Ok(unsafe {
                VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, device)?
            });
        }

        // Single safetensors file
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            return Ok(unsafe {
                VarBuilder::from_mmaped_safetensors(&[single], dtype, device)?
            });
        }

        Err(CoreError::Config {
            field: "model_dir".to_string(),
            reason: format!(
                "No safetensors weights found in {}",
                model_dir.display()
            ),
        })
    }
}

impl Model for LlamaModel {
    fn load(config: &ModelConfig, device: &Device) -> CoreResult<Self> {
        info!(
            repo_id = %config.repo_id,
            model_dir = %config.model_dir.display(),
            "Loading LLaMA/Mistral model"
        );

        // Parse HF config
        let hf_cfg_path = config.model_dir.join("config.json");
        let hf_cfg_str = std::fs::read_to_string(&hf_cfg_path).map_err(CoreError::Io)?;
        let hf_cfg: HfConfig = serde_json::from_str(&hf_cfg_str)?;

        let dtype = match hf_cfg.torch_dtype.as_deref() {
            Some("bfloat16") => DType::BF16,
            Some("float16") => DType::F16,
            _ => DType::F32,
        };

        let llama_config = hf_cfg.into_llama_config();
        debug!(config = ?llama_config, "Resolved LLaMA config");

        // Load tokenizer
        let tokenizer_path = config.model_dir.join("tokenizer.json");
        let tokenizer = Arc::new(Tokenizer::from_file(&tokenizer_path)?);

        // Detect which chat template format this model uses
        let chat_format = if tokenizer.token_id("<|im_start|>").is_some() {
            "chatml".to_string()
        } else if tokenizer.token_id("<|eot_id|>").is_some() {
            "llama3".to_string()
        } else {
            "llama2".to_string()
        };

        // Load weights
        let vb = Self::load_weights(&config.model_dir, device, dtype)?;

        // Build candle KV cache and model
        let cache = Cache::new(true, dtype, &llama_config, device)?;
        let inner = CandleLlama::load(vb, &llama_config)?;

        info!(chat_format = %chat_format, "Model loaded successfully");

        Ok(Self {
            inner,
            cache,
            tokenizer,
            chat_format,
            config: llama_config,
            device: device.clone(),
            dtype,
            max_sequence_length: config.max_sequence_length,
        })
    }

    fn generate(
        &mut self,
        input: GenerateInput,
        params: SamplingParams,
    ) -> CoreResult<mpsc::UnboundedReceiver<CoreResult<Token>>> {
        let seq_len = input.token_ids.len();
        if seq_len > self.max_sequence_length {
            return Err(CoreError::SequenceTooLong {
                got: seq_len,
                max: self.max_sequence_length,
            });
        }

        // Reset KV cache before every generation.
        // Without this, stale K/V data from the previous request leaks into
        // the attention computation of the new one — corrupting outputs.
        // Recreating Cache is cheap: it allocates a Vec of Nones.
        self.cache = Cache::new(true, self.dtype, &self.config, &self.device)?;

        let (tx, rx) = mpsc::unbounded_channel::<CoreResult<Token>>();

        let mut token_history: Vec<u32> = input.token_ids.clone();
        let max_new = params.max_new_tokens;

        // Prefill: forward pass on the full prompt
        let input_tensor = Tensor::new(token_history.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.inner.forward(&input_tensor, 0, &mut self.cache)?;

        // Sample first token from last position of logits
        let last_logits = logits.i((0, seq_len - 1))?;
        let next_id = sample_token(&last_logits, &params, &token_history)?;

        token_history.push(next_id);
        let is_eos = params.stop_token_ids.contains(&next_id);

        // Don't send stop tokens to output (synchronous generate)
        if is_eos {
            return Ok(rx);
        }

        let text = self.tokenizer.decode_token(next_id)?;
        let _ = tx.send(Ok(Token { id: next_id, text, is_eos: false }));

        if max_new <= 1 {
            return Ok(rx);
        }

        let mut pos = seq_len;

        // Autoregressive decode loop
        for _ in 1..max_new {
            let last_id = *token_history.last().unwrap();
            let input_step = Tensor::new(&[last_id], &self.device)?.unsqueeze(0)?;
            let logits_step = self.inner.forward(&input_step, pos, &mut self.cache)?;
            let step_logits = logits_step.i((0, 0))?;
            let next_id = sample_token(&step_logits, &params, &token_history)?;

            // Cap history to the penalty window to prevent unbounded growth.
            // token_history is only used for penalty computation, not for the
            // model's KV cache (which is managed separately).
            let cap = params.repeat_last_n.unwrap_or(usize::MAX);
            if token_history.len() >= cap {
                token_history.drain(..token_history.len() - cap + 1);
            }
            token_history.push(next_id);

            let is_eos = params.stop_token_ids.contains(&next_id);

            // Don't send stop tokens to output (synchronous generate)
            if is_eos {
                break;
            }

            let text = self.tokenizer.decode_token(next_id)?;
            if tx.send(Ok(Token { id: next_id, text, is_eos: false })).is_err() {
                break;
            }
            pos += 1;
        }

        Ok(rx)
    }

    fn generate_stream(
        &mut self,
        input: GenerateInput,
        params: SamplingParams,
        tx: mpsc::UnboundedSender<CoreResult<Token>>,
    ) -> CoreResult<()> {
        let seq_len = input.token_ids.len();
        if seq_len > self.max_sequence_length {
            return Err(CoreError::SequenceTooLong {
                got: seq_len,
                max: self.max_sequence_length,
            });
        }

        // Reset KV cache before every generation (same reason as in `generate`).
        self.cache = Cache::new(true, self.dtype, &self.config, &self.device)?;

        let mut token_history: Vec<u32> = input.token_ids.clone();
        let max_new = params.max_new_tokens;

        // Prefill: forward pass on the full prompt
        let input_tensor = Tensor::new(token_history.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.inner.forward(&input_tensor, 0, &mut self.cache)?;

        // Sample first token from last position of logits
        let last_logits = logits.i((0, seq_len - 1))?;
        let next_id = sample_token(&last_logits, &params, &token_history)?;

        token_history.push(next_id);
        let is_eos = params.stop_token_ids.contains(&next_id);

        // Don't send stop tokens to output (streaming generate)
        if is_eos {
            return Ok(());
        }

        let text = self.tokenizer.decode_token(next_id)?;
        if tx.send(Ok(Token { id: next_id, text, is_eos: false })).is_err() {
            return Ok(());
        }

        if max_new <= 1 {
            return Ok(());
        }

        let mut pos = seq_len;

        // Autoregressive decode loop
        for _ in 1..max_new {
            let last_id = *token_history.last().unwrap();
            let input_step = Tensor::new(&[last_id], &self.device)?.unsqueeze(0)?;
            let logits_step = self.inner.forward(&input_step, pos, &mut self.cache)?;
            let step_logits = logits_step.i((0, 0))?;
            let next_id = sample_token(&step_logits, &params, &token_history)?;

            // Cap history to the penalty window — same reason as generate().
            let cap = params.repeat_last_n.unwrap_or(usize::MAX);
            if token_history.len() >= cap {
                token_history.drain(..token_history.len() - cap + 1);
            }
            token_history.push(next_id);

            let is_eos = params.stop_token_ids.contains(&next_id);

            // Don't send stop tokens to output (streaming generate)
            if is_eos {
                break;
            }

            let text = self.tokenizer.decode_token(next_id)?;
            if tx.send(Ok(Token { id: next_id, text, is_eos: false })).is_err() {
                break;
            }
            pos += 1;
        }

        Ok(())
    }
}
