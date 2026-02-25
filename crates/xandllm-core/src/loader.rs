use std::sync::Arc;

use candle_core::Device;
use tokio::sync::mpsc;

use crate::{
    error::{CoreError, CoreResult},
    llama::LlamaModel,
    model::{GenerateInput, Model, ModelConfig, SamplingParams, Token},
    quantized::QuantizedModel,
    tokenizer::Tokenizer,
};

/// Auto-detects whether a model directory contains GGUF weights or safetensors,
/// and loads the appropriate backend.
///
/// Decision rules (checked in order):
/// 1. If any `.gguf` file is present → `QuantizedModel`
/// 2. If `model.safetensors` or `model.safetensors.index.json` is present → `LlamaModel`
/// 3. Error
pub enum AnyModel {
    Llama(LlamaModel),
    Quantized(QuantizedModel),
}

impl AnyModel {
    /// Return the model's tokenizer as a cheap Arc clone.
    ///
    /// Call this before moving the model into an `Arc<Mutex<dyn Model>>` so the
    /// tokenizer can be shared independently (e.g. passed to the API server).
    pub fn tokenizer_arc(&self) -> Arc<Tokenizer> {
        match self {
            AnyModel::Llama(m) => m.tokenizer_arc(),
            AnyModel::Quantized(m) => m.tokenizer_arc(),
        }
    }

    /// Architecture identifier string (e.g. `"qwen2"`, `"llama"`).
    pub fn architecture(&self) -> &str {
        match self {
            AnyModel::Llama(m) => m.architecture(),
            AnyModel::Quantized(m) => m.architecture(),
        }
    }

    /// Chat template format to use for this model (e.g. `"chatml"`,
    /// `"llama2"`, `"llama3"`).
    ///
    /// This is determined at load time by probing the vocabulary for
    /// well-known special tokens, so it is reliable even for models whose
    /// GGUF architecture tag does not match their template (e.g. Nanbeige
    /// reports `"llama"` but uses ChatML).
    pub fn chat_format(&self) -> &str {
        match self {
            AnyModel::Llama(m) => m.chat_format(),
            AnyModel::Quantized(m) => m.chat_format(),
        }
    }
}

impl std::fmt::Debug for AnyModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnyModel::Llama(_) => write!(f, "AnyModel::Llama"),
            AnyModel::Quantized(_) => write!(f, "AnyModel::Quantized"),
        }
    }
}

impl Model for AnyModel {
    fn load(config: &ModelConfig, device: &Device) -> CoreResult<Self> {
        let dir = &config.model_dir;

        // Check for GGUF first
        let has_gguf = std::fs::read_dir(dir)
            .map(|mut d| {
                d.any(|e| {
                    e.ok()
                        .and_then(|e| e.path().extension().map(|x| x == "gguf"))
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

        if has_gguf {
            return QuantizedModel::load(config, device).map(AnyModel::Quantized);
        }

        let has_safetensors = dir.join("model.safetensors").exists()
            || dir.join("model.safetensors.index.json").exists();

        if has_safetensors {
            return LlamaModel::load(config, device).map(AnyModel::Llama);
        }

        Err(CoreError::Config {
            field: "model_dir".to_string(),
            reason: format!(
                "No supported model weights (.gguf or .safetensors) found in {}",
                dir.display()
            ),
        })
    }

    fn generate(
        &mut self,
        input: GenerateInput,
        params: SamplingParams,
    ) -> CoreResult<mpsc::UnboundedReceiver<CoreResult<Token>>> {
        match self {
            AnyModel::Llama(m) => m.generate(input, params),
            AnyModel::Quantized(m) => m.generate(input, params),
        }
    }

    fn generate_stream(
        &mut self,
        input: GenerateInput,
        params: SamplingParams,
        tx: mpsc::UnboundedSender<CoreResult<Token>>,
    ) -> CoreResult<()> {
        match self {
            AnyModel::Llama(m) => m.generate_stream(input, params, tx),
            AnyModel::Quantized(m) => m.generate_stream(input, params, tx),
        }
    }
}
