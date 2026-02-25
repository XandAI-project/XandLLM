use std::path::PathBuf;

use candle_core::Device;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::CoreResult;

/// A single generated token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// Raw token id as produced by the model vocabulary.
    pub id: u32,
    /// Decoded text fragment for this token.
    pub text: String,
    /// Whether this is an end-of-sequence token.
    pub is_eos: bool,
}

/// Parameters controlling text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    /// Limit sampling to top K tokens (applied before top_p).
    pub top_k: Option<usize>,
    /// Penalty for repeating tokens (1.0 = no penalty, >1.0 = discourage repetition).
    pub repetition_penalty: f64,
    /// OpenAI-style frequency penalty (penalizes tokens based on frequency).
    pub frequency_penalty: f64,
    /// OpenAI-style presence penalty (penalizes tokens that have appeared).
    pub presence_penalty: f64,
    /// Random seed for reproducible generation.
    pub seed: Option<u64>,
    /// Greedy decoding when true (ignores temperature / top_p).
    pub greedy: bool,
    /// Token ids that should stop generation.
    pub stop_token_ids: Vec<u32>,
    /// Rolling window for repetition/frequency/presence penalties.
    ///
    /// Only the last `n` tokens are considered when computing penalties.
    /// `None` means the full history is used (unbounded).  Set to a small
    /// value (e.g. 64) to cap O(T) penalty scans on long sequences —
    /// this is the `repeat_last_n` parameter from llama.cpp.
    pub repeat_last_n: Option<usize>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: None,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
            greedy: false,
            stop_token_ids: vec![],
            // Default 64: matches llama.cpp's repeat_last_n default.
            // Bounds penalty cost to O(64) regardless of sequence length.
            repeat_last_n: Some(64),
        }
    }
}

/// Input to the model for a single generation request.
#[derive(Debug, Clone)]
pub struct GenerateInput {
    /// Pre-tokenized prompt ids.
    pub token_ids: Vec<u32>,
}

/// Configuration for loading a model from the local cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Hugging Face repo id (e.g. `mistralai/Mistral-7B-v0.1`).
    pub repo_id: String,
    /// Git revision / branch.
    pub revision: String,
    /// Directory where model files are stored.
    pub model_dir: PathBuf,
    /// Maximum supported sequence length.
    pub max_sequence_length: usize,
}

/// Core trait that every model architecture must implement.
///
/// Implementations must be `Send + Sync` so they can be shared across async tasks
/// behind an `Arc<Mutex<dyn Model>>`.
pub trait Model: Send + Sync {
    /// Load model weights and tokenizer from `config.model_dir`.
    fn load(config: &ModelConfig, device: &Device) -> CoreResult<Self>
    where
        Self: Sized;

    /// Synchronous generation — fills the entire channel before returning.
    ///
    /// The full inference loop runs on the calling thread and completes before
    /// the receiver is returned. Use this for non-streaming API responses where
    /// the caller wants to collect all tokens before responding.
    fn generate(
        &mut self,
        input: GenerateInput,
        params: SamplingParams,
    ) -> CoreResult<mpsc::UnboundedReceiver<CoreResult<Token>>>;

    /// Async-friendly streaming generation — sends tokens as they're produced.
    ///
    /// This is a **synchronous** method that drives the inference loop on the
    /// calling thread, but it **returns immediately** after validation. Call it
    /// inside `tokio::task::spawn_blocking` (without `.await`) so tokens flow
    /// into the channel while the async SSE handler drains it concurrently.
    ///
    /// The channel closes automatically when generation completes (EOS hit, max
    /// tokens reached, or `tx` is dropped by the caller).
    fn generate_stream(
        &mut self,
        input: GenerateInput,
        params: SamplingParams,
        tx: mpsc::UnboundedSender<CoreResult<Token>>,
    ) -> CoreResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_default_values() {
        let p = SamplingParams::default();
        assert_eq!(p.max_new_tokens, 512);
        assert!((p.temperature - 0.7).abs() < f64::EPSILON);
        assert!((p.top_p - 0.9).abs() < f64::EPSILON);
        assert!(p.top_k.is_none());
        assert!((p.repetition_penalty - 1.0).abs() < f64::EPSILON);
        assert!(p.frequency_penalty.abs() < f64::EPSILON);
        assert!(p.presence_penalty.abs() < f64::EPSILON);
        assert!(p.seed.is_none());
        assert!(!p.greedy);
        assert!(p.stop_token_ids.is_empty());
    }

    #[test]
    fn test_token_serde_roundtrip() {
        let original = Token {
            id: 42,
            text: "hello world".to_string(),
            is_eos: false,
        };
        let json = serde_json::to_string(&original).unwrap();
        let decoded: Token = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, 42);
        assert_eq!(decoded.text, "hello world");
        assert!(!decoded.is_eos);
    }

    #[test]
    fn test_eos_token_roundtrip() {
        let tok = Token { id: 2, text: "</s>".to_string(), is_eos: true };
        let json = serde_json::to_string(&tok).unwrap();
        let back: Token = serde_json::from_str(&json).unwrap();
        assert!(back.is_eos);
    }

    #[test]
    fn test_model_config_serde() {
        let cfg = ModelConfig {
            repo_id: "meta-llama/Llama-3.1-8B".to_string(),
            revision: "main".to_string(),
            model_dir: std::path::PathBuf::from("/tmp/model"),
            max_sequence_length: 8192,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.repo_id, "meta-llama/Llama-3.1-8B");
        assert_eq!(back.max_sequence_length, 8192);
    }

    #[test]
    fn test_generate_input_holds_token_ids() {
        let input = GenerateInput { token_ids: vec![1, 2, 3, 4] };
        assert_eq!(input.token_ids.len(), 4);
        assert_eq!(input.token_ids[0], 1);
    }
}
