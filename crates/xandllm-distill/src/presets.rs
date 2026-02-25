//! Architecture size presets for fresh student initialisation.
//!
//! Each variant maps to a well-known Llama-family config so the student can be
//! constructed without any pre-existing checkpoint.  The shapes are chosen to
//! match popular open-weight models of the same parameter count:
//!
//! | Preset | Inspiration | Layers | Hidden | Heads | KV heads | ~Params |
//! |--------|-------------|--------|--------|-------|----------|---------|
//! | `1b`   | TinyLlama   | 22     | 2048   | 32    | 4 (GQA)  | ~1.1B   |
//! | `3b`   | Phi-3-mini  | 28     | 3072   | 24    | 8 (GQA)  | ~3B     |
//! | `7b`   | LLaMA-3 8B  | 32     | 4096   | 32    | 8 (GQA)  | ~7B     |

use anyhow::{bail, Result};
use candle_transformers::models::llama::Config as LlamaConfig;

/// Student architecture size preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizePreset {
    /// ~1.1 B parameters — TinyLlama-inspired.
    B1,
    /// ~3 B parameters — Phi-3-mini-inspired.
    B3,
    /// ~7 B parameters — LLaMA-3 8B-inspired.
    B7,
}

impl SizePreset {
    /// Parse a user-supplied string (`"1b"`, `"3b"`, `"7b"`).
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().trim_end_matches('b') {
            "1" => Ok(Self::B1),
            "3" => Ok(Self::B3),
            "7" => Ok(Self::B7),
            other => bail!(
                "Unknown size preset '{}'. Valid values: 1b, 3b, 7b.",
                other
            ),
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::B1 => "1b",
            Self::B3 => "3b",
            Self::B7 => "7b",
        }
    }

    /// Approximate parameter count (for logging / display).
    pub fn approx_params(&self) -> &'static str {
        match self {
            Self::B1 => "~1.1B",
            Self::B3 => "~3B",
            Self::B7 => "~7B",
        }
    }

    /// Build the Candle [`LlamaConfig`] for this preset.
    ///
    /// `vocab_size` must match the teacher tokenizer so the student's embedding
    /// matrix and language-model head have the right dimensions.
    pub fn llama_config(&self, vocab_size: usize) -> LlamaConfig {
        match self {
            Self::B1 => LlamaConfig {
                // TinyLlama-1.1B: 22 layers, GQA 4 KV heads
                hidden_size: 2048,
                intermediate_size: 5632,
                vocab_size,
                num_hidden_layers: 22,
                num_attention_heads: 32,
                num_key_value_heads: 4,
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
                use_flash_attn: false,
                bos_token_id: None,
                eos_token_id: None,
                rope_scaling: None,
                max_position_embeddings: 2048,
                tie_word_embeddings: false,
            },
            Self::B3 => LlamaConfig {
                // Phi-3-mini inspired: 28 layers, GQA 8 KV heads
                hidden_size: 3072,
                intermediate_size: 8192,
                vocab_size,
                num_hidden_layers: 28,
                num_attention_heads: 24,
                num_key_value_heads: 8,
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
                use_flash_attn: false,
                bos_token_id: None,
                eos_token_id: None,
                rope_scaling: None,
                max_position_embeddings: 4096,
                tie_word_embeddings: false,
            },
            Self::B7 => LlamaConfig {
                // LLaMA-3 8B inspired: 32 layers, GQA 8 KV heads
                hidden_size: 4096,
                intermediate_size: 14336,
                vocab_size,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                num_key_value_heads: 8,
                rms_norm_eps: 1e-5,
                rope_theta: 500_000.0,
                use_flash_attn: false,
                bos_token_id: None,
                eos_token_id: None,
                rope_scaling: None,
                max_position_embeddings: 8192,
                tie_word_embeddings: false,
            },
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_sizes() {
        assert_eq!(SizePreset::parse("1b").unwrap(), SizePreset::B1);
        assert_eq!(SizePreset::parse("3b").unwrap(), SizePreset::B3);
        assert_eq!(SizePreset::parse("7b").unwrap(), SizePreset::B7);
        // Case-insensitive
        assert_eq!(SizePreset::parse("1B").unwrap(), SizePreset::B1);
    }

    #[test]
    fn parse_invalid_size_errors() {
        assert!(SizePreset::parse("2b").is_err());
        assert!(SizePreset::parse("").is_err());
    }

    #[test]
    fn llama_config_vocab_size_propagated() {
        let cfg = SizePreset::B1.llama_config(32_000);
        assert_eq!(cfg.vocab_size, 32_000);
    }

    #[test]
    fn llama_config_shapes_differ_by_preset() {
        let b1 = SizePreset::B1.llama_config(32_000);
        let b7 = SizePreset::B7.llama_config(32_000);
        assert!(b1.num_hidden_layers < b7.num_hidden_layers);
        assert!(b1.hidden_size < b7.hidden_size);
    }
}
