//! # xandllm-core
//!
//! Model loading, inference, tokenization, and device abstraction for XandLLM.
//!
//! ## Supported Architectures
//!
//! | Architecture | Format | Struct |
//! |--------------|--------|--------|
//! | LLaMA / Mistral | Safetensors | [`LlamaModel`] |
//! | LLaMA-family | GGUF (quantized) | [`QuantizedModel`] |
//! | Qwen2 | GGUF (quantized) | [`QuantizedModel`] |
//!
//! See the main [README](../../../README.md#supported-llm-architectures) for details.
//!
//! ## Feature Flags
//!
//! | Flag | Effect |
//! |---|---|
//! | `cuda` | Enable CUDA GPU acceleration |
//! | `metal` | Enable Apple Metal GPU acceleration |

pub mod chat_template;
pub mod device;
pub mod error;
pub mod kv_cache;
pub mod llama;
pub mod loader;
pub mod model;
pub mod quantized;
pub mod sampling;
pub mod tokenizer;

pub use device::select_device;
pub use error::{CoreError, CoreResult};
pub use llama::LlamaModel;
pub use loader::AnyModel;
pub use model::{GenerateInput, Model, ModelConfig, SamplingParams, Token};
pub use quantized::QuantizedModel;
pub use tokenizer::Tokenizer;
