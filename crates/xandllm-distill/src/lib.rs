//! # xandllm-distill
//!
//! Knowledge-distillation utilities for XandLLM.
//!
//! Distils a large *teacher* model (GGUF or safetensors) into a smaller
//! *student* model using sequence-level distillation:
//!
//! 1. **Phase 1** — The teacher model generates completions for every prompt
//!    in the dataset and saves the results to an intermediate JSONL file.
//! 2. **Phase 2** — A trainable student model (Llama-family, backed by a
//!    [`candle_nn::VarMap`]) is trained with cross-entropy loss to predict
//!    those teacher-generated completions.
//!
//! ## Usage
//!
//! ```text
//! xandllm distill \
//!   --model-from  Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
//!   --dataset     ./my_dataset \
//!   --model-to    ./output/XandLM-1B \
//!   --size        1b \
//!   --type        safetensor
//! ```
//!
//! or, using an existing smaller model as the student base:
//!
//! ```text
//! xandllm distill \
//!   --model-from  Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
//!   --dataset     ./my_dataset \
//!   --model-to    ./output/MyFineTuned \
//!   --student-base Qwen/Qwen2.5-1.5B \
//!   --type        safetensor
//! ```

pub mod dataset;
pub mod distiller;
pub mod export;
pub mod presets;
pub mod student;
pub mod teacher;

pub use distiller::{DistillConfig, Distiller};
pub use export::OutputFormat;
pub use presets::SizePreset;
