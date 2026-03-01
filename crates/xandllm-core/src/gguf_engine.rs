/*!
 * gguf_engine.rs — Rust FFI wrapper for the custom C/CUDA Gemma 3 engine
 *
 * This module is compiled only when the `cuda_engine` feature is active.
 * It:
 *  1. Declares the `extern "C"` FFI bindings that mirror `csrc/engine.h`.
 *  2. Wraps the opaque C handle in a safe `GgufEngine` struct.
 *  3. Implements the `Model` trait so `GgufEngine` slots into the existing
 *     loader / API pipeline without changing any route code.
 *
 * ## Key design decisions
 *
 * - **Tokenizer**: we reuse the existing Rust `Tokenizer` (built from GGUF
 *   metadata via candle's `gguf_file::Content::read`).  Only the tokenizer
 *   metadata path of candle is used; no candle tensors or inference code run.
 *
 * - **Sampling**: we reuse `sampling::sample_token` from `sampling.rs`.
 *   `xandengine_forward` copies logits to a caller-supplied CPU `Vec<f32>`, so
 *   no extra GPU→CPU transfer is needed before sampling.
 *
 * - **Stop detection**: identical logic to `QuantizedModel::generate_stream`.
 *
 * - **Thread safety**: `GgufEngine` is `Send + Sync` because:
 *   - The C handle pointer is opaque; we never alias it from multiple threads
 *     simultaneously (the `Arc<Mutex<dyn Model>>` in the server guarantees
 *     single-threaded access at the model level).
 *   - The `Arc<Tokenizer>` is cheaply cloneable and read-only after load.
 */

#[cfg(feature = "cuda_engine")]
mod inner {
    use std::collections::HashSet;
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use candle_core::{quantized::gguf_file, Device};
    use tokio::sync::mpsc;
    use tracing::info;

    use crate::{
        error::{CoreError, CoreResult},
        model::{GenerateInput, Model, ModelConfig, SamplingParams, Token},
        sampling::sample_token,
        tokenizer::Tokenizer,
    };

    // ── Text stop helper (copied from quantized.rs) ───────────────────────────

    fn find_text_stop(text: &str, stops: &[String]) -> Option<usize> {
        stops.iter().filter_map(|s| text.find(s.as_str())).min()
    }

    // ── FFI declarations ──────────────────────────────────────────────────────

    /// Opaque handle to the C engine — never dereferenced from Rust.
    #[repr(C)]
    pub struct XandEngine {
        _priv: [u8; 0],
    }

    extern "C" {
        fn xandengine_create(
            path: *const c_char,
            gpu_id: c_int,
            max_ctx: usize,
        ) -> *mut XandEngine;

        fn xandengine_destroy(e: *mut XandEngine);
        fn xandengine_reset_kv(e: *mut XandEngine);

        fn xandengine_forward(
            e: *mut XandEngine,
            tokens: *const i32,
            n_tokens: c_int,
            logits_out: *mut f32,
            vocab_size: c_int,
        ) -> c_int;

        fn xandengine_vocab_size(e: *const XandEngine) -> c_int;
        fn xandengine_chat_format(e: *const XandEngine) -> *const c_char;
    }

    // ── GgufEngine ───────────────────────────────────────────────────────────

    /// Inference engine backed by the custom C/CUDA Gemma 3 implementation.
    pub struct GgufEngine {
        /// Opaque C engine handle — owns all GPU memory.
        handle: *mut XandEngine,
        /// Shared tokenizer (built from GGUF metadata on load).
        tokenizer: Arc<Tokenizer>,
        /// Chat format string returned by the C engine ("gemma").
        chat_format: String,
        /// Vocabulary size — pre-fetched to avoid repeated FFI calls.
        vocab_size: usize,
        /// Maximum context length used at creation time.
        max_seq_len: usize,
    }

    // SAFETY: XandEngine is a single-threaded C struct protected externally by
    // Arc<Mutex<dyn Model>> in the server.  No data races are possible given
    // that guarantee.
    unsafe impl Send for GgufEngine {}
    unsafe impl Sync for GgufEngine {}

    impl Drop for GgufEngine {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { xandengine_destroy(self.handle) };
                self.handle = std::ptr::null_mut();
            }
        }
    }

    impl GgufEngine {
        /// Return a cheap `Arc` clone of the tokenizer for sharing with the API
        /// server (mirrors `QuantizedModel::tokenizer_arc`).
        pub fn tokenizer_arc(&self) -> Arc<Tokenizer> {
            Arc::clone(&self.tokenizer)
        }

        /// Architecture tag — always `"gemma3"` for this engine.
        pub fn architecture(&self) -> &str {
            "gemma3"
        }

        /// Chat template format — always `"gemma"` for this engine.
        pub fn chat_format(&self) -> &str {
            &self.chat_format
        }
    }

    // ── Tokenizer loading ─────────────────────────────────────────────────────

    /// Build a `Tokenizer` from the GGUF file using candle's metadata reader.
    ///
    /// We use candle *only* to parse the metadata section of the first GGUF
    /// shard (the part before the tensor data).  No candle tensors are loaded.
    fn load_tokenizer(gguf_path: &Path) -> CoreResult<Tokenizer> {
        let mut file = std::fs::File::open(gguf_path).map_err(CoreError::Io)?;
        let content = gguf_file::Content::read(&mut file).map_err(CoreError::Candle)?;

        // Try tokenizer.json in the same directory first.
        let dir = gguf_path.parent().unwrap_or(Path::new("."));
        let tokenizer_json = dir.join("tokenizer.json");
        if tokenizer_json.exists() {
            return Tokenizer::from_file(&tokenizer_json);
        }

        // Fall back to building the tokenizer from GGUF metadata.
        Tokenizer::from_gguf_metadata(&content.metadata)
    }

    // ── Model trait implementation ────────────────────────────────────────────

    impl Model for GgufEngine {
        fn load(config: &ModelConfig, _device: &Device) -> CoreResult<Self> {
            // Find the first .gguf file in the model directory.
            let gguf_path = find_gguf_file(&config.model_dir).ok_or_else(|| CoreError::Config {
                field: "model_dir".to_string(),
                reason: format!(
                    "No .gguf file found in {}",
                    config.model_dir.display()
                ),
            })?;

            info!(
                path = %gguf_path.display(),
                max_ctx = config.max_sequence_length,
                "GgufEngine: loading Gemma 3 model"
            );

            let tokenizer = Arc::new(load_tokenizer(&gguf_path)?);

            let path_cstr = CString::new(gguf_path.to_str().ok_or_else(|| {
                CoreError::Config {
                    field: "gguf_path".to_string(),
                    reason: "GGUF path contains non-UTF-8 characters".to_string(),
                }
            })?)
            .map_err(|e| CoreError::Config {
                field: "gguf_path".to_string(),
                reason: e.to_string(),
            })?;

            let handle = unsafe {
                xandengine_create(path_cstr.as_ptr(), 0, config.max_sequence_length)
            };

            if handle.is_null() {
                return Err(CoreError::Config {
                    field: "gguf_engine".to_string(),
                    reason: "xandengine_create returned NULL — check stderr for details"
                        .to_string(),
                });
            }

            let vocab_size = unsafe { xandengine_vocab_size(handle) } as usize;

            let chat_format = unsafe {
                let ptr = xandengine_chat_format(handle);
                if ptr.is_null() {
                    "gemma".to_string()
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };

            info!(
                vocab_size,
                chat_format,
                max_ctx = config.max_sequence_length,
                "GgufEngine: model loaded"
            );

            Ok(GgufEngine {
                handle,
                tokenizer,
                chat_format,
                vocab_size,
                max_seq_len: config.max_sequence_length,
            })
        }

        fn generate(
            &mut self,
            input: GenerateInput,
            params: SamplingParams,
        ) -> CoreResult<mpsc::UnboundedReceiver<CoreResult<Token>>> {
            let (tx, rx) = mpsc::unbounded_channel();
            self.generate_stream(input, params, tx)?;
            Ok(rx)
        }

        fn generate_stream(
            &mut self,
            input: GenerateInput,
            params: SamplingParams,
            tx: mpsc::UnboundedSender<CoreResult<Token>>,
        ) -> CoreResult<()> {
            let seq_len = input.token_ids.len();
            if seq_len == 0 {
                return Err(CoreError::Config {
                    field: "token_ids".to_string(),
                    reason: "Prompt encoded to zero tokens".to_string(),
                });
            }
            if seq_len > self.max_seq_len {
                return Err(CoreError::SequenceTooLong {
                    got: seq_len,
                    max: self.max_seq_len,
                });
            }

            // Reset KV cache for new conversation turn.
            unsafe { xandengine_reset_kv(self.handle) };

            let stop_ids: HashSet<u32> = params.stop_token_ids.iter().copied().collect();
            let mut token_history: Vec<u32> = input.token_ids.clone();

            // Allocate CPU logits buffer (reused across decode steps).
            let mut logits_cpu = vec![0.0f32; self.vocab_size];

            // ── Prefill ───────────────────────────────────────────────────────
            let tokens_i32: Vec<i32> = input.token_ids.iter().map(|&id| id as i32).collect();
            let rc = unsafe {
                xandengine_forward(
                    self.handle,
                    tokens_i32.as_ptr(),
                    seq_len as c_int,
                    logits_cpu.as_mut_ptr(),
                    self.vocab_size as c_int,
                )
            };
            if rc != 0 {
                return Err(CoreError::Config {
                    field: "xandengine_forward".to_string(),
                    reason: format!("prefill returned error code {rc}"),
                });
            }

            // Sample the first decode token from prefill logits.
            let first_id = sample_from_cpu_logits(&logits_cpu, &params, &token_history)?;
            token_history.push(first_id);

            if stop_ids.contains(&first_id) {
                return Ok(());
            }

            let first_text = self.tokenizer.decode_token(first_id)?;

            let max_stop_len: usize =
                params.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0) + 64;
            let mut accumulated = String::new();
            let mut think_closed = !params.thinking_mode;

            if !think_closed && first_text.contains("</think>") {
                think_closed = true;
            }

            // Check first token against text stops.
            if think_closed && !params.stop_strings.is_empty() {
                accumulated.push_str(&first_text);
                if let Some(cut) = find_text_stop(&accumulated, &params.stop_strings) {
                    tracing::debug!(
                        cut,
                        context = %&accumulated[..cut.min(accumulated.len())],
                        "Text stop matched on first token"
                    );
                    let safe = accumulated[..cut].to_string();
                    if !safe.is_empty() {
                        let _ = tx.send(Ok(Token { id: first_id, text: safe, is_eos: false }));
                    }
                    return Ok(());
                }
            } else {
                accumulated.push_str(&first_text);
            }

            if tx.send(Ok(Token { id: first_id, text: first_text, is_eos: false })).is_err() {
                return Ok(());
            }

            if accumulated.len() > max_stop_len {
                let trim_at = accumulated.len() - max_stop_len;
                accumulated.drain(..trim_at);
            }

            if params.max_new_tokens <= 1 {
                return Ok(());
            }

            // ── Decode loop ───────────────────────────────────────────────────
            let mut last_id = first_id;
            let mut pos = seq_len; // position of the next token to generate

            'gen: for _ in 1..params.max_new_tokens {
                let token_i32 = [last_id as i32];
                let rc = unsafe {
                    xandengine_forward(
                        self.handle,
                        token_i32.as_ptr(),
                        1,
                        logits_cpu.as_mut_ptr(),
                        self.vocab_size as c_int,
                    )
                };
                if rc != 0 {
                    let _ = tx.send(Err(CoreError::Config {
                        field: "xandengine_forward".to_string(),
                        reason: format!("decode step returned error code {rc}"),
                    }));
                    break 'gen;
                }

                let next_id = sample_from_cpu_logits(&logits_cpu, &params, &token_history)?;

                // Cap history to penalty window.
                let cap = params.repeat_last_n.unwrap_or(usize::MAX);
                if token_history.len() >= cap {
                    token_history.drain(..token_history.len() - cap + 1);
                }
                token_history.push(next_id);

                if stop_ids.contains(&next_id) {
                    break 'gen;
                }

                let text = self.tokenizer.decode_token(next_id)?;

                if !think_closed && text.contains("</think>") {
                    think_closed = true;
                }

                if think_closed && !params.stop_strings.is_empty() {
                    accumulated.push_str(&text);
                    if let Some(cut_in_suffix) =
                        find_text_stop(&accumulated, &params.stop_strings)
                    {
                        tracing::debug!(
                            cut = cut_in_suffix,
                            context = %&accumulated[..cut_in_suffix.min(accumulated.len())],
                            "Text stop matched during generation"
                        );
                        let already_safe_len = accumulated.len() - text.len();
                        if cut_in_suffix > already_safe_len {
                            let safe_part = &accumulated[already_safe_len..cut_in_suffix];
                            if !safe_part.is_empty() {
                                let _ = tx.send(Ok(Token {
                                    id: next_id,
                                    text: safe_part.to_string(),
                                    is_eos: false,
                                }));
                            }
                        }
                        break 'gen;
                    }
                    if accumulated.len() > max_stop_len {
                        let trim_at = accumulated.len() - max_stop_len;
                        accumulated.drain(..trim_at);
                    }
                }

                if tx.send(Ok(Token { id: next_id, text, is_eos: false })).is_err() {
                    break 'gen;
                }
                last_id = next_id;
                pos += 1;

                // Guard against KV cache overflow.
                if pos >= self.max_seq_len {
                    break 'gen;
                }
            }

            Ok(())
        }
    }

    // ── Sampling adapter ──────────────────────────────────────────────────────

    /// Wrap a CPU `Vec<f32>` logits buffer in a candle `Tensor` (CPU device)
    /// and delegate to the existing `sample_token` function.
    ///
    /// The logits are already on the CPU after `xandengine_forward`, so this
    /// creates a zero-copy 1-D CPU tensor by wrapping the slice.
    fn sample_from_cpu_logits(
        logits: &[f32],
        params: &SamplingParams,
        history: &[u32],
    ) -> CoreResult<u32> {
        use candle_core::Tensor;
        let t = Tensor::from_slice(logits, logits.len(), &Device::Cpu)
            .map_err(CoreError::Candle)?;
        sample_token(&t, params, history)
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    /// Find the first `.gguf` file in `dir`, sorted lexicographically so that
    /// multi-shard models return the first shard.
    fn find_gguf_file(dir: &Path) -> Option<PathBuf> {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "gguf"))
            .collect();
        entries.sort();
        entries.into_iter().next()
    }
} // mod inner

// ── Re-exports ────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda_engine")]
pub use inner::GgufEngine;

// Provide a stub type when the feature is disabled so other modules can refer
// to `GgufEngine` without conditional compilation everywhere.
#[cfg(not(feature = "cuda_engine"))]
pub struct GgufEngine;

#[cfg(not(feature = "cuda_engine"))]
impl GgufEngine {
    pub fn tokenizer_arc(&self) -> std::sync::Arc<crate::tokenizer::Tokenizer> {
        unimplemented!("GgufEngine requires the cuda_engine feature")
    }
    pub fn architecture(&self) -> &str { "gemma3" }
    pub fn chat_format(&self)  -> &str { "gemma"  }
}

#[cfg(not(feature = "cuda_engine"))]
impl crate::model::Model for GgufEngine {
    fn load(
        _config: &crate::model::ModelConfig,
        _device: &candle_core::Device,
    ) -> crate::error::CoreResult<Self> {
        Err(crate::error::CoreError::Config {
            field: "backend".to_string(),
            reason: "GgufEngine requires the cuda_engine feature flag".to_string(),
        })
    }

    fn generate(
        &mut self,
        _input: crate::model::GenerateInput,
        _params: crate::model::SamplingParams,
    ) -> crate::error::CoreResult<tokio::sync::mpsc::UnboundedReceiver<crate::error::CoreResult<crate::model::Token>>> {
        unimplemented!("GgufEngine requires the cuda_engine feature")
    }

    fn generate_stream(
        &mut self,
        _input: crate::model::GenerateInput,
        _params: crate::model::SamplingParams,
        _tx: tokio::sync::mpsc::UnboundedSender<crate::error::CoreResult<crate::model::Token>>,
    ) -> crate::error::CoreResult<()> {
        unimplemented!("GgufEngine requires the cuda_engine feature")
    }
}
