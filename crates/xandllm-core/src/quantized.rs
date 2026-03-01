use std::collections::{HashMap, HashSet};
use std::io::Seek;
use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{quantized::gguf_file, Device, IndexOp, Tensor};
use candle_transformers::models::{quantized_gemma3, quantized_llama, quantized_phi, quantized_phi3, quantized_qwen2, quantized_qwen3};
use tokio::sync::mpsc;
use tracing::info;

use crate::{
    error::{CoreError, CoreResult},
    model::{GenerateInput, Model, ModelConfig, SamplingParams, Token},
    sampling::sample_token,
    tokenizer::Tokenizer,
};

// ─── GGUF load error enrichment ───────────────────────────────────────────────

/// Candle emits "unknown dtype for tensor N" when a GGUF file contains an
/// IQ-family (imatrix) quantization type such as IQ4_XS (dtype 23), IQ3_S,
/// IQ2_XXS, etc.  These types are used by Unsloth "UD" (Unsloth Dynamic)
/// mixed-precision files (e.g. *-UD-Q4_K_XL.gguf) and are not implemented
/// in candle 0.9.x.
///
/// This helper intercepts that opaque candle error and replaces it with an
/// `UnsupportedQuantization` error that names the problematic dtype and tells
/// the user exactly which quant suffixes to download instead.
fn map_gguf_load_error(err: candle_core::Error, model_dir: &std::path::Path) -> CoreError {
    let msg = err.to_string();
    if msg.contains("unknown dtype") {
        // Extract the raw dtype number from the error string ("unknown dtype for tensor N").
        let dtype_hint = msg
            .split_whitespace()
            .last()
            .map(|s| {
                // Map known IQ dtype numbers to their names for a friendlier message.
                match s {
                    "16" => "IQ2_XXS (dtype 16)".to_string(),
                    "17" => "IQ2_XS (dtype 17)".to_string(),
                    "18" => "IQ3_XXS (dtype 18)".to_string(),
                    "19" => "IQ1_S (dtype 19)".to_string(),
                    "20" => "IQ4_NL (dtype 20)".to_string(),
                    "21" => "IQ3_S (dtype 21)".to_string(),
                    "22" => "IQ2_S (dtype 22)".to_string(),
                    "23" => "IQ4_XS (dtype 23)".to_string(),
                    "29" => "IQ1_M (dtype 29)".to_string(),
                    other => format!("dtype {other}"),
                }
            })
            .unwrap_or_else(|| msg.clone());

        // Reconstruct a best-effort repo ID from the cache directory name
        // (e.g. "unsloth__gemma-3-12b-it-GGUF/main" → "unsloth/gemma-3-12b-it-GGUF").
        let repo = model_dir
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().replace("__", "/"))
            .unwrap_or_else(|| "the-model-repo".to_string());

        CoreError::UnsupportedQuantization {
            detail: dtype_hint,
            repo,
        }
    } else {
        CoreError::Candle(err)
    }
}

// ─── Architecture dispatch ────────────────────────────────────────────────────

/// Wraps quantized model weights for all supported architectures behind the
/// same `forward` call.
enum QuantizedWeights {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Phi2(quantized_phi::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
    Gemma(quantized_gemma3::ModelWeights),
}

impl QuantizedWeights {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, index_pos),
            Self::Qwen2(m) => m.forward(x, index_pos),
            Self::Qwen3(m) => m.forward(x, index_pos),
            Self::Phi2(m) => m.forward(x, index_pos),
            Self::Phi3(m) => m.forward(x, index_pos),
            Self::Gemma(m) => m.forward(x, index_pos),
        }
    }

    /// Reset any internally-managed KV caches that use append semantics
    /// (e.g. `ConcatKvCache` in Qwen3). Must be called before each new
    /// full-prompt prefill so stale cache entries from a previous turn don't
    /// corrupt the causal-mask shape calculation.
    fn clear_kv_cache(&mut self) {
        if let Self::Qwen3(m) = self {
            m.clear_kv_cache();
        }
    }
}

// ─── Multi-shard GGUF reader ──────────────────────────────────────────────────

/// One contiguous slice of tensor data belonging to a single shard file.
struct Segment {
    /// Byte offset in the virtual concatenated space where this segment starts.
    virtual_start: u64,
    /// Number of tensor-data bytes in this shard file.
    data_len: u64,
    /// Open file handle for this shard.
    file: std::fs::File,
    /// Absolute byte offset in the real file where the tensor data begins.
    file_data_start: u64,
}

/// A `Read + Seek` adapter that presents multiple GGUF shard files as a single
/// contiguous byte stream.
///
/// The first shard's tensor data occupies virtual offsets `[0, shard1_len)`.
/// The second shard follows immediately, and so on.
struct MultiFileReader {
    segments: Vec<Segment>,
    pos: u64,
    total_len: u64,
}

impl MultiFileReader {
    fn segment_idx_for(&self, pos: u64) -> Option<usize> {
        self.segments
            .iter()
            .position(|s| pos >= s.virtual_start && pos < s.virtual_start + s.data_len)
    }
}

impl std::io::Read for MultiFileReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let Some(idx) = self.segment_idx_for(self.pos) else {
            return Ok(0); // EOF
        };
        let seg = &mut self.segments[idx];
        let file_offset = seg.file_data_start + (self.pos - seg.virtual_start);
        let remaining = (seg.virtual_start + seg.data_len - self.pos) as usize;
        let to_read = buf.len().min(remaining);
        seg.file.seek(std::io::SeekFrom::Start(file_offset))?;
        let n = seg.file.read(&mut buf[..to_read])?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl std::io::Seek for MultiFileReader {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.pos = match pos {
            std::io::SeekFrom::Start(n) => n,
            std::io::SeekFrom::End(n) => {
                if n < 0 {
                    self.total_len.saturating_sub((-n) as u64)
                } else {
                    self.total_len + n as u64
                }
            }
            std::io::SeekFrom::Current(n) => {
                if n < 0 {
                    self.pos.saturating_sub((-n) as u64)
                } else {
                    self.pos + n as u64
                }
            }
        };
        Ok(self.pos)
    }
}

// ─── Shard merging ────────────────────────────────────────────────────────────

/// Read all GGUF shards and merge them into a single `(Content, MultiFileReader)`.
///
/// - Metadata is taken from the first shard (alphabetically).
/// - Tensor infos from all shards are merged; each info's `offset` is remapped
///   into a virtual address space starting at 0.
/// - The returned `Content` always has `tensor_data_offset = 0`.
fn merge_gguf_shards(paths: &[PathBuf]) -> CoreResult<(gguf_file::Content, MultiFileReader)> {
    let mut sorted = paths.to_vec();
    sorted.sort(); // lex sort: -00001- < -00002- < …

    let mut first_magic = None;
    let mut first_metadata = None;
    let mut all_tensor_infos = HashMap::new();
    let mut segments = Vec::new();
    let mut virtual_cursor = 0u64;

    for (shard_idx, path) in sorted.iter().enumerate() {
        let file_len = std::fs::metadata(path).map_err(CoreError::Io)?.len();
        let mut file = std::fs::File::open(path).map_err(CoreError::Io)?;
        // Use map_err so an "unknown dtype" candle error (IQ-family quant types
        // not supported by candle) surfaces as UnsupportedQuantization instead
        // of the cryptic "Candle error: unknown dtype for tensor N".
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| map_gguf_load_error(e, path.parent().unwrap_or(path)))?;

        let tensor_data_start = content.tensor_data_offset;
        let tensor_data_len = file_len - tensor_data_start;

        // Remap each tensor's offset into the virtual address space
        for (name, info) in content.tensor_infos {
            all_tensor_infos.insert(
                name,
                gguf_file::TensorInfo {
                    ggml_dtype: info.ggml_dtype,
                    shape: info.shape,
                    offset: virtual_cursor + info.offset,
                },
            );
        }

        if shard_idx == 0 {
            first_magic = Some(content.magic);
            first_metadata = Some(content.metadata);
        }
        // For shard > 0: metadata is dropped (only shard 0 has the authoritative copy)

        segments.push(Segment {
            virtual_start: virtual_cursor,
            data_len: tensor_data_len,
            file,
            file_data_start: tensor_data_start,
        });
        virtual_cursor += tensor_data_len;
    }

    let merged = gguf_file::Content {
        magic: first_magic.unwrap(),
        metadata: first_metadata.unwrap(),
        tensor_infos: all_tensor_infos,
        tensor_data_offset: 0, // offsets in tensor_infos are already absolute in virtual space
    };

    let reader = MultiFileReader {
        segments,
        pos: 0,
        total_len: virtual_cursor,
    };

    Ok((merged, reader))
}

// ─── Chat format detection ────────────────────────────────────────────────────

/// Determine which chat template format a model uses.
///
/// The probe order is:
/// 1. GGUF `tokenizer.chat_template` contains `<think>` (qwen2/qwen3 arch)
///    → ChatML-Thinking (Qwen3-Thinking variants)
/// 2. Vocabulary contains `<think>` as single token (fallback for same case)
/// 3. Vocabulary contains `<|im_start|>` → ChatML  (covers Qwen2, Nanbeige, etc.)
/// 4. Vocabulary contains `<|eot_id|>`   → LLaMA-3 instruct format
/// 5. Vocabulary contains `<|end|>`      → Phi-3 instruct format
/// 6. Vocabulary contains `<start_of_turn>` → Gemma instruct format
/// 7. Fall back to architecture string
fn detect_chat_format(arch: &str, tokenizer: &Tokenizer, chat_template: Option<&str>) -> String {
    // Primary: inspect the Jinja chat template embedded in GGUF metadata.
    // This is the canonical detection method used by llama.cpp / ollama:
    // Qwen3-Thinking models include `<think>` in their assistant turn prefix
    // (e.g. `<|im_start|>assistant\n<think>\n`), whereas standard Qwen3 and
    // Qwen2 instruct models do not.
    // Note: `<think>` is a plain XML-style text marker tokenized as multiple
    // BPE pieces — it is NOT a single special token, so token_id() cannot
    // detect it reliably.
    if matches!(arch, "qwen2" | "qwen3") {
        if let Some(tmpl) = chat_template {
            if tmpl.contains("<think>") {
                return "chatml-thinking".to_string();
            }
        }
        // Secondary fallback: if this model somehow has <think> as a single
        // vocabulary entry (e.g. a future model or custom tokenizer), honour it.
        if tokenizer.token_id("<think>").is_some() {
            return "chatml-thinking".to_string();
        }
    }
    if tokenizer.token_id("<|im_start|>").is_some() {
        return "chatml".to_string();
    }
    if tokenizer.token_id("<|eot_id|>").is_some() {
        return "llama3".to_string();
    }
    if tokenizer.token_id("<|end|>").is_some() {
        return "phi3".to_string();
    }
    if tokenizer.token_id("<start_of_turn>").is_some() {
        return "gemma".to_string();
    }
        match arch {
        "qwen2" | "qwen3"                      => "chatml",
        "llama"                                => "llama2",
        "phi2"                                 => "phi2",
        "phi3"                                 => "phi3",
        "gemma" | "gemma2" | "gemma3" | "gemma3n" => "gemma",
        other                                  => other,
    }
    .to_string()
}

// ─── Public model type ────────────────────────────────────────────────────────

/// GGUF quantized model, supporting both LLaMA-family and Qwen2 architectures,
/// and multi-shard GGUF files.
pub struct QuantizedModel {
    weights: QuantizedWeights,
    tokenizer: Arc<Tokenizer>,
    arch: String,
    chat_format: String,
    #[allow(dead_code)]
    device: Device,
    max_sequence_length: usize,
}

impl std::fmt::Debug for QuantizedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedModel")
            .field("max_sequence_length", &self.max_sequence_length)
            .finish()
    }
}

impl QuantizedModel {
    /// Return a cheap clone of the model's tokenizer Arc.
    pub fn tokenizer_arc(&self) -> Arc<Tokenizer> {
        Arc::clone(&self.tokenizer)
    }

    /// The GGUF architecture string (e.g. `"qwen2"`, `"llama"`).
    pub fn architecture(&self) -> &str {
        &self.arch
    }

    /// The chat template format to use for this model (e.g. `"chatml"`,
    /// `"llama2"`, `"llama3"`).  Determined by vocabulary probe at load time
    /// so models like Nanbeige ("llama" arch, ChatML template) are handled
    /// correctly.
    pub fn chat_format(&self) -> &str {
        &self.chat_format
    }

    /// Collect all `.gguf` files in `model_dir`, sorted by name, and verify
    /// that any multi-shard set is complete.
    fn find_all_gguf(model_dir: &PathBuf) -> CoreResult<Vec<PathBuf>> {
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(model_dir).map_err(CoreError::Io)? {
            let entry = entry.map_err(CoreError::Io)?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                paths.push(path);
            }
        }
        if paths.is_empty() {
            return Err(CoreError::Config {
                field: "model_dir".to_string(),
                reason: format!(
                    "No .gguf file found in {}. \
                    Run `xandllm pull <model-id>:<quant>` to download it.",
                    model_dir.display()
                ),
            });
        }
        paths.sort();

        // Verify multi-shard completeness.
        // A shard filename looks like: name-00001-of-00003.gguf
        // For each shard, parse the total count and check that many paths exist.
        for path in &paths {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if let Some(total) = parse_shard_total(stem) {
                let present = paths.len();
                if present < total {
                    return Err(CoreError::Config {
                        field: "model_dir".to_string(),
                        reason: format!(
                            "Incomplete model: {present} of {total} shards found in {}.\n\
                            Run `xandllm pull <model-id>:<quant>` to download missing shards.",
                            model_dir.display()
                        ),
                    });
                }
                break; // Only need to check one shard entry
            }
        }

        Ok(paths)
    }
}

/// Extract the total shard count from a filename stem.
/// Matches the pattern `...-NNNNN-of-MMMMM` and returns `MMMMM` as `usize`.
fn parse_shard_total(stem: &str) -> Option<usize> {
    let idx = stem.rfind("-of-")?;
    let total_str = &stem[idx + 4..];
    // total_str may have additional suffixes — take only the leading digits
    let digits: String = total_str.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<usize>().ok().filter(|&n| n > 1)
}

// ─── Text stop helpers ────────────────────────────────────────────────────────

/// Check whether `text` contains any of the configured text stop strings.
///
/// Returns the byte offset of the first match, or `None` if no match.
fn find_text_stop(text: &str, stop_strings: &[String]) -> Option<usize> {
    stop_strings
        .iter()
        .filter_map(|s| {
            if s.is_empty() { None } else { text.find(s.as_str()) }
        })
        .min()
}

// ─── Logit helpers ────────────────────────────────────────────────────────────

/// Extract the logits for a single sequence position from the model output.
///
/// Different candle model implementations return different shapes:
/// - `[batch, seq_len, vocab]` — full sequence logits (LLaMA, etc.)
/// - `[batch, vocab]`          — only the last token's logits (some Qwen2 variants)
/// - `[vocab]`                 — already squeezed
///
/// In all cases we return a 1-D `[vocab]` tensor ready for sampling.
fn last_token_logits(logits: &Tensor, seq_pos: usize) -> CoreResult<Tensor> {
    match logits.dims() {
        // [batch, seq, vocab] — index batch=0, then the requested position
        [_, _, _] => Ok(logits.i((0, seq_pos))?),
        // [batch, vocab] — batch=0 already gives [vocab]; seq_pos is irrelevant
        [_, _] => Ok(logits.i(0)?),
        // [vocab] — nothing to do
        [_] => Ok(logits.clone()),
        dims => Err(CoreError::Config {
            field: "logits".to_string(),
            reason: format!("Unexpected logits shape: {dims:?}"),
        }),
    }
}

impl Model for QuantizedModel {
    fn load(config: &ModelConfig, device: &Device) -> CoreResult<Self> {
        let gguf_paths = Self::find_all_gguf(&config.model_dir)?;
        let path_names: Vec<String> = gguf_paths
            .iter()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
            .collect();

        // Warn early if any shard looks like an Unsloth Dynamic (UD) file.
        // These mix IQ-family quant types that candle does not support; the
        // load will fail with a clear UnsupportedQuantization error, but an
        // upfront warning gives faster feedback in the logs.
        if path_names.iter().any(|n| n.contains("-UD-")) {
            tracing::warn!(
                files = ?path_names,
                "GGUF file appears to use Unsloth Dynamic (UD) mixed quantization. \
                 UD files include IQ-family tensor types (IQ4_XS, IQ3_S, …) that \
                 candle does not support. The load will fail — please download a \
                 standard quant (Q4_K_M / Q6_K / Q8_0) from the same repository."
            );
        }

        info!(paths = ?path_names, "Loading quantized GGUF model");

        let (mut content, mut reader) = merge_gguf_shards(&gguf_paths)?;

        // Extract architecture and tokenizer metadata BEFORE content is
        // consumed by from_gguf (which takes ownership of it).
        let arch = match content.metadata.get("general.architecture") {
            Some(gguf_file::Value::String(s)) => s.clone(),
            _ => "llama".to_string(),
        };
        info!(architecture = %arch, "Detected GGUF architecture");

        // Clone only the tokenizer.ggml.* entries we may need as a fallback.
        let gguf_tokenizer_meta: HashMap<String, gguf_file::Value> = content
            .metadata
            .iter()
            .filter(|(k, _)| k.starts_with("tokenizer.ggml."))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Extract the Jinja chat template string from GGUF metadata.
        // This is the canonical way to detect thinking models: Qwen3-Thinking
        // variants include `<think>` in their template (e.g. the assistant turn
        // prefix ends with `<|im_start|>assistant\n<think>\n`).
        // Must be cloned here because `content` is consumed by `from_gguf` below.
        let chat_template_jinja: Option<String> = match content.metadata.get("tokenizer.chat_template") {
            Some(gguf_file::Value::String(s)) => Some(s.clone()),
            _ => None,
        };

        let weights = match arch.as_str() {
            "qwen2" => {
                let w = quantized_qwen2::ModelWeights::from_gguf(content, &mut reader, device)
                    .map_err(|e| map_gguf_load_error(e, &config.model_dir))?;
                QuantizedWeights::Qwen2(w)
            }
            "qwen3" => {
                let w = quantized_qwen3::ModelWeights::from_gguf(content, &mut reader, device)
                    .map_err(|e| map_gguf_load_error(e, &config.model_dir))?;
                QuantizedWeights::Qwen3(w)
            }
            "phi2" => {
                let w = quantized_phi::ModelWeights::from_gguf(content, &mut reader, device)
                    .map_err(|e| map_gguf_load_error(e, &config.model_dir))?;
                QuantizedWeights::Phi2(w)
            }
            "phi3" => {
                // use_flash_attn = false: flash attention requires a non-default feature flag
                let w = quantized_phi3::ModelWeights::from_gguf(false, content, &mut reader, device)
                    .map_err(|e| map_gguf_load_error(e, &config.model_dir))?;
                QuantizedWeights::Phi3(w)
            }
            "gemma" | "gemma2" | "gemma3" => {
                // quantized_gemma3 probes gemma3/gemma2/gemma metadata keys internally
                let w = quantized_gemma3::ModelWeights::from_gguf(content, &mut reader, device)
                    .map_err(|e| map_gguf_load_error(e, &config.model_dir))?;
                QuantizedWeights::Gemma(w)
            }
            "gemma3n" => {
                // Gemma 3n stores its GGUF metadata under the "gemma3n.*" prefix
                // (e.g. "gemma3n.attention.head_count"), while the quantized_gemma3
                // loader probes for "gemma3.*", "gemma2.*", etc.
                // The transformer block structure (attention, FFN, norms, tensor
                // names) is identical to Gemma 3 — only the prefix differs.
                // We remap every "gemma3n.*" key to a "gemma3.*" alias so the probe
                // succeeds and the loader reuses the same code path.
                // Extra gemma3n tensors (per-layer embeddings, vision, audio) are
                // simply ignored by the text-only loader.
                let aliases: Vec<(String, gguf_file::Value)> = content
                    .metadata
                    .iter()
                    .filter(|(k, _)| k.starts_with("gemma3n."))
                    .map(|(k, v)| (k.replacen("gemma3n.", "gemma3.", 1), v.clone()))
                    .collect();
                for (k, v) in aliases {
                    content.metadata.entry(k).or_insert(v);
                }
                let w = quantized_gemma3::ModelWeights::from_gguf(content, &mut reader, device)
                    .map_err(|e| map_gguf_load_error(e, &config.model_dir))?;
                QuantizedWeights::Gemma(w)
            }
            _ => {
                let w = quantized_llama::ModelWeights::from_gguf(content, &mut reader, device)
                    .map_err(|e| map_gguf_load_error(e, &config.model_dir))?;
                QuantizedWeights::Llama(w)
            }
        };

        // Prefer a pre-downloaded tokenizer.json; fall back to the vocabulary
        // and merge rules embedded in the GGUF metadata.
        let tokenizer_path = config.model_dir.join("tokenizer.json");
        let tokenizer = Arc::new(if tokenizer_path.exists() {
            info!(path = %tokenizer_path.display(), "Loading tokenizer from file");
            Tokenizer::from_file(&tokenizer_path)?
        } else {
            info!("tokenizer.json not found — building tokenizer from GGUF metadata");
            Tokenizer::from_gguf_metadata(&gguf_tokenizer_meta)?
        });

        // Log whether a chat template was found in GGUF metadata and if it signals thinking.
        match &chat_template_jinja {
            Some(t) => info!(
                has_think_tag = t.contains("<think>"),
                preview = %&t[..t.len().min(120)],
                "GGUF tokenizer.chat_template found"
            ),
            None => info!("GGUF tokenizer.chat_template not present in metadata"),
        }
        let chat_format = detect_chat_format(&arch, &tokenizer, chat_template_jinja.as_deref());
        info!(chat_format = %chat_format, "Detected chat format");

        info!("Quantized model loaded successfully");
        Ok(Self {
            weights,
            tokenizer,
            arch,
            chat_format,
            device: device.clone(),
            max_sequence_length: config.max_sequence_length,
        })
    }

    fn generate(
        &mut self,
        input: GenerateInput,
        params: SamplingParams,
    ) -> CoreResult<mpsc::UnboundedReceiver<CoreResult<Token>>> {
        let seq_len = input.token_ids.len();
        if seq_len == 0 {
            return Err(CoreError::Config {
                field: "token_ids".to_string(),
                reason: "Prompt encoded to zero tokens — check tokenizer or prompt input"
                    .to_string(),
            });
        }
        if seq_len > self.max_sequence_length {
            return Err(CoreError::SequenceTooLong {
                got: seq_len,
                max: self.max_sequence_length,
            });
        }

        let (tx, rx) = mpsc::unbounded_channel::<CoreResult<Token>>();

        // Build a HashSet once so every per-token stop-ID check is O(1).
        let stop_ids: HashSet<u32> = params.stop_token_ids.iter().copied().collect();

        let mut token_history = input.token_ids.clone();

        // Clear any append-style KV caches (e.g. Qwen3 ConcatKvCache) so that
        // stale entries from a previous turn don't widen the KV dimension beyond
        // what the causal mask expects when offset=0.
        self.weights.clear_kv_cache();

        let input_tensor = Tensor::new(input.token_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.weights.forward(&input_tensor, 0)?;
        let last_logits = last_token_logits(&logits, seq_len - 1)?;

        let first_id = sample_token(&last_logits, &params, &token_history)?;
        token_history.push(first_id);

        // Don't send stop tokens to output (synchronous generate)
        if stop_ids.contains(&first_id) {
            return Ok(rx);
        }

        let first_text = self.tokenizer.decode_token(first_id)?;

        // Collect all tokens first so we can truncate at text stop boundaries
        // before returning any output to the caller.
        let mut generated: Vec<(u32, String)> = vec![(first_id, first_text.clone())];

        // think_closed = false  → still inside the <think> block; text stops suppressed.
        // think_closed = true   → outside the think block (or not a thinking model).
        // For thinking mode the prompt ends with <think>, so the first generated
        // token is already inside the block.
        let mut think_closed = !params.thinking_mode;
        // Check if the very first decoded token closes the think block.
        if !think_closed && first_text.contains("</think>") {
            think_closed = true;
        }

        if params.max_new_tokens > 1 {
            let mut pos = seq_len;
            let mut last_id = first_id;

            'gen: for _ in 1..params.max_new_tokens {
                let step_input = Tensor::new(&[last_id], &self.device)?.unsqueeze(0)?;
                let step_logits = self.weights.forward(&step_input, pos)?;
                let step_last = last_token_logits(&step_logits, 0)?;
                let next_id = sample_token(&step_last, &params, &token_history)?;

                // Cap history to the penalty window to prevent unbounded Vec growth.
                let cap = params.repeat_last_n.unwrap_or(usize::MAX);
                if token_history.len() >= cap {
                    token_history.drain(..token_history.len() - cap + 1);
                }
                token_history.push(next_id);

                if stop_ids.contains(&next_id) {
                    break 'gen;
                }

                let text = self.tokenizer.decode_token(next_id)?;

                // Track think block closure so text stops are only active outside it.
                if !think_closed && text.contains("</think>") {
                    think_closed = true;
                }

                // Check text-based stop strings against the accumulated response,
                // but only once the think block has been closed.
                if think_closed && !params.stop_strings.is_empty() {
                    let accumulated: String = generated.iter().map(|(_, t)| t.as_str()).collect::<String>() + &text;
                    if let Some(cut) = find_text_stop(&accumulated, &params.stop_strings) {
                        // Truncate at the stop pattern boundary.
                        let safe = &accumulated[..cut];
                        // Re-emit only the safe portion as a single final token.
                        if !safe.is_empty() {
                            // Replace the already-collected tokens with a single truncated entry.
                            generated.clear();
                            generated.push((next_id, safe.to_string()));
                        } else {
                            generated.clear();
                        }
                        break 'gen;
                    }
                }

                generated.push((next_id, text));
                last_id = next_id;
                pos += 1;
            }
        }

        // Send all collected tokens to the channel.
        for (id, text) in generated {
            if !text.is_empty() {
                let _ = tx.send(Ok(Token { id, text, is_eos: false }));
            }
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
        if seq_len == 0 {
            return Err(CoreError::Config {
                field: "token_ids".to_string(),
                reason: "Prompt encoded to zero tokens — check tokenizer or prompt input"
                    .to_string(),
            });
        }
        if seq_len > self.max_sequence_length {
            return Err(CoreError::SequenceTooLong {
                got: seq_len,
                max: self.max_sequence_length,
            });
        }

        // Build a HashSet once so every per-token stop-ID check is O(1).
        let stop_ids: HashSet<u32> = params.stop_token_ids.iter().copied().collect();

        let mut token_history = input.token_ids.clone();

        // Clear any append-style KV caches (e.g. Qwen3 ConcatKvCache) so that
        // stale entries from a previous turn don't widen the KV dimension beyond
        // what the causal mask expects when offset=0.
        self.weights.clear_kv_cache();

        let input_tensor = Tensor::new(input.token_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.weights.forward(&input_tensor, 0)?;
        let last_logits = last_token_logits(&logits, seq_len - 1)?;

        let first_id = sample_token(&last_logits, &params, &token_history)?;
        token_history.push(first_id);

        // Don't send stop tokens to output
        if stop_ids.contains(&first_id) {
            return Ok(());
        }

        let first_text = self.tokenizer.decode_token(first_id)?;

        // Track accumulated text for multi-token text stop detection.
        // We keep only a rolling suffix of max_stop_len characters to bound memory.
        let max_stop_len: usize = params.stop_strings.iter().map(|s| s.len()).max().unwrap_or(0) + 64;
        let mut accumulated = String::new();

        // think_closed = false → inside think block; text stops suppressed.
        // think_closed = true  → outside (or not a thinking model); stops active.
        let mut think_closed = !params.thinking_mode;
        // Check if the first token itself closes the think block.
        if !think_closed && first_text.contains("</think>") {
            think_closed = true;
        }

        // Check the very first token against text stops before sending
        // (only when outside a think block).
        let first_safe = if think_closed && !params.stop_strings.is_empty() {
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
            true
        } else {
            true
        };

        if first_safe {
            if accumulated.is_empty() { accumulated.push_str(&first_text); }
            if tx.send(Ok(Token { id: first_id, text: first_text, is_eos: false })).is_err() {
                return Ok(());
            }
        }

        // Trim accumulated buffer to bound its size.
        if accumulated.len() > max_stop_len {
            let trim_at = accumulated.len() - max_stop_len;
            accumulated.drain(..trim_at);
        }

        if params.max_new_tokens <= 1 {
            return Ok(());
        }

        let mut pos = seq_len;
        let mut last_id = first_id;

        'gen: for _ in 1..params.max_new_tokens {
            let step_input = Tensor::new(&[last_id], &self.device)?.unsqueeze(0)?;
            let step_logits = self.weights.forward(&step_input, pos)?;
            let step_last = last_token_logits(&step_logits, 0)?;
            let next_id = sample_token(&step_last, &params, &token_history)?;

            // Cap history to the penalty window — same reason as generate().
            let cap = params.repeat_last_n.unwrap_or(usize::MAX);
            if token_history.len() >= cap {
                token_history.drain(..token_history.len() - cap + 1);
            }
            token_history.push(next_id);

            if stop_ids.contains(&next_id) {
                break 'gen;
            }

            let text = self.tokenizer.decode_token(next_id)?;

            // Track think block closure so text stops are only active outside it.
            if !think_closed && text.contains("</think>") {
                think_closed = true;
            }

            // Text-based stop detection against the rolling suffix buffer,
            // only applied once the think block has closed.
            if think_closed && !params.stop_strings.is_empty() {
                accumulated.push_str(&text);
                if let Some(cut_in_suffix) = find_text_stop(&accumulated, &params.stop_strings) {
                    tracing::debug!(
                        cut = cut_in_suffix,
                        context = %&accumulated[..cut_in_suffix.min(accumulated.len())],
                        "Text stop matched during generation"
                    );
                    // Compute how many chars of the suffix are safe to emit.
                    // The safe portion is everything before the stop pattern
                    // that was NOT yet sent (i.e., the current token's content).
                    let already_safe_len = accumulated.len() - text.len();
                    if cut_in_suffix > already_safe_len {
                        let safe_part = &accumulated[already_safe_len..cut_in_suffix];
                        if !safe_part.is_empty() {
                            let _ = tx.send(Ok(Token { id: next_id, text: safe_part.to_string(), is_eos: false }));
                        }
                    }
                    break 'gen;
                }
                // Trim the buffer to avoid unbounded growth.
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
        }

        Ok(())
    }
}
