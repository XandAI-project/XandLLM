use std::collections::HashMap;
use std::io::Seek;
use std::path::PathBuf;
use std::sync::Arc;

use candle_core::{quantized::gguf_file, Device, IndexOp, Tensor};
use candle_transformers::models::{quantized_llama, quantized_qwen2};
use tokio::sync::mpsc;
use tracing::info;

use crate::{
    error::{CoreError, CoreResult},
    model::{GenerateInput, Model, ModelConfig, SamplingParams, Token},
    sampling::sample_token,
    tokenizer::Tokenizer,
};

// ─── Architecture dispatch ────────────────────────────────────────────────────

/// Wraps either a LLaMA-family or Qwen2 quantized model so both can be used
/// behind the same `forward` call.
enum QuantizedWeights {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
}

impl QuantizedWeights {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(x, index_pos),
            Self::Qwen2(m) => m.forward(x, index_pos),
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
        let content = gguf_file::Content::read(&mut file)?;

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
/// 1. Vocabulary contains `<|im_start|>`  → ChatML  (covers Qwen2, Nanbeige, etc.)
/// 2. Vocabulary contains `<|eot_id|>`   → LLaMA-3 instruct format
/// 3. Fall back to architecture string
fn detect_chat_format(arch: &str, tokenizer: &Tokenizer) -> String {
    if tokenizer.token_id("<|im_start|>").is_some() {
        return "chatml".to_string();
    }
    if tokenizer.token_id("<|eot_id|>").is_some() {
        return "llama3".to_string();
    }
    match arch {
        "qwen2" => "chatml",
        "llama" => "llama2",
        other => other,
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
        info!(
            paths = ?gguf_paths.iter().map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string()).collect::<Vec<_>>(),
            "Loading quantized GGUF model"
        );

        let (content, mut reader) = merge_gguf_shards(&gguf_paths)?;

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

        let weights = match arch.as_str() {
            "qwen2" => {
                let w = quantized_qwen2::ModelWeights::from_gguf(content, &mut reader, device)?;
                QuantizedWeights::Qwen2(w)
            }
            _ => {
                let w = quantized_llama::ModelWeights::from_gguf(content, &mut reader, device)?;
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

        let chat_format = detect_chat_format(&arch, &tokenizer);
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

        let mut token_history = input.token_ids.clone();

        let input_tensor = Tensor::new(input.token_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.weights.forward(&input_tensor, 0)?;
        let last_logits = last_token_logits(&logits, seq_len - 1)?;

        let first_id = sample_token(&last_logits, &params, &token_history)?;
        token_history.push(first_id);

        let first_eos = params.stop_token_ids.contains(&first_id);

        // Don't send stop tokens to output (synchronous generate)
        if first_eos {
            return Ok(rx);
        }

        let first_text = self.tokenizer.decode_token(first_id)?;
        let _ = tx.send(Ok(Token {
            id: first_id,
            text: first_text,
            is_eos: false,
        }));

        if params.max_new_tokens <= 1 {
            return Ok(rx);
        }

        let mut pos = seq_len;
        let mut last_id = first_id;

        for _ in 1..params.max_new_tokens {
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

            let is_eos = params.stop_token_ids.contains(&next_id);

            // Don't send stop tokens to output (synchronous generate)
            if is_eos {
                break;
            }

            let text = self.tokenizer.decode_token(next_id)?;
            if tx.send(Ok(Token { id: next_id, text, is_eos: false })).is_err() {
                break;
            }
            last_id = next_id;
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

        let mut token_history = input.token_ids.clone();

        let input_tensor = Tensor::new(input.token_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.weights.forward(&input_tensor, 0)?;
        let last_logits = last_token_logits(&logits, seq_len - 1)?;

        let first_id = sample_token(&last_logits, &params, &token_history)?;
        token_history.push(first_id);

        let first_eos = params.stop_token_ids.contains(&first_id);

        // Don't send stop tokens to output
        if first_eos {
            return Ok(());
        }

        let first_text = self.tokenizer.decode_token(first_id)?;
        if tx.send(Ok(Token { id: first_id, text: first_text, is_eos: false })).is_err() {
            return Ok(());
        }

        if params.max_new_tokens <= 1 {
            return Ok(());
        }

        let mut pos = seq_len;
        let mut last_id = first_id;

        for _ in 1..params.max_new_tokens {
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

            let is_eos = params.stop_token_ids.contains(&next_id);

            // Don't send stop tokens to output (streaming generate)
            if is_eos {
                break;
            }

            let text = self.tokenizer.decode_token(next_id)?;
            if tx.send(Ok(Token { id: next_id, text, is_eos: false })).is_err() {
                break;
            }
            last_id = next_id;
            pos += 1;
        }

        Ok(())
    }
}
