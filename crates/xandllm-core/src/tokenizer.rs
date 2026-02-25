use std::collections::HashMap;
use std::path::Path;

use candle_core::quantized::gguf_file;
use tokenizers::{AddedToken, Tokenizer as HfTokenizer};
use tracing::{debug, info};

use crate::error::{CoreError, CoreResult};

/// How individual tokens should be decoded back to text.
///
/// This matters for streaming: when decoding a single token id at a time the
/// HuggingFace `Metaspace` decoder (used for SentencePiece models) strips the
/// leading `▁` because it treats every single-element call as the "first token
/// of a new sequence".  We bypass the decoder pipeline entirely for those and
/// handle the conversion manually.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizerKind {
    /// GPT-2 / Qwen2 style byte-level BPE.  The HF `ByteLevel` decoder works
    /// correctly for single tokens (`Ġ` → space, etc.).
    Gpt2,
    /// SentencePiece Unigram / LLaMA style.  Tokens use `▁` (U+2581) as a
    /// word-boundary marker and `<0xNN>` for byte-fallback characters.
    SentencePiece,
    /// Tokenizer loaded from a `tokenizer.json` file — delegate to HF decoder
    /// as configured in the file.
    File,
}

/// Thin wrapper around the Hugging Face `tokenizers` crate.
#[derive(Debug)]
pub struct Tokenizer {
    inner: HfTokenizer,
    eos_token_id: Option<u32>,
    bos_token_id: Option<u32>,
    kind: TokenizerKind,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &Path) -> CoreResult<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| CoreError::Tokenizer(e.to_string()))?;

        let eos_token_id = Self::special_token_id(&inner, &["</s>", "<|endoftext|>", "<eos>"]);
        let bos_token_id = Self::special_token_id(&inner, &["<s>", "<|startoftext|>", "<bos>"]);

        debug!(
            eos_token_id,
            bos_token_id,
            vocab_size = inner.get_vocab_size(true),
            "Tokenizer loaded from file"
        );

        Ok(Self { inner, eos_token_id, bos_token_id, kind: TokenizerKind::File })
    }

    /// Build a tokenizer from the `tokenizer.ggml.*` metadata embedded in a
    /// GGUF file, without requiring a separate `tokenizer.json`.
    ///
    /// Dispatches on `tokenizer.ggml.model`:
    /// - `"gpt2"`  → byte-level BPE (GPT-2 / Qwen2 style)
    /// - `"llama"` → SentencePiece Unigram (LLaMA / Mistral / Nanbeige style)
    pub fn from_gguf_metadata(
        metadata: &HashMap<String, gguf_file::Value>,
    ) -> CoreResult<Self> {
        let tokens = gguf_string_array(metadata, "tokenizer.ggml.tokens").ok_or_else(|| {
            CoreError::Config {
                field: "tokenizer".to_string(),
                reason: "GGUF file has no tokenizer.ggml.tokens — cannot build tokenizer"
                    .to_string(),
            }
        })?;

        let tokenizer_model = match metadata.get("tokenizer.ggml.model") {
            Some(gguf_file::Value::String(s)) => s.clone(),
            _ => "gpt2".to_string(),
        };

        let merges = gguf_string_array(metadata, "tokenizer.ggml.merges").unwrap_or_default();

        info!(
            vocab_size = tokens.len(),
            merges = merges.len(),
            tokenizer_model = %tokenizer_model,
            "Building tokenizer from GGUF metadata"
        );

        // ── Build the inner model ─────────────────────────────────────────────

        let (mut inner, kind) = match tokenizer_model.as_str() {
            "llama" | "llama2" | "llama3" => {
                // SentencePiece Unigram: uses log-probability scores for
                // Viterbi segmentation — no merge rules needed.
                let scores = gguf_f32_array(metadata, "tokenizer.ggml.scores")
                    .unwrap_or_else(|| vec![0.0f32; tokens.len()]);

                let vocab_with_scores: Vec<(String, f64)> = tokens
                    .iter()
                    .zip(scores.iter().chain(std::iter::repeat(&0.0f32)))
                    .map(|(t, &s)| (t.clone(), s as f64))
                    .collect();

                let unk_id = gguf_u32(metadata, "tokenizer.ggml.unknown_token_id")
                    .map(|id| id as usize);

                let unigram =
                    tokenizers::models::unigram::Unigram::from(vocab_with_scores, unk_id, true)
                        .map_err(|e| CoreError::Tokenizer(format!("Unigram build error: {e}")))?;

                let inner = HfTokenizer::new(unigram);
                (inner, TokenizerKind::SentencePiece)
            }
            _ => {
                // Default: GPT-2 / Qwen2 byte-level BPE.
                let vocab: HashMap<String, u32> = tokens
                    .iter()
                    .enumerate()
                    .map(|(i, t)| (t.clone(), i as u32))
                    .collect();

                let merge_pairs: Vec<(String, String)> = merges
                    .iter()
                    .filter_map(|m| {
                        let mut parts = m.splitn(2, ' ');
                        let a = parts.next()?.to_string();
                        let b = parts.next()?.to_string();
                        Some((a, b))
                    })
                    .collect();

                let bpe = tokenizers::models::bpe::BPE::builder()
                    .vocab_and_merges(vocab, merge_pairs)
                    .byte_fallback(true)
                    .build()
                    .map_err(|e| CoreError::Tokenizer(format!("BPE build error: {e}")))?;

                let inner = HfTokenizer::new(bpe);
                (inner, TokenizerKind::Gpt2)
            }
        };

        // ── Configure pre-tokenizer and decoder ──────────────────────────────

        match kind {
            TokenizerKind::SentencePiece => {
                use tokenizers::decoders::byte_fallback::ByteFallback;
                use tokenizers::decoders::metaspace::Metaspace as MetaspaceDec;
                use tokenizers::decoders::sequence::Sequence as SeqDec;
                use tokenizers::pre_tokenizers::metaspace::{Metaspace as MetaspacePre, PrependScheme};
                // `decode_token` bypasses this decoder for single-token
                // streaming (see that method); the chain is still useful for
                // batch decode (e.g. the API server).
                inner.with_pre_tokenizer(Some(MetaspacePre::new('▁', PrependScheme::First, false)));
                inner.with_decoder(Some(SeqDec::new(vec![
                    tokenizers::DecoderWrapper::ByteFallback(ByteFallback::new()),
                    tokenizers::DecoderWrapper::Metaspace(MetaspaceDec::new(
                        '▁',
                        PrependScheme::First,
                        false,
                    )),
                ])));
            }
            TokenizerKind::Gpt2 if tokenizer_model.as_str() == "gpt2" => {
                use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDec;
                use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
                inner.with_pre_tokenizer(Some(ByteLevelPre::new(false, true, true)));
                inner.with_decoder(Some(ByteLevelDec::new(false, true, true)));
            }
            _ => {}
        }

        // Register special tokens so the tokenizer recognises them.
        //
        // IMPORTANT: special tokens must be registered with `add_special_tokens`
        // so the pre-tokenizer bypasses them and they are encoded as single token
        // IDs rather than being split into byte-level fragments.  Without this,
        // chat-format markers like `<|im_start|>` become garbled multi-token
        // sequences and the model cannot parse conversation structure.
        let eos_token_id = gguf_u32(metadata, "tokenizer.ggml.eos_token_id");
        let bos_token_id = gguf_u32(metadata, "tokenizer.ggml.bos_token_id");

        let mut special_ids: std::collections::HashSet<usize> = std::collections::HashSet::new();

        // Always register EOS and BOS
        for id in [eos_token_id, bos_token_id].into_iter().flatten() {
            special_ids.insert(id as usize);
        }

        // Register tokens flagged as CONTROL (type 3) or USER_DEFINED (type 4)
        // in `tokenizer.ggml.token_type`.
        //
        // NOTE: Qwen2/GPT-style GGUF files store the type array as U32, while
        // some older models use I32.  Both variants must be handled — previously
        // only I32 was checked, causing all Qwen2 special tokens to be missed.
        if let Some(gguf_file::Value::Array(types)) = metadata.get("tokenizer.ggml.token_type") {
            for (i, typ) in types.iter().enumerate() {
                let is_control = match typ {
                    gguf_file::Value::I32(t) => *t == 3 || *t == 4,
                    gguf_file::Value::U32(t) => *t == 3 || *t == 4,
                    gguf_file::Value::U8(t)  => *t == 3 || *t == 4,
                    _ => false,
                };
                if is_control {
                    special_ids.insert(i);
                }
            }
        }

        // Also unconditionally treat any token whose text looks like a chat
        // control marker as special, regardless of the type array.  This covers
        // models that omit or mislabel token_type entries.
        for (i, tok) in tokens.iter().enumerate() {
            if (tok.starts_with("<|") && tok.ends_with("|>"))
                || tok == "<s>"
                || tok == "</s>"
                || tok == "<unk>"
                || tok == "<pad>"
                || (tok.starts_with('<') && tok.ends_with('>') && tok.len() <= 32)
            {
                special_ids.insert(i);
            }
        }

        let special: Vec<AddedToken> = special_ids
            .into_iter()
            .filter_map(|i| tokens.get(i))
            .map(|tok| AddedToken::from(tok.clone(), true))
            .collect();

        if !special.is_empty() {
            info!(count = special.len(), "Registering special tokens in tokenizer");
            inner.add_special_tokens(&special);
        }

        debug!(
            eos_token_id,
            bos_token_id,
            vocab_size = inner.get_vocab_size(true),
            "Tokenizer built from GGUF metadata"
        );

        Ok(Self { inner, eos_token_id, bos_token_id, kind })
    }

    /// Encode a text string to a sequence of token ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> CoreResult<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| CoreError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a sequence of token ids back to a string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> CoreResult<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| CoreError::Tokenizer(e.to_string()))
    }

    /// Decode a single token id to its string representation.
    ///
    /// For SentencePiece tokenizers this bypasses the HuggingFace decoder
    /// pipeline entirely.  The `Metaspace` decoder uses `PrependScheme::First`
    /// which removes the leading `▁` from whichever token it considers the
    /// "first" in a sequence — in single-token calls that is every token, so
    /// every leading `▁` would be dropped instead of being converted to a
    /// space, causing words to run together in streaming output.
    pub fn decode_token(&self, id: u32) -> CoreResult<String> {
        if self.kind == TokenizerKind::SentencePiece {
            let raw = self
                .inner
                .id_to_token(id)
                .ok_or_else(|| CoreError::Tokenizer(format!("Unknown token id: {id}")))?;

            // Byte-fallback tokens: <0xNN> → actual byte value
            if raw.len() == 6 && raw.starts_with("<0x") && raw.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&raw[3..5], 16) {
                    return Ok(String::from_utf8_lossy(&[b]).to_string());
                }
            }

            // ▁ (U+2581) is the SentencePiece word-boundary marker → space
            return Ok(raw.replace('\u{2581}', " "));
        }

        // GPT-2 / ByteLevel: the HF ByteLevel decoder handles single tokens
        // correctly (Ġ → space, Ċ → newline, etc.).
        self.decode(&[id], false)
    }

    /// The end-of-sequence token id, if known.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// The beginning-of-sequence token id, if known.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// The size of the model vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Look up the token id for an exact token string (e.g. `"<|im_end|>"`).
    ///
    /// Returns `None` if the token is not in the vocabulary.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.inner.get_vocab(true).get(text).copied()
    }

    fn special_token_id(tokenizer: &HfTokenizer, candidates: &[&str]) -> Option<u32> {
        let vocab = tokenizer.get_vocab(true);
        for candidate in candidates {
            if let Some(&id) = vocab.get(*candidate) {
                return Some(id);
            }
        }
        None
    }
}

// ─── GGUF metadata helpers ────────────────────────────────────────────────────

/// Extract an array of strings from GGUF metadata.
fn gguf_string_array(
    metadata: &HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<Vec<String>> {
    match metadata.get(key) {
        Some(gguf_file::Value::Array(arr)) => {
            let strings: Vec<String> = arr
                .iter()
                .filter_map(|v| match v {
                    gguf_file::Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            if strings.is_empty() { None } else { Some(strings) }
        }
        _ => None,
    }
}

/// Extract an array of f32 values from GGUF metadata.
fn gguf_f32_array(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Option<Vec<f32>> {
    match metadata.get(key) {
        Some(gguf_file::Value::Array(arr)) => {
            let floats: Vec<f32> = arr
                .iter()
                .filter_map(|v| match v {
                    gguf_file::Value::F32(f) => Some(*f),
                    _ => None,
                })
                .collect();
            if floats.is_empty() { None } else { Some(floats) }
        }
        _ => None,
    }
}

/// Extract a u32 value from GGUF metadata, tolerating different integer widths.
pub fn gguf_u32(metadata: &HashMap<String, gguf_file::Value>, key: &str) -> Option<u32> {
    match metadata.get(key) {
        Some(gguf_file::Value::U32(v)) => Some(*v),
        Some(gguf_file::Value::U64(v)) => Some(*v as u32),
        Some(gguf_file::Value::I32(v)) if *v >= 0 => Some(*v as u32),
        _ => None,
    }
}
