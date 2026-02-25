//! Dataset loading for distillation.
//!
//! Reads JSONL files from a directory. Each line must be a JSON object with at
//! least `"prompt"` and `"completion"` string fields:
//!
//! ```json
//! {"prompt": "What is 2+2?", "completion": "2+2 equals 4."}
//! ```
//!
//! All `.jsonl` and `.json` files in the directory are concatenated in
//! alphabetical order.

use std::fs;
use std::io::{self, BufRead};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::info;

use xandllm_core::Tokenizer;

/// A single training example.
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub prompt: String,
    pub completion: String,
}

/// A tokenized training batch.
///
/// `input_ids` and `labels` both have shape `[batch, seq_len]`.
/// `completion_mask` is `1.0` for completion tokens and `0.0` for prompt
/// tokens; loss is computed only where the mask is `1.0`.
#[derive(Debug, Clone)]
pub struct TokenizedBatch {
    /// Token ids, shape `[batch, seq_len]`.
    pub input_ids: Vec<Vec<u32>>,
    /// Shifted labels (next-token targets), shape `[batch, seq_len]`.
    pub labels: Vec<Vec<u32>>,
    /// 1.0 for tokens that belong to the *completion* portion; 0.0 otherwise.
    pub completion_mask: Vec<Vec<f32>>,
}

// ── Deserialization helper ────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RawDataPoint {
    prompt: String,
    completion: String,
}

// ── DataLoader ────────────────────────────────────────────────────────────────

/// Loads all JSONL data from a directory and provides batch iteration.
pub struct DataLoader {
    data: Vec<DataPoint>,
}

impl DataLoader {
    /// Scan `dir` for `*.jsonl` and `*.json` files and load all data points.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let mut entries: Vec<_> = fs::read_dir(dir)
            .with_context(|| format!("Cannot read dataset directory: {}", dir.display()))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                s.ends_with(".jsonl") || s.ends_with(".json")
            })
            .map(|e| e.path())
            .collect();

        // Deterministic order
        entries.sort();

        anyhow::ensure!(!entries.is_empty(), "No .jsonl or .json files found in {}", dir.display());

        let mut data = Vec::new();
        for path in &entries {
            let file = fs::File::open(path)
                .with_context(|| format!("Cannot open {}", path.display()))?;
            for (line_no, line) in io::BufReader::new(file).lines().enumerate() {
                let line = line.with_context(|| format!("IO error reading {}", path.display()))?;
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let raw: RawDataPoint = serde_json::from_str(trimmed).with_context(|| {
                    format!(
                        "JSON parse error at {}:{} — expected {{\"prompt\":…,\"completion\":…}}",
                        path.display(),
                        line_no + 1
                    )
                })?;
                data.push(DataPoint { prompt: raw.prompt, completion: raw.completion });
            }
            info!(path = %path.display(), count = data.len(), "Loaded dataset file");
        }

        info!(total = data.len(), "Dataset loaded");
        Ok(Self { data })
    }

    /// Total number of examples in the dataset.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over `DataPoint` slices of `batch_size`.
    pub fn iter_batches(&self, batch_size: usize) -> impl Iterator<Item = &[DataPoint]> {
        self.data.chunks(batch_size)
    }

    /// Return the underlying data as a slice.
    pub fn as_slice(&self) -> &[DataPoint] {
        &self.data
    }

    /// Tokenize a single batch into [`TokenizedBatch`].
    ///
    /// Each example is encoded as `[prompt_tokens..., completion_tokens...]`.
    /// The `completion_mask` is `1.0` for every completion token so the
    /// training loss is computed only over what the student must learn to
    /// generate.  Sequences are right-padded with the EOS token id (or 0) to
    /// the length of the longest sequence in the batch.
    pub fn tokenize_batch(
        batch: &[DataPoint],
        tokenizer: &Arc<Tokenizer>,
        max_seq_len: usize,
    ) -> Result<TokenizedBatch> {
        let pad_id = tokenizer.eos_token_id().unwrap_or(0);

        let mut all_input_ids: Vec<Vec<u32>> = Vec::with_capacity(batch.len());
        let mut all_labels: Vec<Vec<u32>> = Vec::with_capacity(batch.len());
        let mut all_masks: Vec<Vec<f32>> = Vec::with_capacity(batch.len());

        for dp in batch {
            // Tokenize prompt and completion separately so we know the split point.
            // add_special_tokens=false: the full prompt template is already in place.
            let prompt_ids = tokenizer.encode(&dp.prompt, false)?;
            let completion_ids = tokenizer.encode(&dp.completion, false)?;

            let total = prompt_ids.len() + completion_ids.len();
            let seq_len = total.min(max_seq_len);

            // Concatenate then truncate
            let full: Vec<u32> = prompt_ids
                .iter()
                .chain(completion_ids.iter())
                .copied()
                .take(seq_len)
                .collect();

            let prompt_len_capped = prompt_ids.len().min(seq_len);

            // input_ids is the sequence; labels is shifted by 1 (next-token prediction).
            // For position i, label[i] = full[i+1].  The last token has no target, so
            // we drop it from input_ids and take full[1..] as labels.
            let input_ids: Vec<u32> = full[..full.len().saturating_sub(1)].to_vec();
            let labels: Vec<u32> = full[1..].to_vec();

            // Mask: 1.0 for every position where the *label* is a completion token.
            // Labels start at index 1 of `full`, so a label at position j corresponds
            // to full[j+1].  It is a completion token when j+1 >= prompt_len_capped.
            let mask: Vec<f32> = (0..labels.len())
                .map(|j| if j + 1 >= prompt_len_capped { 1.0_f32 } else { 0.0_f32 })
                .collect();

            all_input_ids.push(input_ids);
            all_labels.push(labels);
            all_masks.push(mask);
        }

        // Pad to common length within this batch
        let max_len = all_input_ids.iter().map(|v| v.len()).max().unwrap_or(0);

        for i in 0..all_input_ids.len() {
            let pad = max_len - all_input_ids[i].len();
            all_input_ids[i].extend(std::iter::repeat(pad_id).take(pad));
            all_labels[i].extend(std::iter::repeat(pad_id).take(pad));
            all_masks[i].extend(std::iter::repeat(0.0_f32).take(pad));
        }

        Ok(TokenizedBatch {
            input_ids: all_input_ids,
            labels: all_labels,
            completion_mask: all_masks,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn make_dataset(lines: &[&str]) -> TempDir {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data.jsonl");
        let mut f = fs::File::create(&path).unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        dir
    }

    #[test]
    fn loads_jsonl_correctly() {
        let dir = make_dataset(&[
            r#"{"prompt":"hello","completion":"world"}"#,
            r#"{"prompt":"foo","completion":"bar"}"#,
        ]);
        let loader = DataLoader::from_dir(dir.path()).unwrap();
        assert_eq!(loader.len(), 2);
        assert_eq!(loader.as_slice()[0].prompt, "hello");
        assert_eq!(loader.as_slice()[1].completion, "bar");
    }

    #[test]
    fn skips_blank_lines() {
        let dir = make_dataset(&[
            r#"{"prompt":"a","completion":"b"}"#,
            "",
            r#"{"prompt":"c","completion":"d"}"#,
        ]);
        let loader = DataLoader::from_dir(dir.path()).unwrap();
        assert_eq!(loader.len(), 2);
    }

    #[test]
    fn iter_batches_chunking() {
        let dir = make_dataset(&[
            r#"{"prompt":"1","completion":"a"}"#,
            r#"{"prompt":"2","completion":"b"}"#,
            r#"{"prompt":"3","completion":"c"}"#,
            r#"{"prompt":"4","completion":"d"}"#,
            r#"{"prompt":"5","completion":"e"}"#,
        ]);
        let loader = DataLoader::from_dir(dir.path()).unwrap();
        let batches: Vec<_> = loader.iter_batches(2).collect();
        assert_eq!(batches.len(), 3); // [2, 2, 1]
        assert_eq!(batches[2].len(), 1);
    }

    #[test]
    fn empty_dir_errors() {
        let dir = TempDir::new().unwrap();
        let result = DataLoader::from_dir(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn malformed_json_errors() {
        let dir = make_dataset(&[r#"{"prompt":"a"}"#]); // missing "completion"
        let result = DataLoader::from_dir(dir.path());
        assert!(result.is_err());
    }
}
