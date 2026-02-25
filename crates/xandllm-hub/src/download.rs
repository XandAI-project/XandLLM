use std::path::PathBuf;

use futures::StreamExt;
use hf_hub::{
    api::tokio::{Api, ApiBuilder},
    Repo, RepoType,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use tracing::{debug, info, warn};

use crate::{
    cache::{FileMetadata, ModelCache},
    error::{HubError, HubResult},
    integrity::sha256_file,
};

/// Metadata and tokenizer files present in most HF model repos.
const DEFAULT_MODEL_FILES: &[&str] = &[
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
];

const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";
const SAFETENSORS_SINGLE: &str = "model.safetensors";

/// Progress bar template for files whose `Content-Length` is known.
const PB_TEMPLATE_SIZED: &str =
    "{wide_msg}\n[{bar:50.cyan/blue}] {bytes}/{total_bytes}  {bytes_per_sec}  ETA {eta}";

/// Progress bar template when the total size is not known.
const PB_TEMPLATE_SPINNER: &str =
    "{spinner:.green} {wide_msg}  {bytes}  {bytes_per_sec}  [{elapsed_precise}]";

// ─── Downloader ───────────────────────────────────────────────────────────────

/// High-level model downloader.
pub struct ModelDownloader {
    cache: ModelCache,
    api: Api,
    /// Authenticated reqwest client for streaming downloads.
    http: reqwest::Client,
    /// HF Hub bearer token, if available.
    hf_token: Option<String>,
}

impl ModelDownloader {
    /// Create a downloader using the default cache directory.
    ///
    /// Reads `HUGGING_FACE_HUB_TOKEN` from the environment if present.
    pub fn new(cache: ModelCache) -> HubResult<Self> {
        let hf_token = std::env::var("HUGGING_FACE_HUB_TOKEN")
            .ok()
            .filter(|t| !t.is_empty());

        let mut api_builder = ApiBuilder::new();
        if let Some(ref token) = hf_token {
            api_builder = api_builder.with_token(Some(token.clone()));
        }
        let api = api_builder.build().map_err(|e| HubError::DownloadFailed {
            file: "<api init>".into(),
            reason: e.to_string(),
        })?;

        let http = reqwest::Client::builder()
            .user_agent("xandllm/0.1")
            .build()
            .map_err(|e| HubError::DownloadFailed {
                file: "<http client>".into(),
                reason: e.to_string(),
            })?;

        Ok(Self { cache, api, http, hf_token })
    }

    // ─── Public entry points ──────────────────────────────────────────────────

    /// Download all files for a model repository (safetensors format).
    pub async fn pull(
        &self,
        repo_id: &str,
        revision: &str,
        mp: Option<&MultiProgress>,
    ) -> HubResult<Vec<PathBuf>> {
        info!(repo_id, revision, "Pulling model");

        let repo = self.api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let mut filenames: Vec<String> = DEFAULT_MODEL_FILES
            .iter()
            .map(|s| s.to_string())
            .collect();

        match repo.get(SAFETENSORS_INDEX).await {
            Ok(index_path) => {
                let index_str = std::fs::read_to_string(&index_path).map_err(HubError::Io)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)?;
                if let Some(weight_map) = index["weight_map"].as_object() {
                    let mut shards: Vec<String> = weight_map
                        .values()
                        .filter_map(|v| v.as_str())
                        .map(String::from)
                        .collect::<std::collections::HashSet<_>>()
                        .into_iter()
                        .collect();
                    shards.sort();
                    filenames.extend(shards);
                    filenames.push(SAFETENSORS_INDEX.to_string());
                }
            }
            Err(_) => {
                filenames.push(SAFETENSORS_SINGLE.to_string());
            }
        }

        filenames.dedup();
        self.download_all(&repo, repo_id, revision, &filenames, mp).await
    }

    /// Download all GGUF shard files plus associated tokenizer files.
    ///
    /// If `gguf_filename` is `Some`, only that exact file is used (no discovery).
    /// If `gguf_filename` is `None`, the repo is queried and `quant_tag`
    /// (e.g. `"Q4_0"`, `"Q4_K_M"`) is used to select the best matching files.
    pub async fn pull_gguf(
        &self,
        repo_id: &str,
        revision: &str,
        gguf_filename: Option<&str>,
        quant_tag: Option<&str>,
        mp: Option<&MultiProgress>,
    ) -> HubResult<Vec<PathBuf>> {
        let gguf_files: Vec<String> = match gguf_filename {
            Some(explicit) => vec![explicit.to_string()],
            None => self.discover_gguf_filenames(repo_id, revision, quant_tag).await?,
        };

        info!(repo_id, revision, files = gguf_files.len(), "Pulling GGUF model");
        for f in &gguf_files {
            info!(filename = %f, "Will download");
        }

        let repo = self.api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let mut filenames: Vec<String> = DEFAULT_MODEL_FILES
            .iter()
            .map(|s| s.to_string())
            .collect();
        filenames.extend(gguf_files);

        // Tokenizer files are optional — if the GGUF repo does not include them
        // the tokenizer will be built from the vocabulary embedded in the GGUF
        // metadata at load time (see xandllm-core::Tokenizer::from_gguf_metadata).
        let paths = self.download_all(&repo, repo_id, revision, &filenames, mp).await?;
        Ok(paths)
    }

    // ─── Discovery ────────────────────────────────────────────────────────────

    /// Query the repo file listing and return the minimal set of `.gguf` files
    /// needed for the requested quantization.
    ///
    /// Priority (for any given quant tag):
    /// 1. A single complete file (no `-NNNNN-of-MMMMM` suffix) — preferred.
    /// 2. All matching shard files when no single-file variant exists.
    ///
    /// If `quant_tag` is `None` and the repo has exactly one `.gguf` file, it
    /// is returned; otherwise an error lists available quantizations.
    async fn discover_gguf_filenames(
        &self,
        repo_id: &str,
        revision: &str,
        quant_tag: Option<&str>,
    ) -> HubResult<Vec<String>> {
        let repo = self.api.repo(Repo::with_revision(
            repo_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let info = repo.info().await.map_err(|e| {
            let hint = if e.to_string().contains("401") || e.to_string().contains("403") {
                " — this repo may require authentication: set HUGGING_FACE_HUB_TOKEN"
            } else {
                ""
            };
            HubError::DownloadFailed {
                file: format!("<repo info: {repo_id}>"),
                reason: format!("{e}{hint}"),
            }
        })?;

        let mut all_gguf: Vec<String> = info
            .siblings
            .iter()
            .filter(|s| s.rfilename.ends_with(".gguf"))
            .map(|s| s.rfilename.clone())
            .collect();
        all_gguf.sort();

        if all_gguf.is_empty() {
            return Err(HubError::DownloadFailed {
                file: repo_id.to_string(),
                reason: "No .gguf files found in this repository".to_string(),
            });
        }

        if let Some(tag) = quant_tag {
            let tag_lower = tag.to_lowercase();
            let matched: Vec<String> = all_gguf
                .iter()
                .filter(|f| f.to_lowercase().contains(&tag_lower))
                .cloned()
                .collect();

            if matched.is_empty() {
                let available = unique_quant_labels(&all_gguf);
                return Err(HubError::DownloadFailed {
                    file: repo_id.to_string(),
                    reason: format!(
                        "No .gguf file matching '{}' in '{}'.\nAvailable quantizations:\n{}",
                        tag,
                        repo_id,
                        available.iter().map(|t| format!("  {t}")).collect::<Vec<_>>().join("\n")
                    ),
                });
            }

            return Ok(prefer_single_over_shards(matched));
        }

        // No tag: single file → return it; otherwise list available options
        if all_gguf.len() == 1 {
            return Ok(all_gguf);
        }

        let available = unique_quant_labels(&all_gguf);
        Err(HubError::DownloadFailed {
            file: repo_id.to_string(),
            reason: format!(
                "Multiple quantizations available in '{}'. Specify one via <model-id>:<quant> or --gguf-file.\nAvailable:\n{}",
                repo_id,
                available.iter().map(|t| format!("  {t}")).collect::<Vec<_>>().join("\n")
            ),
        })
    }

    // ─── Download engine ──────────────────────────────────────────────────────

    /// Download every filename in the list, warning on optional failures.
    /// Returns `Err` if ALL required (non-optional) files fail.
    async fn download_all(
        &self,
        repo: &hf_hub::api::tokio::ApiRepo,
        repo_id: &str,
        revision: &str,
        filenames: &[String],
        mp: Option<&MultiProgress>,
    ) -> HubResult<Vec<PathBuf>> {
        let mut local_paths = Vec::new();
        let mut failures = 0usize;

        for filename in filenames {
            match self.download_file(repo, repo_id, revision, filename, mp).await {
                Ok(p) => local_paths.push(p),
                Err(e) => {
                    let is_optional = DEFAULT_MODEL_FILES.contains(&filename.as_str());
                    if is_optional {
                        warn!(filename, error = %e, "Optional file not available, skipping");
                    } else {
                        warn!(filename, error = %e, "Failed to download required file");
                        failures += 1;
                    }
                }
            }
        }

        if local_paths.is_empty() && failures > 0 {
            return Err(HubError::DownloadFailed {
                file: repo_id.to_string(),
                reason: format!(
                    "All {failures} required file(s) failed to download. \
                    Check your HUGGING_FACE_HUB_TOKEN if this is a gated repo."
                ),
            });
        }

        Ok(local_paths)
    }

    /// Download a single file with a real-time progress bar.
    ///
    /// Uses `repo.url()` to build the CDN URL, then streams the body via
    /// `reqwest` so every received chunk advances the progress bar immediately.
    /// Writes to `<dest>.tmp` and renames atomically on success.
    async fn download_file(
        &self,
        repo: &hf_hub::api::tokio::ApiRepo,
        repo_id: &str,
        revision: &str,
        filename: &str,
        mp: Option<&MultiProgress>,
    ) -> HubResult<PathBuf> {
        let dest = self.cache.file_path(repo_id, revision, filename);

        if self.cache.is_cached(repo_id, revision, filename) {
            debug!(filename, "Already cached, skipping download");
            return Ok(dest);
        }

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(HubError::Io)?;
        }

        let url = repo.url(filename);
        info!(filename, "Downloading");

        // ── Build request with optional auth ──────────────────────────────────
        let mut req = self.http.get(&url);
        if let Some(ref token) = self.hf_token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }

        let response = req.send().await.map_err(|e| {
            let hint = auth_hint(e.to_string().as_str());
            HubError::DownloadFailed {
                file: filename.to_string(),
                reason: format!("{e}{hint}"),
            }
        })?;

        let status = response.status();
        if !status.is_success() {
            let hint = if status.as_u16() == 401 || status.as_u16() == 403 {
                " (hint: set HUGGING_FACE_HUB_TOKEN for gated repos)"
            } else {
                ""
            };
            return Err(HubError::DownloadFailed {
                file: filename.to_string(),
                reason: format!("HTTP {status}{hint}"),
            });
        }

        // ── Set up progress bar ───────────────────────────────────────────────
        let content_length = response.content_length();
        let pb: Option<ProgressBar> = mp.map(|m| {
            if let Some(total) = content_length {
                let pb = m.add(ProgressBar::new(total));
                pb.set_style(
                    ProgressStyle::with_template(PB_TEMPLATE_SIZED)
                        .unwrap()
                        .progress_chars("##-"),
                );
                pb.set_message(filename.to_string());
                pb
            } else {
                let pb = m.add(ProgressBar::new_spinner());
                pb.set_style(
                    ProgressStyle::with_template(PB_TEMPLATE_SPINNER).unwrap(),
                );
                pb.set_message(filename.to_string());
                pb.enable_steady_tick(std::time::Duration::from_millis(100));
                pb
            }
        });

        // ── Stream to temp file ───────────────────────────────────────────────
        let tmp_path = dest.with_extension("gguf.tmp").with_file_name(format!(
            "{}.tmp",
            dest.file_name().unwrap_or_default().to_string_lossy()
        ));

        let result = self
            .stream_to_disk(response, &tmp_path, pb.as_ref())
            .await;

        match result {
            Ok(()) => {
                std::fs::rename(&tmp_path, &dest).map_err(HubError::Io)?;
            }
            Err(e) => {
                let _ = std::fs::remove_file(&tmp_path);
                return Err(e);
            }
        }

        // ── Metadata ──────────────────────────────────────────────────────────
        let sha256 = sha256_file(&dest)?;
        let size_bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        self.cache.write_meta(&FileMetadata {
            repo_id: repo_id.to_string(),
            revision: revision.to_string(),
            filename: filename.to_string(),
            sha256,
            size_bytes,
        })?;

        if let Some(ref pb) = pb {
            pb.finish_with_message(format!("Done  {filename}"));
        }

        Ok(dest)
    }

    /// Write a streaming HTTP response body to `path`, calling `pb.inc()` per chunk.
    async fn stream_to_disk(
        &self,
        response: reqwest::Response,
        path: &PathBuf,
        pb: Option<&ProgressBar>,
    ) -> HubResult<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path).map_err(HubError::Io)?;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| HubError::DownloadFailed {
                file: path.to_string_lossy().to_string(),
                reason: e.to_string(),
            })?;
            file.write_all(&chunk).map_err(HubError::Io)?;
            if let Some(pb) = pb {
                pb.inc(chunk.len() as u64);
            }
        }

        file.flush().map_err(HubError::Io)?;
        Ok(())
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Given a list of matching GGUF filenames, return only the single complete
/// files (no `-NNNNN-of-MMMMM` suffix). Falls back to returning all files
/// (shards) if no single-file variant exists.
fn prefer_single_over_shards(files: Vec<String>) -> Vec<String> {
    let singles: Vec<String> = files
        .iter()
        .filter(|f| !is_shard(f))
        .cloned()
        .collect();

    if singles.is_empty() {
        files // all are shards — return them all
    } else {
        singles
    }
}

/// Returns `true` if the filename has a `-NNNNN-of-MMMMM` shard pattern.
fn is_shard(filename: &str) -> bool {
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    // Look for "-0000N-of-0000M" anywhere in the stem
    stem.contains("-of-")
        && stem
            .split('-')
            .any(|part| part.len() == 5 && part.chars().all(|c| c.is_ascii_digit()))
}

/// Derive human-readable quantization labels from a list of GGUF filenames.
///
/// Strips the shard suffix (`-00001-of-00002`) so multi-part models show as
/// a single label (e.g. `Q4_0` rather than `Q4_0-00001-OF-00002`).
fn unique_quant_labels(filenames: &[String]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    for name in filenames {
        let stem = name.strip_suffix(".gguf").unwrap_or(name);
        // Strip shard suffix
        let label = if let Some(idx) = stem.rfind("-0000") {
            &stem[..idx]
        } else {
            stem
        };
        let tag = label.rsplit('-').next().unwrap_or(label);
        seen.insert(tag.to_uppercase());
    }
    seen.into_iter().collect()
}

fn auth_hint(msg: &str) -> &'static str {
    if msg.contains("401") || msg.contains("403") {
        " (hint: set HUGGING_FACE_HUB_TOKEN for gated repos)"
    } else {
        ""
    }
}
