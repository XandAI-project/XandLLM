use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::error::{HubError, HubResult};

/// Metadata stored alongside each cached file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub repo_id: String,
    pub revision: String,
    pub filename: String,
    pub sha256: String,
    pub size_bytes: u64,
}

/// Manages the on-disk model cache layout.
///
/// Layout: `<cache_dir>/<repo_id>/<revision>/<filename>`
/// Metadata sidecar: `<filename>.meta.json`
#[derive(Debug, Clone)]
pub struct ModelCache {
    root: PathBuf,
}

impl ModelCache {
    /// Create a cache rooted at the given directory, creating it if needed.
    pub fn new(root: impl Into<PathBuf>) -> HubResult<Self> {
        let root: PathBuf = root.into();
        let expanded = expand_tilde(&root);
        std::fs::create_dir_all(&expanded).map_err(HubError::Io)?;
        Ok(Self { root: expanded })
    }

    /// Default cache directory: `~/.cache/xandllm`.
    pub fn default_cache() -> HubResult<Self> {
        let home = dirs::home_dir().ok_or_else(|| {
            HubError::InvalidCacheDir("Cannot determine home directory".to_string())
        })?;
        Self::new(home.join(".cache").join("xandllm"))
    }

    /// Absolute path for a given `(repo_id, revision, filename)` tuple.
    pub fn file_path(&self, repo_id: &str, revision: &str, filename: &str) -> PathBuf {
        self.root
            .join(repo_id.replace('/', "__"))
            .join(revision)
            .join(filename)
    }

    /// Absolute path for the metadata sidecar of a given file.
    pub fn meta_path(&self, repo_id: &str, revision: &str, filename: &str) -> PathBuf {
        let mut p = self.file_path(repo_id, revision, filename);
        let fname = format!("{}.meta.json", filename);
        p.set_file_name(fname);
        p
    }

    /// Returns `true` if the file is already cached (both data and meta exist).
    pub fn is_cached(&self, repo_id: &str, revision: &str, filename: &str) -> bool {
        self.file_path(repo_id, revision, filename).exists()
            && self.meta_path(repo_id, revision, filename).exists()
    }

    /// Persist file metadata to disk.
    pub fn write_meta(&self, meta: &FileMetadata) -> HubResult<()> {
        let path = self.meta_path(&meta.repo_id, &meta.revision, &meta.filename);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(HubError::Io)?;
        }
        let json = serde_json::to_string_pretty(meta)?;
        std::fs::write(&path, json).map_err(HubError::Io)?;
        debug!("Wrote metadata to {}", path.display());
        Ok(())
    }

    /// Read file metadata from disk.
    pub fn read_meta(&self, repo_id: &str, revision: &str, filename: &str) -> HubResult<FileMetadata> {
        let path = self.meta_path(repo_id, revision, filename);
        let json = std::fs::read_to_string(&path).map_err(HubError::Io)?;
        let meta: FileMetadata = serde_json::from_str(&json)?;
        Ok(meta)
    }

    /// List all cached models as `(repo_id, revision)` pairs.
    pub fn list_models(&self) -> HubResult<Vec<(String, String)>> {
        let mut results = Vec::new();
        if !self.root.exists() {
            return Ok(results);
        }
        for entry in std::fs::read_dir(&self.root).map_err(HubError::Io)? {
            let entry = entry.map_err(HubError::Io)?;
            let repo_dir = entry.path();
            if !repo_dir.is_dir() {
                continue;
            }
            let repo_id = entry.file_name().to_string_lossy().replace("__", "/");
            for rev_entry in std::fs::read_dir(&repo_dir).map_err(HubError::Io)? {
                let rev_entry = rev_entry.map_err(HubError::Io)?;
                if rev_entry.path().is_dir() {
                    let revision = rev_entry.file_name().to_string_lossy().to_string();
                    results.push((repo_id.clone(), revision));
                }
            }
        }
        Ok(results)
    }

    /// The directory where all files for a given `(repo_id, revision)` live.
    ///
    /// This is the canonical model directory to pass to inference code.
    pub fn model_dir(&self, repo_id: &str, revision: &str) -> PathBuf {
        self.root
            .join(repo_id.replace('/', "__"))
            .join(revision)
    }

    /// Delete all cached files for a given `(repo_id, revision)`.
    ///
    /// Returns the number of files removed, or an error if the model is not
    /// cached.  If `revision` is `None`, all revisions are deleted.
    pub fn delete_model(&self, repo_id: &str, revision: Option<&str>) -> HubResult<usize> {
        let repo_slug = repo_id.replace('/', "__");
        let repo_root = self.root.join(&repo_slug);

        if !repo_root.exists() {
            return Err(HubError::InvalidCacheDir(format!(
                "Model '{}' is not cached in {}",
                repo_id,
                self.root.display()
            )));
        }

        let dirs_to_remove: Vec<PathBuf> = match revision {
            Some(rev) => {
                let rev_dir = repo_root.join(rev);
                if !rev_dir.exists() {
                    return Err(HubError::InvalidCacheDir(format!(
                        "Model '{}' @ '{}' is not cached in {}",
                        repo_id,
                        rev,
                        self.root.display()
                    )));
                }
                vec![rev_dir]
            }
            None => std::fs::read_dir(&repo_root)
                .map_err(HubError::Io)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect(),
        };

        let mut count = 0usize;
        for dir in &dirs_to_remove {
            count += count_files(dir);
            std::fs::remove_dir_all(dir).map_err(HubError::Io)?;
        }

        // Remove the repo root directory if it is now empty
        if std::fs::read_dir(&repo_root)
            .map(|mut d| d.next().is_none())
            .unwrap_or(true)
        {
            let _ = std::fs::remove_dir(&repo_root);
        }

        Ok(count)
    }

    /// The root cache directory.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Recursively count files in a directory (excludes directories themselves).
fn count_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| {
            if e.path().is_dir() {
                count_files(&e.path())
            } else {
                1
            }
        })
        .sum()
}

fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with('~') {
        if let Some(home) = dirs::home_dir() {
            return home.join(&s[2..]);
        }
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    fn temp_cache(suffix: &str) -> (ModelCache, std::path::PathBuf) {
        let dir = temp_dir().join(format!("xandllm_cache_test_{suffix}"));
        let cache = ModelCache::new(&dir).unwrap();
        (cache, dir)
    }

    #[test]
    fn test_file_path_encodes_slash() {
        let (cache, dir) = temp_cache("fp");
        let p = cache.file_path("owner/repo", "main", "config.json");
        let s = p.to_string_lossy();
        assert!(s.contains("owner__repo"), "slash must become double-underscore");
        assert!(s.contains("main"));
        assert!(s.contains("config.json"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_meta_path_has_suffix() {
        let (cache, dir) = temp_cache("mp");
        let p = cache.meta_path("a/b", "rev", "weights.safetensors");
        assert!(
            p.to_string_lossy().ends_with("weights.safetensors.meta.json"),
            "meta path should end with <filename>.meta.json"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_is_cached_false_initially() {
        let (cache, dir) = temp_cache("ic");
        assert!(!cache.is_cached("test/model", "main", "weights.bin"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_metadata_write_read_roundtrip() {
        let (cache, dir) = temp_cache("meta");
        let meta = FileMetadata {
            repo_id: "test/model".to_string(),
            revision: "main".to_string(),
            filename: "config.json".to_string(),
            sha256: "deadbeef".to_string(),
            size_bytes: 1024,
        };
        cache.write_meta(&meta).unwrap();
        let read_back = cache.read_meta("test/model", "main", "config.json").unwrap();
        assert_eq!(read_back.sha256, "deadbeef");
        assert_eq!(read_back.size_bytes, 1024);
        assert_eq!(read_back.repo_id, "test/model");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_list_models_empty_cache() {
        let (cache, dir) = temp_cache("list_empty");
        let models = cache.list_models().unwrap();
        assert!(models.is_empty(), "fresh cache should have no models");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_list_models_after_creating_revision_dir() {
        let (cache, dir) = temp_cache("list_models");
        // Simulate a cached model by creating the directory structure
        let model_rev_dir = cache.file_path("myorg/mymodel", "v1.0", "");
        std::fs::create_dir_all(&model_rev_dir).unwrap();
        let models = cache.list_models().unwrap();
        assert!(
            models.iter().any(|(r, v)| r == "myorg/mymodel" && v == "v1.0"),
            "listed models should include the created entry, got: {:?}",
            models
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn test_root_returns_correct_path() {
        let (cache, dir) = temp_cache("root");
        assert_eq!(cache.root(), dir.as_path());
        std::fs::remove_dir_all(dir).ok();
    }
}
