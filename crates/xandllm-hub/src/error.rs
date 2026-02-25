use thiserror::Error;

#[derive(Debug, Error)]
pub enum HubError {
    #[error("Hugging Face API error: {0}")]
    Api(#[from] hf_hub::api::tokio::ApiError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Integrity check failed for '{file}': expected {expected}, got {actual}")]
    IntegrityMismatch {
        file: String,
        expected: String,
        actual: String,
    },

    #[error("Model not found in cache: {0}")]
    NotCached(String),

    #[error("Invalid cache directory: {0}")]
    InvalidCacheDir(String),

    #[error("Download failed for '{file}': {reason}")]
    DownloadFailed { file: String, reason: String },
}

pub type HubResult<T> = Result<T, HubError>;
