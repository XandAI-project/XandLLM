use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Model config error: {field} — {reason}")]
    Config { field: String, reason: String },

    #[error("Unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("Sequence too long: {got} tokens, max {max}")]
    SequenceTooLong { got: usize, max: usize },

    #[error("Device not available: {0}")]
    DeviceUnavailable(String),

    #[error("Channel send error: generation stream closed")]
    ChannelClosed,
}

pub type CoreResult<T> = Result<T, CoreError>;
