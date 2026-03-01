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

    /// The GGUF file uses an IQ/imatrix quantization type that candle does not implement.
    /// Users must download a standard Q4_K_M / Q6_K / Q8_0 file instead.
    #[error(
        "Unsupported quantization in GGUF file: {detail}\n\
         Hint: candle does not support IQ-family (imatrix) quant types such as IQ4_XS, IQ3_S, IQ2_XXS, etc.\n\
         These are used by Unsloth \"UD\" (Unsloth Dynamic) files (e.g. *-UD-Q4_K_XL.gguf).\n\
         Download a standard quant instead, for example:\n\
           xandllm pull {repo}:Q4_K_M\n\
           xandllm pull {repo}:Q6_K\n\
           xandllm pull {repo}:Q8_0"
    )]
    UnsupportedQuantization { detail: String, repo: String },
}

pub type CoreResult<T> = Result<T, CoreError>;
