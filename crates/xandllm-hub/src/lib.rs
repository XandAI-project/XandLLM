//! # xandllm-hub
//!
//! Hugging Face model downloading and caching for XandLLM.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use xandllm_hub::{ModelCache, ModelDownloader};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let cache = ModelCache::default_cache()?;
//!     let downloader = ModelDownloader::new(cache)?;
//!     let paths = downloader.pull("mistralai/Mistral-7B-v0.1", "main", None).await?;
//!     println!("Downloaded {} files", paths.len());
//!     Ok(())
//! }
//! ```

pub mod cache;
pub mod download;
pub mod error;
pub mod integrity;

pub use cache::{FileMetadata, ModelCache};
pub use download::ModelDownloader;
pub use error::{HubError, HubResult};
