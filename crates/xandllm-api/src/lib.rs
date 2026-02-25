//! # xandllm-api
//!
//! OpenAI-compatible HTTP API server for XandLLM.
//!
//! ## Endpoints
//!
//! | Method | Path | Description |
//! |---|---|---|
//! | `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming) |
//! | `POST` | `/v1/completions` | Text completions (streaming + non-streaming) |
//! | `GET` | `/v1/models` | List loaded models |
//! | `GET` | `/health` | Liveness probe |

pub mod error;
pub mod middleware;
pub mod routes;
pub mod server;
pub mod streaming;
pub mod types;

#[cfg(test)]
mod tests;

pub use server::{serve, build_router, ServerConfig};
pub use error::{ApiError, ApiResult};
