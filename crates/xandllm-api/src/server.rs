use std::{net::SocketAddr, sync::Arc, time::Duration};

/// Newtype wrapper for the loaded model's repository identifier.
///
/// Using a distinct type instead of bare `Arc<String>` prevents Axum's
/// type-based Extension lookup from colliding with other `Arc<String>`
/// extensions (specifically `ChatFormat`).
#[derive(Debug, Clone)]
pub struct ModelId(pub Arc<String>);

/// Newtype wrapper for the detected chat template format string (e.g. `"gemma"`, `"chatml"`).
///
/// Same motivation as `ModelId` — keeps the Axum Extension type unique so
/// handlers can extract both values without one overwriting the other.
#[derive(Debug, Clone)]
pub struct ChatFormat(pub Arc<String>);

use axum::{
    middleware,
    routing::{get, post},
    Extension, Router,
};
use tokio::sync::Mutex;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};
use tracing::{info, Level};

use xandllm_core::Model;

use crate::{
    middleware::request_id_middleware,
    routes::{
        chat::create_chat_completion,
        completions::create_completion,
        models::{health, list_models},
    },
};

/// Configuration for the API server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub request_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 11435,
            request_timeout_secs: 120,
        }
    }
}

/// Build the axum router with all OpenAI-compatible routes and middleware.
pub fn build_router(
    model: Arc<Mutex<dyn Model + Send>>,
    tokenizer: Arc<xandllm_core::Tokenizer>,
    model_id: Arc<String>,
    chat_format: String,
    timeout_secs: u64,
) -> Router {
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
        .on_response(DefaultOnResponse::new().level(Level::INFO));

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/v1/chat/completions", post(create_chat_completion))
        .route("/v1/completions", post(create_completion))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(Extension(model))
        .layer(Extension(tokenizer))
        .layer(Extension(ModelId(model_id)))
        .layer(Extension(ChatFormat(Arc::new(chat_format))))
        .layer(middleware::from_fn(request_id_middleware))
        .layer(TimeoutLayer::new(Duration::from_secs(timeout_secs)))
        .layer(cors)
        .layer(trace_layer)
}

/// Start the HTTP server and block until a shutdown signal is received.
pub async fn serve(
    model: Arc<Mutex<dyn Model + Send>>,
    tokenizer: Arc<xandllm_core::Tokenizer>,
    model_id: Arc<String>,
    chat_format: String,
    config: ServerConfig,
) -> anyhow::Result<()> {
    let router = build_router(model, tokenizer, model_id, chat_format, config.request_timeout_secs);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    info!(address = %addr, "Starting XandLLM API server");

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shut down gracefully");
    Ok(())
}

/// Resolves on SIGINT (Ctrl-C) or SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => info!("Received Ctrl-C, shutting down"),
        _ = terminate => info!("Received SIGTERM, shutting down"),
    }
}
