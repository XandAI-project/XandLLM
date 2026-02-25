use std::sync::Arc;

use tokio::sync::Mutex;
use tracing::info;

use xandllm_api::{serve, ServerConfig};
use xandllm_core::{select_device, AnyModel, Model, ModelConfig};
use xandllm_hub::ModelCache;

use crate::config::{expand_cache_dir, AppConfig};

/// Start the OpenAI-compatible HTTP server.
///
/// Automatically selects GGUF (quantized) or safetensors loading based on
/// which weight files are present in the model cache directory.
pub async fn run(
    model_id: &str,
    host: Option<&str>,
    port: Option<u16>,
    prefer_gpu: bool,
    config: &AppConfig,
) -> anyhow::Result<()> {
    let cache_dir = expand_cache_dir(&config.model.cache_dir);
    let cache = ModelCache::new(&cache_dir)?;

    let revision = "main";
    let model_dir = cache.model_dir(model_id, revision);

    let device = select_device(
        prefer_gpu || config.device.prefer_gpu,
        config.device.cuda_device_id,
    )?;

    let model_config = ModelConfig {
        repo_id: model_id.to_string(),
        revision: revision.to_string(),
        model_dir: model_dir.clone(),
        max_sequence_length: config.inference.max_sequence_length,
    };

    info!(model_dir = %model_dir.display(), "Loading model for serving");
    let model = AnyModel::load(&model_config, &device)?;

    // Extract the tokenizer Arc and chat format before moving the model into Arc<Mutex<>>.
    // This avoids loading tokenizer.json separately and works for GGUF models
    // whose tokenizer is built from embedded metadata.
    let tokenizer = model.tokenizer_arc();
    let chat_format = model.chat_format().to_string();
    let model: Arc<Mutex<dyn Model + Send>> = Arc::new(Mutex::new(model));
    let model_id_arc = Arc::new(model_id.to_string());

    let server_config = ServerConfig {
        host: host.unwrap_or(&config.server.host).to_string(),
        port: port.unwrap_or(config.server.port),
        request_timeout_secs: config.server.request_timeout_secs,
    };

    info!(
        host = %server_config.host,
        port = server_config.port,
        "Server starting"
    );

    serve(model, tokenizer, model_id_arc, chat_format, server_config).await?;
    Ok(())
}
