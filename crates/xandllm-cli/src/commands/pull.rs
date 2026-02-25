use indicatif::MultiProgress;
use tracing::info;

use xandllm_hub::{ModelCache, ModelDownloader};

use crate::config::{expand_cache_dir, AppConfig};

/// Download a model from Hugging Face Hub into the local cache.
///
/// Accepts two input formats:
/// - `owner/repo`                — plain HF repo ID (safetensors or single GGUF)
/// - `owner/repo:QUANT_TAG`      — selects a specific GGUF quantization
/// - `hf.co/owner/repo:QUANT`   — Ollama-style prefix, stripped automatically
///
/// If the repo name contains "GGUF" or a quant tag is present, `pull_gguf` is
/// used; otherwise all safetensors shards are downloaded.
pub async fn run(
    raw_model_id: &str,
    revision: &str,
    explicit_gguf_file: Option<&str>,
    config: &AppConfig,
) -> anyhow::Result<()> {
    let (repo_id, quant_tag) = parse_model_id(raw_model_id);
    let cache_dir = expand_cache_dir(&config.model.cache_dir);

    info!(
        model_id = raw_model_id,
        repo_id,
        quant_tag,
        revision,
        cache_dir = %cache_dir.display(),
        "Pulling model"
    );

    let cache = ModelCache::new(&cache_dir)?;
    let downloader = ModelDownloader::new(cache)?;
    let mp = MultiProgress::new();

    let use_gguf = explicit_gguf_file.is_some()
        || quant_tag.is_some()
        || repo_id.contains("GGUF")
        || repo_id.contains("gguf");

    let paths = if use_gguf {
        downloader
            .pull_gguf(repo_id, revision, explicit_gguf_file, quant_tag, Some(&mp))
            .await?
    } else {
        downloader.pull(repo_id, revision, Some(&mp)).await?
    };

    println!(
        "\nModel '{}' cached to {}",
        repo_id,
        cache_dir.display()
    );
    println!("{} file(s):", paths.len());
    for p in &paths {
        println!("  {}", p.display());
    }

    Ok(())
}

/// Parse `[hf.co/]owner/repo[:quant_tag]` into `(repo_id, Option<quant_tag>)`.
///
/// Examples:
/// - `"Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"` → `("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", Some("Q4_0"))`
/// - `"hf.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"` → same
/// - `"bartowski/Llama-3.1-8B-GGUF"` → `("bartowski/Llama-3.1-8B-GGUF", None)`
fn parse_model_id(raw: &str) -> (&str, Option<&str>) {
    // Strip optional "hf.co/" prefix used in Ollama-style references
    let stripped = raw.strip_prefix("hf.co/").unwrap_or(raw);

    // Split on the last `:` to separate repo from quantization tag.
    // Guard against Windows absolute paths (C:\...) and HF repo names never
    // contain a colon, so any colon here is a tag separator.
    if let Some(colon) = stripped.rfind(':') {
        let repo = &stripped[..colon];
        let tag = &stripped[colon + 1..];
        // A valid tag is non-empty and contains no path separators
        if !tag.is_empty() && !tag.contains('/') && !tag.contains('\\') {
            return (repo, Some(tag));
        }
    }

    (stripped, None)
}

#[cfg(test)]
mod tests {
    use super::parse_model_id;

    #[test]
    fn plain_repo_id() {
        assert_eq!(parse_model_id("Qwen/Qwen2.5-7B"), ("Qwen/Qwen2.5-7B", None));
    }

    #[test]
    fn repo_with_quant_tag() {
        assert_eq!(
            parse_model_id("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"),
            ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", Some("Q4_0")),
        );
    }

    #[test]
    fn ollama_hf_co_prefix_stripped() {
        assert_eq!(
            parse_model_id("hf.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"),
            ("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", Some("Q4_0")),
        );
    }

    #[test]
    fn ollama_hf_co_without_tag() {
        assert_eq!(
            parse_model_id("hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"),
            ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", None),
        );
    }
}
