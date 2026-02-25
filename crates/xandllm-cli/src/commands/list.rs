use xandllm_hub::ModelCache;

use crate::config::{expand_cache_dir, AppConfig};

/// List all models stored in the local cache.
pub fn run(config: &AppConfig) -> anyhow::Result<()> {
    let cache_dir = expand_cache_dir(&config.model.cache_dir);
    let cache = ModelCache::new(&cache_dir)?;

    let models = cache.list_models()?;

    if models.is_empty() {
        println!("No models cached in {}.", cache_dir.display());
        println!("Use `xandllm pull <model-id>` to download a model.");
        return Ok(());
    }

    println!("{:<50} {:<20}", "MODEL ID", "REVISION");
    println!("{}", "-".repeat(72));
    for (repo_id, revision) in &models {
        println!("{:<50} {:<20}", repo_id, revision);
    }
    println!("\n{} model(s) cached in {}", models.len(), cache_dir.display());

    Ok(())
}
