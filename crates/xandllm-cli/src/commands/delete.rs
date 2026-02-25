use xandllm_hub::ModelCache;

use crate::config::{expand_cache_dir, AppConfig};

/// Remove a cached model from local storage.
///
/// If `revision` is `None`, all revisions of the model are deleted.
pub fn run(
    model_id: &str,
    revision: Option<&str>,
    config: &AppConfig,
) -> anyhow::Result<()> {
    let cache_dir = expand_cache_dir(&config.model.cache_dir);
    let cache = ModelCache::new(&cache_dir)?;

    let rev_display = revision.unwrap_or("<all revisions>");
    let count = cache.delete_model(model_id, revision)?;

    println!(
        "Deleted '{}' @ {} — {} file(s) removed from {}",
        model_id,
        rev_display,
        count,
        cache_dir.display()
    );

    Ok(())
}
