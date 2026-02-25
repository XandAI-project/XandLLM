use std::path::PathBuf;

use config::{Config, ConfigError, Environment, File};
use serde::{Deserialize, Serialize};

/// Full runtime configuration loaded from TOML + env vars.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub inference: InferenceConfig,
    pub model: ModelCacheConfig,
    pub device: DeviceConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub request_timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InferenceConfig {
    pub max_batch_size: usize,
    pub max_sequence_length: usize,
    pub default_max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelCacheConfig {
    pub cache_dir: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeviceConfig {
    pub prefer_gpu: bool,
    pub cuda_device_id: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 11435,
                request_timeout_secs: 120,
            },
            inference: InferenceConfig {
                max_batch_size: 8,
                max_sequence_length: 4096,
                default_max_new_tokens: 4096,
                temperature: 0.7,
                top_p: 0.9,
            },
            model: ModelCacheConfig {
                cache_dir: "~/.cache/xandllm".to_string(),
            },
            device: DeviceConfig {
                prefer_gpu: true,
                cuda_device_id: 0,
            },
        }
    }
}

/// Load configuration from:
/// 1. Built-in defaults
/// 2. `config/default.toml` (if present)
/// 3. A custom config file path (if provided)
/// 4. Environment variables prefixed with `XANDLLM_`
pub fn load_config(config_file: Option<&PathBuf>) -> Result<AppConfig, ConfigError> {
    let mut builder = Config::builder()
        // Layer 1: defaults baked in
        .set_default("server.host", "0.0.0.0")?
        .set_default("server.port", 11435_i64)?
        .set_default("server.request_timeout_secs", 120_i64)?
        .set_default("inference.max_batch_size", 8_i64)?
        .set_default("inference.max_sequence_length", 4096_i64)?
        .set_default("inference.default_max_new_tokens", 4096_i64)?
        .set_default("inference.temperature", 0.7)?
        .set_default("inference.top_p", 0.9)?
        .set_default("model.cache_dir", "~/.cache/xandllm")?
        .set_default("device.prefer_gpu", true)?
        .set_default("device.cuda_device_id", 0_i64)?
        // Layer 2: project default.toml
        .add_source(File::with_name("config/default").required(false));

    // Layer 3: optional user-supplied config file
    if let Some(path) = config_file {
        builder = builder.add_source(File::from(path.as_path()).required(true));
    }

    // Layer 4: environment variables (XANDLLM_SERVER_PORT, etc.)
    builder = builder.add_source(
        Environment::with_prefix("XANDLLM")
            .separator("_")
            .try_parsing(true),
    );

    builder.build()?.try_deserialize()
}

/// Expand `~` in cache_dir to the actual home directory.
pub fn expand_cache_dir(raw: &str) -> PathBuf {
    if raw.starts_with('~') {
        if let Some(home) = dirs::home_dir() {
            return home.join(&raw[2..]);
        }
    }
    PathBuf::from(raw)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── load_config defaults ──────────────────────────────────────────────────

    #[test]
    fn test_default_server_port() {
        let cfg = load_config(None).unwrap();
        assert_eq!(cfg.server.port, 11435);
    }

    #[test]
    fn test_default_server_host() {
        let cfg = load_config(None).unwrap();
        assert_eq!(cfg.server.host, "0.0.0.0");
    }

    #[test]
    fn test_default_timeout() {
        let cfg = load_config(None).unwrap();
        assert_eq!(cfg.server.request_timeout_secs, 120);
    }

    #[test]
    fn test_default_inference_values() {
        let cfg = load_config(None).unwrap();
        assert_eq!(cfg.inference.max_sequence_length, 4096);
        assert_eq!(cfg.inference.default_max_new_tokens, 4096);
        assert!((cfg.inference.temperature - 0.7).abs() < f64::EPSILON);
        assert!((cfg.inference.top_p - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_prefer_gpu_true() {
        let cfg = load_config(None).unwrap();
        assert!(cfg.device.prefer_gpu);
        assert_eq!(cfg.device.cuda_device_id, 0);
    }

    #[test]
    fn test_default_cache_dir_contains_xandllm() {
        let cfg = load_config(None).unwrap();
        assert!(
            cfg.model.cache_dir.contains("xandllm"),
            "cache_dir should contain 'xandllm', got: {}",
            cfg.model.cache_dir
        );
    }

    // ── load_config from a custom file ────────────────────────────────────────

    #[test]
    fn test_custom_config_file_overrides_defaults() {
        let dir = std::env::temp_dir().join("xandllm_cfg_test");
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("custom.toml");
        std::fs::write(&file, "[server]\nport = 9999\nhost = \"127.0.0.1\"\nrequest_timeout_secs = 60\n").unwrap();

        let cfg = load_config(Some(&file)).unwrap();
        assert_eq!(cfg.server.port, 9999);
        assert_eq!(cfg.server.host, "127.0.0.1");

        std::fs::remove_dir_all(dir).ok();
    }

    // ── expand_cache_dir ──────────────────────────────────────────────────────

    #[test]
    fn test_expand_absolute_path_unchanged() {
        let path = expand_cache_dir("/absolute/path/to/cache");
        assert_eq!(path, PathBuf::from("/absolute/path/to/cache"));
    }

    #[test]
    fn test_expand_tilde_produces_non_tilde_prefix() {
        let path = expand_cache_dir("~/.cache/xandllm");
        let s = path.to_string_lossy();
        assert!(
            !s.starts_with('~'),
            "Expanded path must not start with '~', got: {s}"
        );
        assert!(
            s.contains("xandllm"),
            "Expanded path must still contain 'xandllm', got: {s}"
        );
    }

    #[test]
    fn test_expand_relative_path_unchanged() {
        let path = expand_cache_dir("relative/path");
        assert_eq!(path, PathBuf::from("relative/path"));
    }

    // ── AppConfig Default impl ────────────────────────────────────────────────

    #[test]
    fn test_app_config_default_matches_load_config() {
        let from_load = load_config(None).unwrap();
        let default = AppConfig::default();
        assert_eq!(from_load.server.port, default.server.port);
        assert_eq!(from_load.inference.max_sequence_length, default.inference.max_sequence_length);
    }
}
