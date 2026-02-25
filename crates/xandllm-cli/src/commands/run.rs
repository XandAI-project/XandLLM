use std::io::Write;
use std::time::Instant;

use tracing::info;

use xandllm_core::{chat_template, select_device, AnyModel, GenerateInput, Model, ModelConfig, SamplingParams};
use xandllm_hub::ModelCache;

use crate::config::{expand_cache_dir, AppConfig};
use crate::think_filter::ThinkFilter;

/// Run local inference and stream tokens to stdout.
///
/// Automatically selects GGUF (quantized) or safetensors loading based on
/// which weight files are present in the model cache directory.
pub async fn run(
    model_id: &str,
    prompt: &str,
    system: &str,
    raw: bool,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    repetition_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    seed: Option<u64>,
    prefer_gpu: bool,
    show_think: bool,
    stats: bool,
    config: &AppConfig,
) -> anyhow::Result<()> {
    let cache_dir = expand_cache_dir(&config.model.cache_dir);
    let cache = ModelCache::new(&cache_dir)?;

    let revision = "main";
    let model_dir = cache.model_dir(model_id, revision);

    let device = select_device(prefer_gpu || config.device.prefer_gpu, config.device.cuda_device_id)?;

    let model_config = ModelConfig {
        repo_id: model_id.to_string(),
        revision: revision.to_string(),
        model_dir: model_dir.clone(),
        max_sequence_length: config.inference.max_sequence_length,
    };

    let load_start = Instant::now();
    info!(model_dir = %model_dir.display(), "Loading model for local inference");
    let mut model = AnyModel::load(&model_config, &device)?;
    let load_elapsed = load_start.elapsed();

    // Reuse the tokenizer already built during model loading (may be from GGUF
    // metadata when no tokenizer.json is present in the cache directory).
    let tokenizer = model.tokenizer_arc();

    // Apply the chat template appropriate for this model unless the caller
    // explicitly opted out with --raw.  Use chat_format() rather than
    // architecture() so models that report a different arch than their actual
    // template (e.g. Nanbeige: arch="llama", template=ChatML) are handled
    // correctly.
    let chat_fmt = model.chat_format().to_string();
    let formatted = if raw {
        info!("--raw flag set: sending prompt without chat template");
        prompt.to_string()
    } else {
        let f = chat_template::format_prompt(&chat_fmt, prompt, system);
        info!(chat_format = %chat_fmt, "Applied chat template");
        f
    };

    // add_special_tokens=false: the chat template already embeds the special
    // tokens (<|im_start|>, etc.) as literal text.
    let token_ids = tokenizer.encode(&formatted, false)?;
    let input = GenerateInput { token_ids };

    // Collect stop-token IDs for this model's specific chat format.
    // Using format-specific tokens prevents models like Nanbeige (LLaMA-derived
    // ChatML) from having </s> registered as a stop token, which would
    // truncate generation inside <think> blocks.
    let mut stop_token_ids: Vec<u32> = chat_template::stop_token_strings_for_format(&chat_fmt)
        .iter()
        .filter_map(|s| tokenizer.token_id(s))
        .collect();

    // For non-ChatML formats (and --raw mode), include the native EOS token as
    // a safety net so base/llama2/llama3 models still terminate naturally.
    // ChatML models deliberately exclude </s> because it appears mid-stream.
    if raw || !matches!(chat_fmt.as_str(), "chatml" | "qwen2") {
        if let Some(eos) = tokenizer.eos_token_id() {
            if !stop_token_ids.contains(&eos) {
                stop_token_ids.push(eos);
            }
        }
    }

    let params = SamplingParams {
        max_new_tokens: max_tokens.unwrap_or(config.inference.default_max_new_tokens),
        temperature: temperature.unwrap_or(config.inference.temperature),
        top_p: top_p.unwrap_or(config.inference.top_p),
        top_k,
        repetition_penalty: repetition_penalty.unwrap_or(1.0),
        frequency_penalty: frequency_penalty.unwrap_or(0.0),
        presence_penalty: presence_penalty.unwrap_or(0.0),
        seed,
        greedy: false,
        stop_token_ids,
    };

    info!("Generating response");
    let mut rx = model.generate(input, params)?;

    let mut filter = if show_think { None } else { Some(ThinkFilter::new()) };
    let mut token_count: usize = 0;
    let gen_start = Instant::now();

    while let Some(result) = rx.recv().await {
        let token = result?;
        if token.is_eos {
            break;
        }
        token_count += 1;
        let text = match &mut filter {
            Some(f) => f.push(&token.text),
            None => Some(token.text.clone()),
        };
        if let Some(t) = text {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }

    // Flush any text held back by the filter — fires whether generation ended
    // on an EOS token or because the token budget was exhausted.
    if let Some(f) = &mut filter {
        if let Some(t) = f.flush() {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    println!();

    if stats {
        let gen_elapsed = gen_start.elapsed();
        let tok_per_sec = if gen_elapsed.as_secs_f64() > 0.0 {
            token_count as f64 / gen_elapsed.as_secs_f64()
        } else {
            0.0
        };
        eprintln!(
            "\n[stats] model loaded in {:.2} s | {} tokens generated | {:.1} tok/s",
            load_elapsed.as_secs_f64(),
            token_count,
            tok_per_sec,
        );
    }

    Ok(())
}
