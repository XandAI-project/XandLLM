use std::io::{self, Write};
use std::time::Instant;

use tracing::info;

use xandllm_core::{chat_template, select_device, AnyModel, GenerateInput, Model, ModelConfig, SamplingParams};
use xandllm_hub::ModelCache;

use crate::config::{expand_cache_dir, AppConfig};
use crate::think_filter::ThinkFilter;

/// Start an interactive multi-turn chat session.
///
/// The model is loaded once and kept in memory for the entire session.
/// Conversation history is accumulated across turns so the model maintains
/// context.  Type `/quit` or press Ctrl+C / Ctrl+D to exit.
pub async fn run(
    model_id: &str,
    system: &str,
    prefer_gpu: bool,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    repetition_penalty: Option<f64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    seed: Option<u64>,
    show_think: bool,
    stats: bool,
    config: &AppConfig,
) -> anyhow::Result<()> {
    let cache_dir = expand_cache_dir(&config.model.cache_dir);
    let cache = ModelCache::new(&cache_dir)?;

    let revision = "main";
    let model_dir = cache.model_dir(model_id, revision);

    let device =
        select_device(prefer_gpu || config.device.prefer_gpu, config.device.cuda_device_id)?;

    let model_config = ModelConfig {
        repo_id: model_id.to_string(),
        revision: revision.to_string(),
        model_dir: model_dir.clone(),
        max_sequence_length: config.inference.max_sequence_length,
    };

    let load_start = Instant::now();
    info!(model_dir = %model_dir.display(), "Loading model for chat session");
    let mut model = AnyModel::load(&model_config, &device)?;
    let load_elapsed = load_start.elapsed();

    let tokenizer = model.tokenizer_arc();
    let chat_fmt = model.chat_format().to_string();

    let mut stop_token_ids: Vec<u32> = chat_template::stop_token_strings_for_format(&chat_fmt)
        .iter()
        .filter_map(|s| tokenizer.token_id(s))
        .collect();

    // For non-ChatML formats, include the native EOS token so base/llama2/llama3
    // models still terminate naturally. ChatML models deliberately exclude </s>
    // because it appears mid-stream inside <think> blocks.
    if !matches!(chat_fmt.as_str(), "chatml" | "qwen2" | "qwen3" | "chatml-thinking") {
        if let Some(eos) = tokenizer.eos_token_id() {
            if !stop_token_ids.contains(&eos) {
                stop_token_ids.push(eos);
            }
        }
    }

    let mut history: Vec<(String, String)> = Vec::new();

    if stats {
        eprintln!("[stats] model loaded in {:.2} s", load_elapsed.as_secs_f64());
    }

    println!("\nXandLLM Chat  |  model: {model_id}  |  format: {chat_fmt}  |  type /quit to exit\n");

    loop {
        // ── Read user input ───────────────────────────────────────────────
        print!("You: ");
        io::stdout().flush()?;

        let mut line = String::new();
        match io::stdin().read_line(&mut line) {
            Ok(0) => {
                // EOF — Ctrl+D on Unix, but also fires on Windows pipe close
                println!("\nGoodbye!");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("Read error: {e}");
                break;
            }
        }

        let user_input = line.trim().to_string();

        if user_input.is_empty() {
            continue;
        }
        if user_input == "/quit" || user_input == "/exit" {
            println!("Goodbye!");
            break;
        }
        if user_input == "/clear" {
            history.clear();
            println!("[Conversation history cleared]\n");
            continue;
        }

        // ── Build prompt with full history ────────────────────────────────
        let prompt =
            chat_template::build_chat_prompt(&chat_fmt, system, &history, &user_input);

        let token_ids = tokenizer
            .encode(&prompt, false)
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {e}"))?;

        let context_len = token_ids.len();
        let max_ctx = config.inference.max_sequence_length;

        if context_len >= max_ctx {
            eprintln!(
                "[Warning] Context length ({context_len}) has reached the model limit \
                 ({max_ctx}). Consider starting a new session with /clear."
            );
        }

        let input = GenerateInput { token_ids };

        let stop_strings: Vec<String> = xandllm_core::chat_template::stop_text_strings_for_format(&chat_fmt)
            .iter()
            .map(|s| s.to_string())
            .collect();

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
            stop_token_ids: stop_token_ids.clone(),
            repeat_last_n: Some(64),
            stop_strings,
            thinking_mode: chat_fmt == "chatml-thinking",
        };

        // ── Stream the response ───────────────────────────────────────────
        let mut rx = model.generate(input, params)?;

        print!("\nAssistant: ");
        io::stdout().flush()?;

        let mut response = String::new();
        let mut filter = if show_think { None } else { Some(ThinkFilter::new()) };
        let mut token_count: usize = 0;
        let gen_start = Instant::now();

        while let Some(result) = rx.recv().await {
            let token = result?;
            if token.is_eos {
                break;
            }
            token_count += 1;
            // Accumulate the raw text for conversation history.
            response.push_str(&token.text);
            // Pass through the think-filter before printing.
            let visible = match &mut filter {
                Some(f) => f.push(&token.text),
                None => Some(token.text.clone()),
            };
            if let Some(t) = visible {
                print!("{t}");
                io::stdout().flush()?;
            }
        }

        // Flush any text held back by the filter — fires whether generation
        // ended on an EOS token or because the token budget was exhausted.
        if let Some(f) = &mut filter {
            if let Some(t) = f.flush() {
                print!("{t}");
                io::stdout().flush()?;
            }
        }

        println!("\n");

        if stats {
            let gen_elapsed = gen_start.elapsed();
            let tok_per_sec = if gen_elapsed.as_secs_f64() > 0.0 {
                token_count as f64 / gen_elapsed.as_secs_f64()
            } else {
                0.0
            };
            eprintln!(
                "[stats] {} tokens generated | {:.1} tok/s",
                token_count,
                tok_per_sec,
            );
        }

        history.push((user_input, response));
    }

    Ok(())
}
