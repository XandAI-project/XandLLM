use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod commands;
mod config;
mod think_filter;

use config::load_config;

/// XandLLM — high-performance LLM inference engine
#[derive(Debug, Parser)]
#[command(name = "xandllm", version, about, long_about = None, arg_required_else_help = true)]
struct Cli {
    /// Path to a custom configuration file (TOML).
    #[arg(long, global = true, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Log format: "pretty" (default) or "json".
    #[arg(long, global = true, default_value = "pretty", value_name = "FORMAT")]
    log_format: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Start the OpenAI-compatible HTTP server.
    Serve {
        /// Model to load (Hugging Face repo id or cached model id).
        #[arg(long, short = 'm')]
        model: String,

        /// Address to bind to.
        #[arg(long, default_value = "0.0.0.0")]
        host: Option<String>,

        /// Port to listen on.
        #[arg(long, short = 'p')]
        port: Option<u16>,

        /// Prefer GPU acceleration (CUDA/Metal).
        #[arg(long)]
        gpu: bool,
    },

    /// Start an interactive multi-turn chat session (model stays in memory).
    Chat {
        /// Model to load (Hugging Face repo id or cached model id).
        #[arg(long, short = 'm')]
        model: String,

        /// System prompt injected at the start of every session.
        #[arg(long, default_value = "You are a helpful assistant.")]
        system: String,

        /// Maximum new tokens per response.
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Sampling temperature (0.0 to 2.0, default 0.7).
        #[arg(long)]
        temperature: Option<f64>,

        /// Top-p nucleus sampling threshold (0.0 to 1.0, default 0.9).
        #[arg(long)]
        top_p: Option<f64>,

        /// Top-k sampling: limit to top K tokens (disabled by default).
        #[arg(long)]
        top_k: Option<usize>,

        /// Repetition penalty (1.0 = no penalty, >1.0 = discourage repetition).
        #[arg(long)]
        repetition_penalty: Option<f64>,

        /// Frequency penalty (OpenAI-style, default 0.0).
        #[arg(long)]
        frequency_penalty: Option<f64>,

        /// Presence penalty (OpenAI-style, default 0.0).
        #[arg(long)]
        presence_penalty: Option<f64>,

        /// Random seed for reproducible generation.
        #[arg(long)]
        seed: Option<u64>,

        /// Prefer GPU acceleration (CUDA/Metal).
        #[arg(long)]
        gpu: bool,

        /// Show <think>…</think> reasoning blocks instead of hiding them.
        #[arg(long)]
        show_think: bool,

        /// Print timing and throughput stats after each response.
        #[arg(long)]
        stats: bool,
    },

    /// Run local inference and stream output to stdout.
    Run {
        /// Model to use.
        #[arg(long, short = 'm')]
        model: String,

        /// Prompt text.
        #[arg(long)]
        prompt: String,

        /// System prompt injected into the chat template.
        #[arg(long, default_value = "You are a helpful assistant.")]
        system: String,

        /// Skip chat-template formatting and send the raw prompt to the model.
        /// Use this for base/completion models that have no chat template.
        #[arg(long)]
        raw: bool,

        /// Maximum number of new tokens to generate.
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Sampling temperature (0.0 to 2.0, default 0.7).
        #[arg(long)]
        temperature: Option<f64>,

        /// Top-p nucleus sampling threshold (0.0 to 1.0, default 0.9).
        #[arg(long)]
        top_p: Option<f64>,

        /// Top-k sampling: limit to top K tokens (disabled by default).
        #[arg(long)]
        top_k: Option<usize>,

        /// Repetition penalty (1.0 = no penalty, >1.0 = discourage repetition).
        #[arg(long)]
        repetition_penalty: Option<f64>,

        /// Frequency penalty (OpenAI-style, default 0.0).
        #[arg(long)]
        frequency_penalty: Option<f64>,

        /// Presence penalty (OpenAI-style, default 0.0).
        #[arg(long)]
        presence_penalty: Option<f64>,

        /// Random seed for reproducible generation.
        #[arg(long)]
        seed: Option<u64>,

        /// Prefer GPU acceleration (CUDA/Metal).
        #[arg(long)]
        gpu: bool,

        /// Show <think>…</think> reasoning blocks instead of hiding them.
        #[arg(long)]
        show_think: bool,

        /// Print timing and throughput stats after generation.
        #[arg(long)]
        stats: bool,
    },

    /// Download a model from Hugging Face Hub.
    Pull {
        /// Hugging Face model id (e.g. `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`).
        model_id: String,

        /// Git revision, branch, or commit hash.
        #[arg(long, default_value = "main")]
        revision: String,

        /// Explicit GGUF filename to download (e.g. `Qwen2.5-Coder-7B-Instruct-Q4_0.gguf`).
        /// When omitted, the correct file is discovered automatically from the repo listing.
        /// You can also embed the quantization tag directly in the model id: `repo:Q4_0`.
        #[arg(long)]
        gguf_file: Option<String>,
    },

    /// List models stored in the local cache.
    List,

    /// Delete a cached model from local storage.
    Delete {
        /// Model id to delete (e.g. `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`).
        model_id: String,

        /// Revision to delete. Omit to delete all revisions.
        #[arg(long, short = 'r')]
        revision: Option<String>,
    },

    /// Distil a large teacher model into a smaller student model.
    ///
    /// Exactly one of --size or --student-base must be provided.
    ///
    /// MEMORY: training a 1B student (F32) alongside a 7B Q4 teacher requires
    /// roughly 8 GB of VRAM on GPU, or 12–16 GB of RAM on CPU.
    Distill {
        /// Teacher model (HF repo id or cached model id).
        ///
        /// Example: Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
        #[arg(long)]
        model_from: String,

        /// Path to directory containing the training dataset (JSONL files
        /// with {"prompt":"…","completion":"…"} records).
        /// Defaults to ./internal/dataset/ when omitted.
        #[arg(long)]
        dataset: Option<PathBuf>,

        /// Output directory for the distilled model weights and config.
        #[arg(long)]
        model_to: PathBuf,

        /// Output format: "safetensor" (default) or "gguf".
        /// GGUF conversion requires llama.cpp on PATH.
        #[arg(long, default_value = "safetensor")]
        r#type: String,

        /// Student architecture preset for fresh (random) initialisation.
        /// Valid values: 1b (~1.1B params), 3b (~3B params), 7b (~7B params).
        /// Mutually exclusive with --student-base.
        #[arg(long)]
        size: Option<String>,

        /// Existing smaller model to fine-tune (HF repo id or cached model id).
        /// The model is loaded from the local cache as the student's starting
        /// point.  Mutually exclusive with --size.
        #[arg(long)]
        student_base: Option<String>,

        /// Number of training epochs (default 3).
        #[arg(long, default_value_t = 3)]
        epochs: usize,

        /// Batch size for training (default 4).
        #[arg(long, default_value_t = 4)]
        batch_size: usize,

        /// AdamW learning rate (default 0.0001).
        #[arg(long, default_value_t = 1e-4)]
        learning_rate: f64,

        /// Maximum token-sequence length for prompt + completion (default from config).
        #[arg(long)]
        max_seq_len: Option<usize>,

        /// Maximum new tokens the teacher may generate per prompt (default 512).
        #[arg(long)]
        teacher_max_tokens: Option<usize>,

        /// Use GPU acceleration for both teacher inference and student training.
        #[arg(long)]
        gpu: bool,

        /// Human-readable name embedded in the exported model's config.json.
        /// Also sets the output directory to ./output/<name> when --model-to is omitted.
        /// Example: --name "MyCoder-3B"
        #[arg(long)]
        name: Option<String>,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialise tracing
    init_tracing(&cli.log_format);

    // Load configuration
    let cfg = load_config(cli.config.as_ref())
        .context("Failed to load configuration")?;

    match cli.command {
        Commands::Chat { model, system, max_tokens, temperature, top_p, top_k, repetition_penalty, frequency_penalty, presence_penalty, seed, gpu, show_think, stats } => {
            commands::chat::run(&model, &system, gpu, max_tokens, temperature, top_p, top_k, repetition_penalty, frequency_penalty, presence_penalty, seed, show_think, stats, &cfg).await?;
        }
        Commands::Serve { model, host, port, gpu } => {
            commands::serve::run(&model, host.as_deref(), port, gpu, &cfg).await?;
        }
        Commands::Run { model, prompt, system, raw, max_tokens, temperature, top_p, top_k, repetition_penalty, frequency_penalty, presence_penalty, seed, gpu, show_think, stats } => {
            commands::run::run(&model, &prompt, &system, raw, max_tokens, temperature, top_p, top_k, repetition_penalty, frequency_penalty, presence_penalty, seed, gpu, show_think, stats, &cfg).await?;
        }
        Commands::Pull { model_id, revision, gguf_file } => {
            commands::pull::run(&model_id, &revision, gguf_file.as_deref(), &cfg).await?;
        }
        Commands::List => {
            commands::list::run(&cfg)?;
        }
        Commands::Delete { model_id, revision } => {
            commands::delete::run(&model_id, revision.as_deref(), &cfg)?;
        }
        Commands::Distill {
            model_from,
            dataset,
            model_to,
            r#type: output_type,
            size,
            student_base,
            epochs,
            batch_size,
            learning_rate,
            max_seq_len,
            teacher_max_tokens,
            gpu,
            name,
        } => {
            // --name sets the output dir to ./output/<name> when --model-to is not given
            let resolved_model_to = name.as_deref()
                .map(|n| PathBuf::from("output").join(n))
                .unwrap_or(model_to);

            commands::distill::run(
                &model_from,
                dataset.as_ref(),
                &resolved_model_to,
                &output_type,
                size.as_deref(),
                student_base.as_deref(),
                epochs,
                batch_size,
                learning_rate,
                max_seq_len,
                teacher_max_tokens,
                gpu,
                name.as_deref(),
                &cfg,
            )
            .await?;
        }
    }

    Ok(())
}

fn init_tracing(log_format: &str) {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let registry = tracing_subscriber::registry().with(env_filter);

    if log_format == "json" {
        registry
            .with(fmt::layer().json())
            .init();
    } else {
        registry
            .with(fmt::layer().pretty())
            .init();
    }
}
