# XandLLM

A production-grade, high-performance LLM inference engine written in Rust. Functionally comparable to vLLM / llama.cpp with an OpenAI-compatible HTTP API, CLI tooling, Docker Compose deployment, and built-in **knowledge distillation**.

## Features

- **GPU acceleration** via CUDA (automatic CPU fallback)
- **OpenAI-compatible API** — drop-in replacement for `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- **Streaming** via Server-Sent Events (`text/event-stream`)
- **Hugging Face Hub** model downloading and caching (with auto-download on demand)
- **CLI** for local inference, interactive chat, and model management
- **Knowledge distillation** — compress a large teacher model into a smaller student
- **Docker Compose** with NVIDIA GPU passthrough

---

## Supported LLM Architectures

XandLLM currently supports the following model architectures and formats:

### Architectures

| Architecture | Format | Models | Chat Template |
|---|---|---|---|
| **LLaMA** | Safetensors / GGUF | LLaMA, LLaMA-2, LLaMA-3, Mistral, CodeLlama | `llama2`, `llama3` |
| **Qwen2** | GGUF | Qwen2, Qwen2.5, Qwen2.5-Coder | `chatml` |
| **ChatML-compatible** | GGUF | Nanbeige, DeepSeek, and other ChatML models | `chatml` |

### Model Formats

- **GGUF** (quantized) — Q4_0, Q8_0, and other quantization levels
- **Safetensors** (full precision) — LLaMA-family models

### Chat Templates

- **ChatML** (`chatml`) — Qwen2, Nanbeige, DeepSeek, and other instruction-tuned models
- **LLaMA-2** (`llama2`) — `[INST]` format for LLaMA-2 and compatible models
- **LLaMA-3** (`llama3`) — Header-based format for LLaMA-3 instruct models

Chat template detection is automatic based on vocabulary probing, ensuring correct formatting even when the architecture tag doesn't match the template (e.g., Nanbeige reports `"llama"` but uses ChatML).

### Example Models

```bash
# Qwen2 models
xandllm pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0

# LLaMA-3 models
xandllm pull meta-llama/Llama-3.1-8B-Instruct

# ChatML-compatible models
xandllm pull tantk/Nanbeige4.1-3B-GGUF
```

---

## Quickstart

### Prerequisites

- Rust 1.76+ (`rustup install stable`)
- (Optional) CUDA 12.x toolkit for GPU acceleration

### Build

**Linux:**

```bash
# CPU-only
bash scripts/build-linux.sh

# With CUDA GPU support (requires CUDA 12.x toolkit)
bash scripts/build-linux-cuda.sh
```

**Windows:**

```bat
scripts\build-cuda.bat
```

**Manual (any platform):**

```bash
# CPU-only
cargo install --path crates/xandllm-cli --locked

# With CUDA
cargo install --path crates/xandllm-cli --features cuda --locked
```

### Pull a model

```bash
export HUGGING_FACE_HUB_TOKEN=hf_...   # only needed for gated models
xandllm pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0
```

### Run local inference

```bash
# Streaming (default)
xandllm run \
  --model Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  --prompt "Explain quantum entanglement in one paragraph."

# With performance stats
xandllm run \
  --model Qwen/Qwen2.5-Coder-7B-Instruct-GGUF \
  --prompt "Write a haiku about Rust." \
  --stats
```

### Interactive chat

```bash
xandllm chat --model Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --gpu
```

### Start the API server

```bash
xandllm serve --model Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --port 11435 --gpu
```

### List / delete cached models

```bash
xandllm list
xandllm delete Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
```

---

## Knowledge Distillation

Compress a large teacher model into a smaller, faster student model using your own dataset.

Models that are not already in the local cache are **downloaded automatically** — no prior `xandllm pull` step is required.

### Dataset format

Each line of every `.jsonl` file in the dataset directory must be:

```json
{"prompt": "Explain recursion.", "completion": "Recursion is a technique where a function calls itself..."}
```

### Fresh student (random weights from a size preset)

```bash
xandllm distill \
  --model-from  Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0 \
  --dataset     ./my_dataset \
  --model-to    ./output/XandLM-1B \
  --size        1b \
  --gpu
```

| Flag | Values | Description |
|---|---|---|
| `--size` | `1b`, `3b`, `7b` | Student architecture preset |
| `--epochs` | integer (default `3`) | Training passes over the dataset |
| `--batch-size` | integer (default `4`) | Sequences per gradient step |
| `--learning-rate` | float (default `1e-4`) | AdamW learning rate |
| `--teacher-max-tokens` | integer (default `512`) | Max tokens the teacher generates per sample |

### Fine-tune an existing smaller model

```bash
xandllm distill \
  --model-from    Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0 \
  --dataset       ./my_dataset \
  --model-to      ./output/MyFineTuned \
  --student-base  Qwen/Qwen2.5-1.5B \
  --epochs        5 \
  --batch-size    2 \
  --learning-rate 5e-5 \
  --gpu
```

### Output formats

| `--type` | Description |
|---|---|
| `safetensor` (default) | HuggingFace SafeTensors — load directly with `xandllm serve` |
| `gguf` | Calls `convert_hf_to_gguf.py` + `llama-quantize` from `llama.cpp` (must be on PATH) |

After distillation, serve the result:

```bash
xandllm serve --model ./output/XandLM-1B
```

### Distillation scripts

Convenience scripts are provided in `scripts/`. Each one auto-downloads the required models and defaults to `./internal/dataset/` when no `--dataset` flag is passed.

| Script | Mode | Teacher | Student |
|---|---|---|---|
| `scripts/distill-1b.sh` | From scratch | Qwen2.5-Coder-7B Q4 | 1B random init |
| `scripts/distill-3b.sh` | From scratch | Qwen2.5-Coder-7B Q4 | 3B random init |
| `scripts/distill-7b.sh` | From scratch | Llama-3.1-8B Q4_K_M | 7B random init |
| `scripts/distill-finetune-1b.sh` | Fine-tune | Qwen2.5-Coder-7B Q4 | TinyLlama-1.1B-Chat |
| `scripts/distill-finetune-3b.sh` | Fine-tune | Qwen2.5-Coder-7B Q4 | Qwen2.5-3B-Instruct |
| `scripts/distill-finetune-7b.sh` | Fine-tune | Llama-3.1-8B Q4_K_M | Mistral-7B-Instruct-v0.3 |

All scripts accept `--dataset <path>`, `--output <path>`, and `--no-gpu` overrides:

```bash
bash scripts/distill-1b.sh --dataset ./my_data --output ./my-1b-model
bash scripts/distill-finetune-3b.sh --no-gpu
```

---

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request**

```json
{
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user",   "content": "Hello, what can you do?" }
  ],
  "stream": false,
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

**Response (non-streaming)**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1708000000,
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "..." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 12, "completion_tokens": 48, "total_tokens": 60 }
}
```

**Streaming** — set `"stream": true` to receive `text/event-stream` SSE chunks:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},...}]}

data: [DONE]
```

---

### `POST /v1/completions`

Raw text completion endpoint.

**Request**

```json
{
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
  "prompt": "The capital of France is",
  "max_tokens": 10,
  "stream": false
}
```

---

### `GET /v1/models`

Returns a list of the currently loaded model(s).

---

### `GET /health`

Liveness probe — returns `{ "status": "ok" }`.

---

## Configuration

Configuration is loaded from (in order of precedence):

1. Built-in defaults
2. `config/default.toml`
3. A custom file passed via `--config <path>`
4. Environment variables prefixed with `XANDLLM_`

### `config/default.toml`

```toml
[server]
host = "0.0.0.0"
port = 11435
request_timeout_secs = 120

[inference]
max_batch_size = 8
max_sequence_length = 4096
default_max_new_tokens = 512
temperature = 0.7
top_p = 0.9

[model]
cache_dir = "~/.cache/xandllm"

[device]
prefer_gpu = true
cuda_device_id = 0
```

### Environment variable overrides

| Variable | Description |
|---|---|
| `RUST_LOG` | Log level (`trace`, `debug`, `info`, `warn`, `error`) |
| `HUGGING_FACE_HUB_TOKEN` | Hugging Face Hub token (for gated / private models) |
| `XANDLLM_SERVER_PORT` | Override server port |
| `XANDLLM_DEVICE_PREFER_GPU` | `true`/`false` |
| `XANDLLM_MODEL_CACHE_DIR` | Override model cache directory |

---

## Docker Compose

```bash
# Copy and fill in your HF token
cp .env .env.local
echo "HUGGING_FACE_HUB_TOKEN=hf_..." >> .env.local

# Build and start (with GPU passthrough)
docker compose --env-file .env.local -f docker/docker-compose.yml up --build

# Test
curl http://localhost:11435/health
```

The service uses `nvidia/cuda:12.3.0-runtime-ubuntu22.04` at runtime and a
`nvidia/cuda:12.3.0-devel-ubuntu22.04` builder with `cargo-chef` for fast
incremental rebuilds.

---

## CLI Reference

```
xandllm serve    --model <id>   [--host <h>] [--port <p>] [--gpu]
xandllm run      --model <id>   --prompt <text> [--max-tokens <n>] [--stats] [--gpu]
xandllm chat     --model <id>   [--system <text>] [--stats] [--gpu]
xandllm pull     <model-id>     [--revision <rev>]
xandllm list
xandllm delete   <model-id>

xandllm distill
  --model-from   <id>           Teacher model (auto-downloaded if not cached)
  --dataset      <dir>          JSONL dataset directory
  --model-to     <path>         Output directory for distilled model
  --type         safetensor|gguf
  --size         1b|3b|7b       Fresh student (mutually exclusive with --student-base)
  --student-base <id>           Existing model to fine-tune (auto-downloaded if not cached)
  [--epochs <n>]  [--batch-size <n>]  [--learning-rate <f>]
  [--teacher-max-tokens <n>]    [--max-seq-len <n>]
  [--gpu]
```

---

## Project Structure

```
xandllm/
├── Cargo.toml                   # workspace manifest
├── config/default.toml          # default runtime configuration
├── crates/
│   ├── xandllm-core/            # model loading, inference, tokenization, KV-cache
│   ├── xandllm-api/             # axum HTTP server, OpenAI-compatible routes
│   ├── xandllm-cli/             # clap CLI (serve, run, chat, pull, list, delete, distill)
│   ├── xandllm-hub/             # HF model downloading & caching
│   └── xandllm-distill/         # knowledge distillation (dataset, student, teacher, export)
├── frontend/                    # React streaming chat UI
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md
```

---

## Roadmap

### Mixture of Experts (MoE) Models

- [ ] **Mixtral** (Mistral AI) — MoE architecture with 8 experts
- [ ] **Qwen-MoE** — Qwen's MoE variant
- [ ] **DeepSeek-MoE** — DeepSeek's MoE architecture
- [ ] Generic MoE routing / expert management

### Additional Architectures

- [ ] **Phi** (Microsoft) — compact language models
- [ ] **Gemma** (Google) — open Gemma models
- [ ] **Falcon** — TII Falcon models
- [ ] **Mamba** — state-space models (SSM)
- [ ] **RWKV** — recurrent neural network architecture
- [ ] **GPT-2** / **GPT-NeoX** — compatibility with older models

### Multi-Modal Support

- [ ] Vision-Language Models (LLaVA, Qwen-VL)
- [ ] Audio models (Whisper, AudioLM)
- [ ] Multi-modal tokenization (image tokens, audio tokens)

### Advanced Features

- [ ] **Continuous batching** — dynamic batching for improved throughput
- [ ] **PagedAttention** — memory-efficient attention mechanism
- [ ] **Tensor parallelism** — multi-GPU inference
- [ ] **Pipeline parallelism** — model sharding across GPUs
- [ ] **Quantization formats** — AWQ, GPTQ, EXL2
- [x] **Knowledge distillation** — teacher-to-student compression with custom datasets
- [ ] **LoRA / QLoRA adapters** — low-rank adaptation without full retraining
- [ ] **Speculative decoding** — faster generation with draft models

### Performance Optimizations

- [ ] **Flash Attention** — memory-efficient attention implementation
- [ ] **Kernel fusion** — fused CUDA kernels for common operations
- [ ] **INT4 quantization** — lower precision inference
- [ ] **Sparse attention** — attention pattern optimization

### API Enhancements

- [ ] **Function calling** — tool/function use support
- [ ] **JSON mode** — structured output generation
- [ ] **Logprobs** — token-level log probabilities
- [ ] **Multiple choices** — generate N completions per request

---

## Versioning

This project uses semantic versioning: `major.minor.commits`

Current version: **0.1.0**
