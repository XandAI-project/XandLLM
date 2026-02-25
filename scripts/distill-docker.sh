#!/usr/bin/env bash
# Run XandLLM knowledge distillation inside Docker.
#
# This script works around host CUDA version incompatibilities (e.g. CUDA 13.x
# on RTX 5000 Blackwell cards) by running the build and inference entirely
# inside a container that carries CUDA 12.6.  Only the NVIDIA driver on the
# host needs to be ≥ 12.6 — which any modern driver satisfies.
#
# Prerequisites:
#   - Docker Engine (https://docs.docker.com/engine/install/)
#   - NVIDIA Container Toolkit (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
#       sudo apt install -y nvidia-container-toolkit
#       sudo nvidia-ctk runtime configure --runtime=docker
#       sudo systemctl restart docker
#
# Usage:
#   bash scripts/distill-docker.sh [OPTIONS]
#
# Options:
#   --name      <name>      Human-readable model name; sets output to ./output/<name>
#   --size      <1b|3b|7b>  Fresh student from size preset (default: 3b)
#   --student-base <id>     Fine-tune an existing model instead of --size
#   --model-from <id>       Teacher model (default: Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0)
#   --output    <path>      Host output directory (default: ./output/<name> or ./output/XandLM-Distilled)
#   --dataset   <path>      Host dataset directory (default: ./internal/dataset)
#   --type      safetensor|gguf  (default: safetensor)
#   --epochs    <n>         (default: 3)
#   --batch-size <n>        (default: 2)
#   --learning-rate <f>     (default: 5e-5)
#   --teacher-max-tokens <n> (default: 512)
#   --rebuild               Force rebuild of the Docker image before running
#   --no-gpu                Run on CPU only (very slow — not recommended)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$WORKSPACE_ROOT/docker/docker-compose.yml"

# ── Defaults ───────────────────────────────────────────────────────────────────
MODEL_FROM="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"
MODEL_NAME=""
SIZE="3b"
STUDENT_BASE=""
OUTPUT_HOST=""
DATASET_HOST="$WORKSPACE_ROOT/internal/dataset"
TYPE="safetensor"
EPOCHS=3
BATCH_SIZE=2
LR="5e-5"
MAX_TOKENS=512
REBUILD=false
GPU=true

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)             MODEL_NAME="$2";    shift 2 ;;
        --size)             SIZE="$2";          shift 2 ;;
        --student-base)     STUDENT_BASE="$2";  shift 2 ;;
        --model-from)       MODEL_FROM="$2";    shift 2 ;;
        --output)           OUTPUT_HOST="$2";   shift 2 ;;
        --dataset)          DATASET_HOST="$2";  shift 2 ;;
        --type)             TYPE="$2";          shift 2 ;;
        --epochs)           EPOCHS="$2";        shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";    shift 2 ;;
        --learning-rate)    LR="$2";            shift 2 ;;
        --teacher-max-tokens) MAX_TOKENS="$2";  shift 2 ;;
        --rebuild)          REBUILD=true;       shift   ;;
        --no-gpu)           GPU=false;          shift   ;;
        *) echo "[WARN] Unknown argument: $1";  shift   ;;
    esac
done

# ── Validate student mode ─────────────────────────────────────────────────────
if [[ -n "$SIZE" && -n "$STUDENT_BASE" ]]; then
    echo "[ERROR] --size and --student-base are mutually exclusive."
    exit 1
fi
if [[ -z "$SIZE" && -z "$STUDENT_BASE" ]]; then
    echo "[ERROR] Specify either --size (1b|3b|7b) or --student-base <model-id>."
    exit 1
fi

# ── Resolve paths ─────────────────────────────────────────────────────────────
# If --name given, default output to ./output/<name> on the host
if [[ -n "$MODEL_NAME" && -z "$OUTPUT_HOST" ]]; then
    OUTPUT_HOST="$WORKSPACE_ROOT/output/$MODEL_NAME"
elif [[ -z "$OUTPUT_HOST" ]]; then
    OUTPUT_HOST="$WORKSPACE_ROOT/output/XandLM-Distilled"
fi

# Container-side paths (the compose file mounts ../output → /app/output)
OUTPUT_CONTAINER="/app/output/$(basename "$OUTPUT_HOST")"

mkdir -p "$OUTPUT_HOST"

# ── Check Docker + NVIDIA runtime ─────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[ERROR] Docker not found."
    echo "  Install: https://docs.docker.com/engine/install/"
    exit 1
fi

if ! docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "[WARN] NVIDIA Docker runtime not detected. GPU passthrough may not work."
    echo "  Install NVIDIA Container Toolkit:"
    echo "    sudo apt install -y nvidia-container-toolkit"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker"
    echo "    sudo systemctl restart docker"
fi

# ── Build image ───────────────────────────────────────────────────────────────
cd "$WORKSPACE_ROOT"

if $REBUILD || ! docker image inspect xandllm:latest &>/dev/null; then
    echo "[BUILD] Building xandllm:latest (CUDA 12.6 inside the container) ..."
    echo "        (First build takes 5–15 minutes)"
    docker build -f docker/Dockerfile -t xandllm:latest .
    echo "[BUILD] Image ready."
    echo ""
fi

# ── Print job summary ─────────────────────────────────────────────────────────
echo "======================================================"
echo "  XandLLM Distillation via Docker"
echo "  Image    : xandllm:latest (CUDA 12.6)"
echo "  Teacher  : $MODEL_FROM"
[[ -n "$SIZE" ]]         && echo "  Size     : $SIZE (fresh weights)"
[[ -n "$STUDENT_BASE" ]] && echo "  Base     : $STUDENT_BASE (fine-tune)"
[[ -n "$MODEL_NAME" ]]   && echo "  Name     : $MODEL_NAME"
echo "  Output   : $OUTPUT_HOST"
echo "  Dataset  : $DATASET_HOST"
echo "======================================================"
echo ""

# ── Export env vars for docker compose ───────────────────────────────────────
export DISTILL_MODEL_FROM="$MODEL_FROM"
export DISTILL_MODEL_TO="$OUTPUT_CONTAINER"
export DISTILL_SIZE="$SIZE"
export DISTILL_STUDENT_BASE="$STUDENT_BASE"
export DISTILL_NAME="$MODEL_NAME"
export DISTILL_EPOCHS="$EPOCHS"
export DISTILL_BATCH_SIZE="$BATCH_SIZE"
export DISTILL_LR="$LR"
export DISTILL_MAX_TOKENS="$MAX_TOKENS"
export DISTILL_TYPE="$TYPE"

# Override dataset mount if user specified a custom path
EXTRA_VOLUMES=""
if [[ "$DATASET_HOST" != "$WORKSPACE_ROOT/internal/dataset" ]]; then
    EXTRA_VOLUMES="-v $DATASET_HOST:/app/internal/dataset:ro"
fi

# ── Run distillation ──────────────────────────────────────────────────────────
GPU_FLAGS=""
if $GPU; then
    GPU_FLAGS="--gpus all"
fi

docker run --rm \
    $GPU_FLAGS \
    -e DISTILL_MODEL_FROM \
    -e DISTILL_MODEL_TO \
    -e DISTILL_SIZE \
    -e DISTILL_STUDENT_BASE \
    -e DISTILL_NAME \
    -e DISTILL_EPOCHS \
    -e DISTILL_BATCH_SIZE \
    -e DISTILL_LR \
    -e DISTILL_MAX_TOKENS \
    -e DISTILL_TYPE \
    -e RUST_LOG="${RUST_LOG:-info}" \
    -e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}" \
    -v xandllm_model-cache:/root/.cache/xandllm \
    -v "$WORKSPACE_ROOT/internal/dataset:/app/internal/dataset:ro" \
    -v "$OUTPUT_HOST:/app/output/$(basename "$OUTPUT_HOST")" \
    $EXTRA_VOLUMES \
    --entrypoint /bin/sh \
    xandllm:latest \
    -c '
        set -e
        ARGS="--model-from $DISTILL_MODEL_FROM"
        ARGS="$ARGS --model-to $DISTILL_MODEL_TO"
        ARGS="$ARGS --type $DISTILL_TYPE"
        ARGS="$ARGS --epochs $DISTILL_EPOCHS"
        ARGS="$ARGS --batch-size $DISTILL_BATCH_SIZE"
        ARGS="$ARGS --learning-rate $DISTILL_LR"
        ARGS="$ARGS --teacher-max-tokens $DISTILL_MAX_TOKENS"
        ARGS="$ARGS --gpu"
        [ -n "$DISTILL_SIZE" ]         && ARGS="$ARGS --size $DISTILL_SIZE"
        [ -n "$DISTILL_STUDENT_BASE" ] && ARGS="$ARGS --student-base $DISTILL_STUDENT_BASE"
        [ -n "$DISTILL_NAME" ]         && ARGS="$ARGS --name $DISTILL_NAME"
        echo "==> xandllm distill $ARGS"
        exec xandllm distill $ARGS
    '

echo ""
echo "======================================================"
echo "  Distillation complete!"
echo "  Output: $OUTPUT_HOST"
echo ""
echo "  To serve the model locally:"
echo "    xandllm serve --model $OUTPUT_HOST"
echo ""
echo "  Or add it to docker-compose and serve via Docker:"
echo "    MODEL_ID=$OUTPUT_HOST docker compose -f docker/docker-compose.yml up xandllm"
echo "======================================================"
