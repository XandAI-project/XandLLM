#!/usr/bin/env bash
# Distil Qwen2.5-Coder-7B (teacher) → fresh 3B student from scratch.
#
# The teacher is downloaded automatically if not already cached.
# Uses the built-in ./internal/dataset/ when no --dataset is supplied.
#
# Approximate hardware requirements:
#   GPU: ~10 GB VRAM  (teacher Q4 + 3B student F32)
#   CPU: ~18 GB RAM   (add --no-gpu to force CPU)
#
# Usage:
#   bash scripts/distill-3b.sh [--dataset <path>] [--output <path>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$WORKSPACE_ROOT"

# ── Defaults ───────────────────────────────────────────────────────────────────
TEACHER="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"
OUTPUT="./output/XandLM-Coder-3B"
MODEL_NAME=""
DATASET_ARG=""
GPU_FLAG="--gpu"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET_ARG="--dataset $2"; shift 2 ;;
        --output)   OUTPUT="$2";               shift 2 ;;
        --name)     MODEL_NAME="$2"; OUTPUT="./output/$2"; shift 2 ;;
        --no-gpu)   GPU_FLAG="";               shift   ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

NAME_ARG=${MODEL_NAME:+--name "$MODEL_NAME"}

echo "======================================================"
echo "  XandLLM Distillation — 3B (from scratch)"
echo "  Teacher : $TEACHER"
echo "  Output  : $OUTPUT"
[[ -n "$MODEL_NAME" ]] && echo "  Name    : $MODEL_NAME"
echo "======================================================"
echo ""

xandllm distill \
    --model-from  "$TEACHER" \
    --model-to    "$OUTPUT" \
    --size        3b \
    --type        safetensor \
    --epochs      3 \
    --batch-size  2 \
    --learning-rate 5e-5 \
    --teacher-max-tokens 512 \
    $DATASET_ARG \
    $NAME_ARG \
    $GPU_FLAG

echo ""
echo "Model saved to: $OUTPUT"
echo "Run it with:    xandllm serve --model $OUTPUT"
