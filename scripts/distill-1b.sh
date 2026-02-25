#!/usr/bin/env bash
# Distil Qwen2.5-Coder-7B (teacher) → fresh 1B student from scratch.
#
# The teacher is downloaded automatically if not already cached.
# Uses the built-in ./internal/dataset/ when no --dataset is supplied.
#
# Approximate hardware requirements:
#   GPU: ~6 GB VRAM  (teacher Q4 + 1B student F32)
#   CPU: ~10 GB RAM  (add --no-gpu to force CPU)
#
# Usage:
#   bash scripts/distill-1b.sh [--dataset <path>] [--output <path>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$WORKSPACE_ROOT"

# ── Defaults ───────────────────────────────────────────────────────────────────
TEACHER="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"
OUTPUT="./output/XandLM-Coder-1B"
MODEL_NAME=""
DATASET_ARG=""
GPU_FLAG="--gpu"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)  DATASET_ARG="--dataset $2"; shift 2 ;;
        --output)   OUTPUT="$2";               shift 2 ;;
        --name)     MODEL_NAME="$2"; OUTPUT="./output/$2"; shift 2 ;;
        --gpu)      GPU_FLAG="--gpu";    shift   ;;

        --no-gpu)   GPU_FLAG="";               shift   ;;
        *) echo "[WARN] Unknown argument: $1"; shift ;;
    esac
done

NAME_ARG=${MODEL_NAME:+--name "$MODEL_NAME"}

echo "======================================================"
echo "  XandLLM Distillation — 1B (from scratch)"
echo "  Teacher : $TEACHER"
echo "  Output  : $OUTPUT"
[[ -n "$MODEL_NAME" ]] && echo "  Name    : $MODEL_NAME"
echo "======================================================"
echo ""

xandllm distill \
    --model-from  "$TEACHER" \
    --model-to    "$OUTPUT" \
    --size        1b \
    --type        safetensor \
    --epochs      3 \
    --batch-size  4 \
    --learning-rate 1e-4 \
    --teacher-max-tokens 512 \
    $DATASET_ARG \
    $NAME_ARG \
    $GPU_FLAG

echo ""
echo "Model saved to: $OUTPUT"
echo "Run it with:    xandllm serve --model $OUTPUT"
