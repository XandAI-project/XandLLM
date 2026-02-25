#!/usr/bin/env bash
# Distil Llama-3.1-8B (teacher) → fresh 7B student from scratch.
#
# The teacher is downloaded automatically if not already cached.
# Uses the built-in ./internal/dataset/ when no --dataset is supplied.
#
# Approximate hardware requirements:
#   GPU: ~20 GB VRAM  (teacher Q4 + 7B student F32) — recommend A100/H100 or 2× consumer GPU
#   CPU: ~32 GB RAM   (add --no-gpu to force CPU — very slow)
#
# Usage:
#   bash scripts/distill-7b.sh [--dataset <path>] [--output <path>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$WORKSPACE_ROOT"

# ── Defaults ───────────────────────────────────────────────────────────────────
TEACHER="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
OUTPUT="./output/XandLM-7B"
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
echo "  XandLLM Distillation — 7B (from scratch)"
echo "  Teacher : $TEACHER"
echo "  Output  : $OUTPUT"
[[ -n "$MODEL_NAME" ]] && echo "  Name    : $MODEL_NAME"
echo "======================================================"
echo ""

xandllm distill \
    --model-from  "$TEACHER" \
    --model-to    "$OUTPUT" \
    --size        7b \
    --type        safetensor \
    --epochs      2 \
    --batch-size  1 \
    --learning-rate 2e-5 \
    --teacher-max-tokens 512 \
    $DATASET_ARG \
    $NAME_ARG \
    $GPU_FLAG

echo ""
echo "Model saved to: $OUTPUT"
echo "Run it with:    xandllm serve --model $OUTPUT"
