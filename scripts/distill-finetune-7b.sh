#!/usr/bin/env bash
# Fine-tune Mistral-7B-Instruct using Llama-3.1-8B as the teacher.
#
# Both models are downloaded automatically if not already cached.
# Uses the built-in ./internal/dataset/ when no --dataset is supplied.
#
# Student base:  mistralai/Mistral-7B-Instruct-v0.3      (7 B params, safetensors)
# Teacher:       bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M
#
# Approximate hardware requirements:
#   GPU: ~24 GB VRAM  (recommend A100/H100 or RTX 4090)
#   CPU: ~40 GB RAM   (add --no-gpu to force CPU — very slow)
#
# Usage:
#   bash scripts/distill-finetune-7b.sh [--dataset <path>] [--output <path>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$WORKSPACE_ROOT"

# ── Defaults ───────────────────────────────────────────────────────────────────
TEACHER="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
STUDENT_BASE="mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT="./output/XandLM-FineTuned-7B"
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
echo "  XandLLM Distillation — Fine-tune 7B (Mistral-7B)"
echo "  Teacher      : $TEACHER"
echo "  Student base : $STUDENT_BASE"
echo "  Output       : $OUTPUT"
[[ -n "$MODEL_NAME" ]] && echo "  Name         : $MODEL_NAME"
echo "======================================================"
echo ""

xandllm distill \
    --model-from   "$TEACHER" \
    --student-base "$STUDENT_BASE" \
    --model-to     "$OUTPUT" \
    --type         safetensor \
    --epochs       3 \
    --batch-size   1 \
    --learning-rate 2e-5 \
    --teacher-max-tokens 512 \
    $DATASET_ARG \
    $NAME_ARG \
    $GPU_FLAG

echo ""
echo "Model saved to: $OUTPUT"
echo "Run it with:    xandllm serve --model $OUTPUT"
