#!/usr/bin/env bash
# Fine-tune TinyLlama 1.1B using Qwen2.5-Coder-7B as the teacher.
#
# Unlike the "from scratch" scripts, this loads TinyLlama's pre-trained weights
# as the student starting point, which converges much faster than random init.
#
# Both models are downloaded automatically if not already cached.
# Uses the built-in ./internal/dataset/ when no --dataset is supplied.
#
# Student base:  TinyLlama/TinyLlama-1.1B-Chat-v1.0  (1.1 B params, safetensors)
# Teacher:       Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0
#
# Approximate hardware requirements:
#   GPU: ~8 GB VRAM
#   CPU: ~12 GB RAM  (add --no-gpu to force CPU)
#
# Usage:
#   bash scripts/distill-finetune-1b.sh [--dataset <path>] [--output <path>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$WORKSPACE_ROOT"

# ── Defaults ───────────────────────────────────────────────────────────────────
TEACHER="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"
STUDENT_BASE="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT="./output/XandLM-FineTuned-1B"
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
echo "  XandLLM Distillation — Fine-tune 1.1B (TinyLlama)"
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
    --epochs       5 \
    --batch-size   4 \
    --learning-rate 5e-5 \
    --teacher-max-tokens 512 \
    $DATASET_ARG \
    $NAME_ARG \
    $GPU_FLAG

echo ""
echo "Model saved to: $OUTPUT"
echo "Run it with:    xandllm serve --model $OUTPUT"
