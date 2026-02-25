#!/usr/bin/env bash
# Fine-tune Qwen2.5-3B-Instruct using Qwen2.5-Coder-7B as the teacher.
#
# Both models are downloaded automatically if not already cached.
# Uses the built-in ./internal/dataset/ when no --dataset is supplied.
#
# Student base:  Qwen/Qwen2.5-3B-Instruct           (3 B params, safetensors)
# Teacher:       Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0
#
# Using a same-family student (Qwen2.5) gives the best results because the
# vocabulary and tokenizer are identical between teacher and student.
#
# Approximate hardware requirements:
#   GPU: ~12 GB VRAM
#   CPU: ~20 GB RAM  (add --no-gpu to force CPU)
#
# Usage:
#   bash scripts/distill-finetune-3b.sh [--dataset <path>] [--output <path>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$WORKSPACE_ROOT"

# ── Defaults ───────────────────────────────────────────────────────────────────
TEACHER="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"
STUDENT_BASE="Qwen/Qwen2.5-3B-Instruct"
OUTPUT="./output/XandLM-Coder-FineTuned-3B"
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
echo "  XandLLM Distillation — Fine-tune 3B (Qwen2.5-3B)"
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
    --batch-size   2 \
    --learning-rate 3e-5 \
    --teacher-max-tokens 512 \
    $DATASET_ARG \
    $NAME_ARG \
    $GPU_FLAG

echo ""
echo "Model saved to: $OUTPUT"
echo "Run it with:    xandllm serve --model $OUTPUT"
