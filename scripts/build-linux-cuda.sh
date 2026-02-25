#!/usr/bin/env bash
# Build XandLLM CLI with CUDA GPU support and install it to ~/.cargo/bin
#
# Requirements:
#   - CUDA 12.x toolkit  (nvcc on PATH, or CUDA_PATH set)
#   - Rust 1.76+
#
# Usage: bash scripts/build-linux-cuda.sh [cuda_version]
#   cuda_version  Optional CUDA version suffix, e.g. "12.6" (default: auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================"
echo "  XandLLM — Linux CUDA build"
echo "======================================================"
echo ""

# Verify Rust toolchain
if ! command -v cargo &>/dev/null; then
    echo "[ERROR] cargo not found. Install Rust via https://rustup.rs"
    exit 1
fi

echo "[INFO] Rust version: $(rustc --version)"

# Verify CUDA toolkit
if ! command -v nvcc &>/dev/null; then
    # Try common CUDA paths
    for candidate in /usr/local/cuda/bin /usr/local/cuda-12*/bin /opt/cuda/bin; do
        if [ -x "$candidate/nvcc" ]; then
            export PATH="$candidate:$PATH"
            break
        fi
    done
fi

if ! command -v nvcc &>/dev/null; then
    echo "[ERROR] nvcc not found. Install the CUDA 12.x toolkit from:"
    echo "        https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "  Alternatively build without GPU support:"
    echo "    bash scripts/build-linux.sh"
    exit 1
fi

NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "[INFO] CUDA toolkit: nvcc $NVCC_VERSION"

# Export canonical CUDA_PATH if not already set
if [ -z "${CUDA_PATH:-}" ]; then
    CUDA_PATH="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_PATH
fi
echo "[INFO] CUDA_PATH: $CUDA_PATH"

# Check for a compatible GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "[INFO] GPU: $GPU_INFO"
else
    echo "[WARN] nvidia-smi not found — cannot verify GPU. Build will proceed."
fi

echo "[INFO] Workspace: $WORKSPACE_ROOT"
echo ""

cd "$WORKSPACE_ROOT"

echo "[BUILD] Compiling xandllm with CUDA support ..."
echo "        (First build may take 5–15 minutes — CUDA kernels compile from source)"
echo ""

cargo install --path crates/xandllm-cli --features cuda --locked

echo ""
echo "======================================================"
echo "  Build complete — CUDA enabled"
echo "  Binary installed at: $(which xandllm 2>/dev/null || echo '~/.cargo/bin/xandllm')"
echo ""
echo "  Quick start:"
echo "    xandllm pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"
echo "    xandllm serve --model Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --gpu"
echo "======================================================"
