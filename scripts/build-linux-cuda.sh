#!/usr/bin/env bash
# Build XandLLM CLI with CUDA GPU support and install it to ~/.cargo/bin
#
# Requirements:
#   - CUDA 12.x toolkit  (nvcc on PATH, or CUDA_PATH set)
#     NOTE: CUDA 13.0 is NOT yet supported by the underlying cudarc crate.
#           If only CUDA 13.0 is installed this script will try to locate a
#           parallel CUDA 12.x installation, and print installation instructions
#           if none is found.
#   - Rust 1.76+
#
# Usage: bash scripts/build-linux-cuda.sh

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

# ── Find a usable CUDA 12.x installation ──────────────────────────────────────

find_cuda12() {
    # Priority list of known CUDA 12.x paths (newest first)
    local candidates=(
        /usr/local/cuda-12.8/bin
        /usr/local/cuda-12.6/bin
        /usr/local/cuda-12.5/bin
        /usr/local/cuda-12.4/bin
        /usr/local/cuda-12.3/bin
        /usr/local/cuda-12.2/bin
        /usr/local/cuda-12.1/bin
        /usr/local/cuda-12.0/bin
        /opt/cuda-12.8/bin
        /opt/cuda-12.6/bin
        /opt/cuda-12.4/bin
        /opt/cuda/bin
    )
    for dir in "${candidates[@]}"; do
        if [[ -x "$dir/nvcc" ]]; then
            local ver
            ver=$("$dir/nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || true)
            if [[ "$ver" == 12.* ]]; then
                echo "$dir"
                return 0
            fi
        fi
    done
    # Also scan /usr/local/cuda-12.*/bin dynamically
    for dir in /usr/local/cuda-12.*/bin; do
        if [[ -x "$dir/nvcc" ]]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

# Resolve nvcc — prefer what's already on PATH
NVCC_BIN=""
if command -v nvcc &>/dev/null; then
    NVCC_BIN="$(command -v nvcc)"
fi

if [[ -n "$NVCC_BIN" ]]; then
    NVCC_VERSION=$("$NVCC_BIN" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "[INFO] Found nvcc $NVCC_VERSION at $NVCC_BIN"

    # CUDA 13.x is not yet supported by the cudarc crate used by candle.
    if [[ "$NVCC_VERSION" == 13.* ]]; then
        echo ""
        echo "[WARN] CUDA $NVCC_VERSION detected — cudarc (the Rust CUDA binding)"
        echo "       does not yet support CUDA 13.x. Searching for a CUDA 12.x"
        echo "       installation to use instead..."
        echo ""

        CUDA12_BIN=$(find_cuda12 || true)
        if [[ -n "$CUDA12_BIN" ]]; then
            NVCC_BIN="$CUDA12_BIN/nvcc"
            NVCC_VERSION=$("$NVCC_BIN" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
            export PATH="$CUDA12_BIN:$PATH"
            export CUDA_PATH="$(dirname "$CUDA12_BIN")"
            echo "[INFO] Using CUDA $NVCC_VERSION at $CUDA12_BIN"
        else
            echo "[ERROR] No CUDA 12.x installation found."
            echo ""
            echo "  The cudarc crate (used by candle) does not yet support CUDA 13.0."
            echo "  Your RTX 5060 Ti (Blackwell) supports CUDA 12.4+, so you can install"
            echo "  CUDA 12.x alongside CUDA 13.0:"
            echo ""
            echo "  Option 1 — Install CUDA 12.8 via apt (Ubuntu 22.04 / 24.04):"
            echo "    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
            echo "    sudo dpkg -i cuda-keyring_1.1-1_all.deb"
            echo "    sudo apt update"
            echo "    sudo apt install -y cuda-toolkit-12-8"
            echo ""
            echo "  Option 2 — Install via the CUDA runfile (keeps both 12.x and 13.x):"
            echo "    https://developer.nvidia.com/cuda-12-8-0-download-archive"
            echo ""
            echo "  After installing CUDA 12.x, re-run this script."
            echo "  (CUDA 13.0 support in cudarc is tracked upstream:"
            echo "   https://github.com/huggingface/candle/issues/3087)"
            exit 1
        fi
    fi
else
    # nvcc not on PATH — search common locations for any 12.x install
    CUDA12_BIN=$(find_cuda12 || true)
    if [[ -n "$CUDA12_BIN" ]]; then
        export PATH="$CUDA12_BIN:$PATH"
        export CUDA_PATH="$(dirname "$CUDA12_BIN")"
        NVCC_VERSION=$("$CUDA12_BIN/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        echo "[INFO] Found CUDA $NVCC_VERSION at $CUDA12_BIN"
    else
        echo "[ERROR] nvcc not found and no CUDA 12.x installation located."
        echo ""
        echo "  Install CUDA 12.x from:"
        echo "    https://developer.nvidia.com/cuda-12-8-0-download-archive"
        echo ""
        echo "  Or build without GPU support:"
        echo "    bash scripts/build-linux.sh"
        exit 1
    fi
fi

# Set CUDA_PATH from the resolved nvcc if not already set
if [[ -z "${CUDA_PATH:-}" ]]; then
    CUDA_PATH="$(dirname "$(dirname "$NVCC_BIN")")"
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

echo "[BUILD] Compiling xandllm with CUDA $NVCC_VERSION support ..."
echo "        (First build may take 5–15 minutes — CUDA kernels compile from source)"
echo ""

cargo install --path crates/xandllm-cli --features cuda --locked

echo ""
echo "======================================================"
echo "  Build complete — CUDA $NVCC_VERSION enabled"
echo "  Binary: $(which xandllm 2>/dev/null || echo '~/.cargo/bin/xandllm')"
echo ""
echo "  Quick start:"
echo "    xandllm pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"
echo "    xandllm serve --model Qwen/Qwen2.5-Coder-7B-Instruct-GGUF --gpu"
echo "======================================================"
