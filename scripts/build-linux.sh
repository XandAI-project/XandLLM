#!/usr/bin/env bash
# Build XandLLM CLI (CPU-only) and install it to ~/.cargo/bin
# Usage: bash scripts/build-linux.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================"
echo "  XandLLM — Linux CPU build"
echo "======================================================"
echo ""

# Verify Rust toolchain
if ! command -v cargo &>/dev/null; then
    echo "[ERROR] cargo not found. Install Rust via https://rustup.rs"
    exit 1
fi

echo "[INFO] Rust version: $(rustc --version)"
echo "[INFO] Workspace: $WORKSPACE_ROOT"
echo ""

cd "$WORKSPACE_ROOT"

echo "[BUILD] Compiling xandllm (CPU-only) ..."
cargo install --path crates/xandllm-cli --locked

echo ""
echo "======================================================"
echo "  Build complete — CPU-only"
echo "  Binary installed at: $(which xandllm 2>/dev/null || echo '~/.cargo/bin/xandllm')"
echo ""
echo "  Quick start:"
echo "    xandllm pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_0"
echo "    xandllm serve --model Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
echo "======================================================"
