# ─── Stage 1: dependency planner (cargo-chef) ──────────────────────────────────
#
# Uses CUDA 12.6 — the host driver version does not need to match;
# the NVIDIA Container Toolkit (nvidia-docker2) maps any host driver ≥ 12.6
# to this runtime.  This lets the image build and run on hosts with CUDA 13.x
# (e.g. RTX 5000 Blackwell cards) without any host-side changes.
FROM nvcr.io/nvidia/cuda:12.6.2-devel-ubuntu22.04 AS chef

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Install Rust + cargo-chef in one layer.
# Using the absolute path for the first cargo invocation avoids PATH
# sourcing issues that occur when the CUDA base image reinitialises PATH
# between RUN steps.
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable \
    && /root/.cargo/bin/cargo install cargo-chef --locked
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# ─── Stage 2: compute the recipe (dependency manifest) ─────────────────────────
FROM chef AS planner

COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ─── Stage 3: build dependencies (cached layer) ────────────────────────────────
FROM chef AS builder

# Tell candle-kernels / bindgen_cuda which SM to compile for.
# This bypasses the nvidia-smi call that would otherwise fail during
# docker build (GPU is not exposed to the build daemon by default).
#
# Override at build time:
#   docker build --build-arg CUDA_COMPUTE_CAP=120 ...   # RTX 5060 Ti (Blackwell)
#   docker build --build-arg CUDA_COMPUTE_CAP=89  ...   # RTX 4000 series (Ada)
#   docker build --build-arg CUDA_COMPUTE_CAP=86  ...   # RTX 3000 series (Ampere)
#
# Default is 89 (Ada Lovelace). PTX from SM_89 runs on SM_120 (Blackwell) via
# NVIDIA's forward-compatibility JIT with no functional difference.
ARG CUDA_COMPUTE_CAP=89
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

COPY --from=planner /app/recipe.json recipe.json

# Build only dependencies — this layer is cached unless Cargo.toml changes
RUN cargo chef cook --release --features cuda --recipe-path recipe.json

# Build the actual binary
COPY . .
RUN cargo build --release --features cuda --bin xandllm

# ─── Stage 4: slim runtime image ───────────────────────────────────────────────
FROM nvcr.io/nvidia/cuda:12.6.2-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /app/target/release/xandllm /usr/local/bin/xandllm

# Copy default config and internal dataset (for distillation default fallback)
COPY config/            /app/config/
COPY internal/dataset/  /app/internal/dataset/

WORKDIR /app

RUN mkdir -p /root/.cache/xandllm /app/output

EXPOSE 11435

ENTRYPOINT ["xandllm"]
CMD ["serve", "--model", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"]
