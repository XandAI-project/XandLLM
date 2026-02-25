# ─── Stage 1: dependency planner (cargo-chef) ─────────────────────────────────
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04 AS chef

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install cargo-chef for layer-cached builds
RUN cargo install cargo-chef --locked

WORKDIR /app

# ─── Stage 2: compute the recipe (dependency manifest) ────────────────────────
FROM chef AS planner

COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ─── Stage 3: build dependencies (cached layer) ───────────────────────────────
FROM chef AS builder

COPY --from=planner /app/recipe.json recipe.json

# Build only dependencies — this layer is cached unless Cargo.toml changes
RUN cargo chef cook --release --features cuda --recipe-path recipe.json

# Build the actual binary
COPY . .
RUN cargo build --release --features cuda --bin xandllm

# ─── Stage 4: slim runtime image ──────────────────────────────────────────────
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /app/target/release/xandllm /usr/local/bin/xandllm

# Copy default config
COPY config/ /app/config/

WORKDIR /app

# Default model cache directory
RUN mkdir -p /root/.cache/xandllm

EXPOSE 11435

ENTRYPOINT ["xandllm"]
# Default: serve LLaMA 3.1 8B Instruct Q4_K_M (GGUF quantized).
# Override MODEL_ID via docker run -e MODEL_ID=... or docker-compose environment.
CMD ["serve", "--model", "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"]
