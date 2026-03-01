/*
 * gemma3.h — Gemma 3 transformer model structs and forward-pass declaration
 *
 * Covers Gemma 3 architecture:
 *   - RMSNorm (with optional post-norm scale)
 *   - Grouped-query attention (GQA) with sliding-window support
 *   - RoPE (rotary position embeddings) per Gemma 3 spec
 *   - SwiGLU feed-forward network
 *   - Per-layer KV cache (CUDA device memory)
 *
 * All device pointers are on the same CUDA device selected at creation time.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Model configuration ──────────────────────────────────────────────────── */

typedef struct {
    int   hidden_size;       /* gemma3.embedding_length   (e.g. 5120 for 12B) */
    int   n_layers;          /* gemma3.block_count        (e.g. 46  for 12B) */
    int   n_heads;           /* gemma3.attention.head_count */
    int   n_kv_heads;        /* gemma3.attention.head_count_kv */
    int   head_dim;          /* gemma3.attention.key_length */
    int   ffn_size;          /* gemma3.feed_forward_length */
    int   vocab_size;        /* tokenizer.ggml.tokens count */
    int   max_seq_len;       /* maximum context length */
    float rope_theta;        /* gemma3.rope.freq_base */
    int   sliding_window;    /* gemma3.attention.sliding_window (0 = disabled) */
    float logit_soft_cap;    /* gemma3.attention.logit_softcapping (0 = none) */
    float final_logit_soft_cap; /* gemma3.final_logit_softcapping (0 = none) */
    float rms_norm_eps;      /* gemma3.attention.layer_norm_rms_epsilon */
} Gemma3Config;

/* ── Per-layer GPU weight pointers ───────────────────────────────────────── */

typedef struct {
    /* Attention projections */
    void    *q;          /* query   weight (hidden × n_heads   × head_dim) */
    void    *k;          /* key     weight (hidden × n_kv_heads × head_dim) */
    void    *v;          /* value   weight (hidden × n_kv_heads × head_dim) */
    void    *o;          /* output  weight (n_heads × head_dim × hidden) */

    /* Feed-forward projections */
    void    *ffn_gate;   /* gate projection (hidden × ffn_size) */
    void    *ffn_up;     /* up   projection (hidden × ffn_size) */
    void    *ffn_down;   /* down projection (ffn_size × hidden) */

    /* RMSNorm weight vectors (float32, length = hidden_size) */
    float   *attn_norm;
    float   *post_attn_norm;
    float   *ffn_norm;
    float   *post_ffn_norm;

    /* Dtype tags for each projection (GGUF_TYPE_*) */
    uint32_t q_dtype, k_dtype, v_dtype, o_dtype;
    uint32_t gate_dtype, up_dtype, down_dtype;

    /* Per-layer KV cache (device, pre-allocated for max_seq_len) */
    float   *kv_k;     /* shape: (max_seq_len × n_kv_heads × head_dim) */
    float   *kv_v;     /* shape: (max_seq_len × n_kv_heads × head_dim) */
} Gemma3Layer;

/* ── Full model handle ────────────────────────────────────────────────────── */

typedef struct {
    Gemma3Config  cfg;

    /* Token embedding table (vocab_size × hidden_size, float32 on device) */
    float        *embed_tokens;

    /* Layer data */
    Gemma3Layer  *layers;     /* cfg.n_layers elements */

    /* Final (output) norm weights (float32, length = hidden_size) */
    float        *final_norm;

    /* Output (lm_head) weight matrix (vocab_size × hidden_size).
     * Gemma 3 ties embed_tokens to lm_head — they share the same data. */
    float        *lm_head;   /* points into embed_tokens */

    /* Working buffers (device) reused across forward passes */
    float        *d_x;       /* residual stream (hidden_size)                  */
    float        *d_attn_in; /* normed input to attention (hidden_size)         */
    float        *d_q;       /* query   vectors (n_heads × head_dim)           */
    float        *d_k;       /* key     vectors (n_kv_heads × head_dim)        */
    float        *d_v;       /* value   vectors (n_kv_heads × head_dim)        */
    float        *d_attn_out;/* post-attention  (hidden_size)                  */
    float        *d_ffn_in;  /* normed input to FFN (hidden_size)              */
    float        *d_gate;    /* gate projection output (ffn_size)              */
    float        *d_up;      /* up   projection output (ffn_size)              */
    float        *d_ffn_out; /* FFN output (hidden_size)                       */
    float        *d_logits;  /* final logits (vocab_size)                      */

    /* Attention softmax scratch — size depends on max_seq_len and n_heads */
    float        *d_attn_scores; /* (n_heads × max_seq_len) — qk dot products */

    /* cuBLAS handle (shared across layers) */
    cublasHandle_t cublas;

    /* CUDA stream */
    cudaStream_t   stream;

    /* Current KV-cache fill position */
    int           cache_pos;
} Gemma3Model;

/* ── CUDA kernel declarations ────────────────────────────────────────────── */

/*
 * RMSNorm in-place: x[i] *= w[i] / rms(x)
 * n = length of x (= hidden_size)
 */
void rms_norm_cuda(float *x, const float *w, int n, float eps, cudaStream_t s);

/*
 * RoPE applied in-place to a (n_heads × head_dim) query or key tensor
 * at sequence position `pos`.  Gemma 3 uses standard RoPE with full
 * head_dim rotation (no partial rotation).
 */
void rope_cuda(float *qk, int n_heads, int head_dim, int pos,
               float theta, cudaStream_t s);

/*
 * Embedding look-up: copy row token_id from embed_table into out.
 * embed_table: (vocab_size × hidden_size) float32 on device.
 */
void embed_lookup_cuda(const float *embed_table, int token_id,
                       float *out, int hidden_size, cudaStream_t s);

/*
 * Element-wise scale: x[i] *= scalar
 */
void scale_cuda(float *x, float scalar, int n, cudaStream_t s);

/*
 * SwiGLU: gate[i] = silu(gate[i]) * up[i], result stored in gate.
 */
void swiglu_cuda(float *gate, const float *up, int n, cudaStream_t s);

/*
 * Element-wise add: a[i] += b[i]
 */
void add_cuda(float *a, const float *b, int n, cudaStream_t s);

/*
 * Soft-cap logits: x[i] = tanh(x[i] / cap) * cap
 */
void softcap_cuda(float *x, float cap, int n, cudaStream_t s);

/*
 * Grouped-query attention for a single query token (decode step).
 * Writes new K/V to kv_cache_k/kv_cache_v at cache_pos.
 * Computes softmax(Q K^T / sqrt(head_dim)) V and accumulates into out.
 *
 * q:           (n_heads  × head_dim) float32 device
 * k, v:        (n_kv_heads × head_dim) float32 device  (for current token)
 * kv_cache_k:  (max_seq_len × n_kv_heads × head_dim) float32 device
 * kv_cache_v:  same layout
 * scores:      scratch buffer (n_heads × cache_pos+1)
 * out:         (n_heads × head_dim) float32 device — overwritten
 */
void attention_decode_cuda(
        const float *q,
        const float *k,
        const float *v,
        float *kv_cache_k,
        float *kv_cache_v,
        float *scores,
        float *out,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        int cache_pos,
        int sliding_window,
        float logit_soft_cap,
        cublasHandle_t cublas,
        cudaStream_t s);

/* ── Public API ──────────────────────────────────────────────────────────── */

/*
 * Allocate and initialise a Gemma3Model from a parsed GgufFile.
 * Allocates all device memory and uploads weights.
 * Returns NULL on failure.
 *
 * GgufFile is defined in gguf.h; callers must include gguf.h before gemma3.h.
 * We use a struct tag reference here so gemma3.h can be included without gguf.h
 * when only the free/reset/forward declarations are needed.
 */
#ifndef GGUF_H  /* avoid forward-decl if gguf.h was already included */
struct GgufFile;
#endif
Gemma3Model *gemma3_init(const struct GgufFile *gf, const Gemma3Config *cfg,
                         int gpu_id);

/* Free all device and host memory. */
void gemma3_free(Gemma3Model *m);

/* Reset the KV cache (set cache_pos = 0). */
void gemma3_reset_kv(Gemma3Model *m);

/*
 * Run one forward pass for a single token at position `pos`.
 * Returns a device pointer to d_logits (vocab_size floats).
 * The pointer remains valid until the next call to gemma3_forward.
 */
float *gemma3_forward(Gemma3Model *m, int token_id, int pos);

#ifdef __cplusplus
}
#endif
