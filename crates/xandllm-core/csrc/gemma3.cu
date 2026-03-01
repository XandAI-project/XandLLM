/*
 * gemma3.cu — Gemma 3 transformer implementation
 *
 * Implements:
 *  - Token embedding lookup
 *  - RMSNorm with weight scale
 *  - RoPE (rotary position embeddings)
 *  - Grouped-query attention (GQA) with sliding window and logit soft-cap
 *  - SwiGLU feed-forward network
 *  - Final norm + lm_head projection
 *  - Per-layer KV cache
 */

#include "gemma3.h"
#include "quant.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── CUDA error check helper ─────────────────────────────────────────────── */

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        abort(); \
    } \
} while (0)

#define CUBLAS_CHECK(x) do { \
    cublasStatus_t _s = (x); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %d at %s:%d\n", _s, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)

/* ── Utility kernels ──────────────────────────────────────────────────────── */

__global__ void embed_lookup_kernel(const float *table, int token_id,
                                    float *out, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size)
        out[i] = table[(long long)token_id * hidden_size + i];
}

extern "C" void embed_lookup_cuda(const float *table, int token_id,
                                  float *out, int hidden_size, cudaStream_t s) {
    int threads = 256;
    int blocks  = (hidden_size + threads - 1) / threads;
    embed_lookup_kernel<<<blocks, threads, 0, s>>>(table, token_id, out, hidden_size);
}

/* ── RMSNorm ──────────────────────────────────────────────────────────────── */
/*
 * Two-pass RMSNorm:
 *   pass 1: compute sum(x^2) across all elements (warp/block reduction)
 *   pass 2: normalise and scale
 *
 * For vectors up to 8192 elements we use a single block of 256 threads.
 */

__global__ void rms_norm_kernel(float *x, const float *w, int n, float eps) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < n; i += blockDim.x)
        sum += x[i] * x[i];

    sdata[tid] = sum;
    __syncthreads();

    /* Block reduction */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    float scale = rsqrtf(sdata[0] / (float)n + eps);
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x)
        x[i] = x[i] * scale * w[i];
}

extern "C" void rms_norm_cuda(float *x, const float *w, int n, float eps, cudaStream_t s) {
    int threads = 256;
    rms_norm_kernel<<<1, threads, threads * sizeof(float), s>>>(x, w, n, eps);
}

/* ── Scale ────────────────────────────────────────────────────────────────── */

__global__ void scale_kernel(float *x, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scalar;
}

extern "C" void scale_cuda(float *x, float scalar, int n, cudaStream_t s) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads, 0, s>>>(x, scalar, n);
}

/* ── Element-wise add ─────────────────────────────────────────────────────── */

__global__ void add_kernel(float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

extern "C" void add_cuda(float *a, const float *b, int n, cudaStream_t s) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, s>>>(a, b, n);
}

/* ── RoPE ─────────────────────────────────────────────────────────────────── */
/*
 * Gemma 3 applies standard RoPE on the full head_dim.
 * For each head, pairs (x[2i], x[2i+1]) are rotated by angle pos/theta^(2i/d).
 */

__global__ void rope_kernel(float *qk, int n_heads, int head_dim, int pos, float theta) {
    int head = blockIdx.x;
    int i    = threadIdx.x; /* pair index: 0 .. head_dim/2 - 1 */

    if (head >= n_heads || i >= head_dim / 2) return;

    float *h = qk + head * head_dim;
    float angle = (float)pos / powf(theta, (2.0f * (float)i) / (float)head_dim);
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    float x0 = h[2 * i + 0];
    float x1 = h[2 * i + 1];
    h[2 * i + 0] = x0 * cos_a - x1 * sin_a;
    h[2 * i + 1] = x0 * sin_a + x1 * cos_a;
}

extern "C" void rope_cuda(float *qk, int n_heads, int head_dim, int pos,
                          float theta, cudaStream_t s) {
    dim3 grid(n_heads), block(head_dim / 2);
    rope_kernel<<<grid, block, 0, s>>>(qk, n_heads, head_dim, pos, theta);
}

/* ── SwiGLU ───────────────────────────────────────────────────────────────── */

__global__ void swiglu_kernel(float *gate, const float *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        /* SiLU: g * sigmoid(g) */
        g = g * (1.0f / (1.0f + expf(-g)));
        gate[i] = g * up[i];
    }
}

extern "C" void swiglu_cuda(float *gate, const float *up, int n, cudaStream_t s) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, s>>>(gate, up, n);
}

/* ── Soft-cap ─────────────────────────────────────────────────────────────── */

__global__ void softcap_kernel(float *x, float cap, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i] / cap) * cap;
}

extern "C" void softcap_cuda(float *x, float cap, int n, cudaStream_t s) {
    if (cap <= 0.0f) return;
    int threads = 256, blocks = (n + threads - 1) / threads;
    softcap_kernel<<<blocks, threads, 0, s>>>(x, cap, n);
}

/* ── Softmax in-place (single row) ───────────────────────────────────────── */

__global__ void softmax_kernel(float *x, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    float mx = -1e30f;
    for (int i = tid; i < n; i += blockDim.x)
        mx = fmaxf(mx, x[i]);
    smem[tid] = mx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    mx = smem[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = expf(x[i] - mx);
        x[i] = v;
        sum += v;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x)
        x[i] /= sum;
}

/* ── GQA decode attention ─────────────────────────────────────────────────── */
/*
 * Attention for a single query token (cache_pos is the 0-indexed position of
 * the current token; after writing to cache it becomes the new cache_pos).
 *
 * 1. Write k, v to kv_cache at position cache_pos.
 * 2. For each query head h:
 *    a. Find the corresponding kv head  (kv_h = h / (n_heads/n_kv_heads))
 *    b. Compute dot products q[h] · k_cache[t][kv_h] for t = 0..cache_pos
 *    c. Apply sliding window mask (if sliding_window > 0)
 *    d. Apply soft-cap (if logit_soft_cap > 0)
 *    e. Softmax
 *    f. Weighted sum over v_cache
 */

/* Write current k/v into the per-layer KV cache at position pos. */
__global__ void kv_cache_write_kernel(
        const float *k, const float *v,
        float *kv_k, float *kv_v,
        int pos, int n_kv_heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_kv_heads * head_dim;
    if (idx < total) {
        kv_k[pos * total + idx] = k[idx];
        kv_v[pos * total + idx] = v[idx];
    }
}

/* Compute q·k scores for one query head. */
__global__ void attn_scores_kernel(
        const float *q,      /* query head: head_dim floats */
        const float *kv_k,   /* full KV cache: (max_seq_len × n_kv_heads × head_dim) */
        float       *scores, /* output: seq_len scores */
        int kv_head_idx,
        int n_kv_heads,
        int head_dim,
        int seq_len,
        int pos,             /* current (latest) position */
        int sliding_window,
        float logit_soft_cap,
        float inv_sqrt_d)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t > pos) return;

    /* Apply sliding window: mask out tokens outside window */
    if (sliding_window > 0 && pos - t >= sliding_window) {
        scores[t] = -1e30f;
        return;
    }

    const float *k_t = kv_k + t * n_kv_heads * head_dim + kv_head_idx * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++)
        dot += q[d] * k_t[d];
    dot *= inv_sqrt_d;

    if (logit_soft_cap > 0.0f)
        dot = tanhf(dot / logit_soft_cap) * logit_soft_cap;

    scores[t] = dot;
}

/* Weighted sum of values using softmax scores. */
__global__ void attn_output_kernel(
        const float *scores,  /* softmax weights, seq_len floats */
        const float *kv_v,    /* full KV cache: (max_seq_len × n_kv_heads × head_dim) */
        float       *out,     /* output: head_dim floats */
        int kv_head_idx,
        int n_kv_heads,
        int head_dim,
        int seq_len)
{
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    float acc = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const float *v_t = kv_v + t * n_kv_heads * head_dim + kv_head_idx * head_dim;
        acc += scores[t] * v_t[d];
    }
    out[d] = acc;
}

extern "C" void attention_decode_cuda(
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
        cudaStream_t s)
{
    (void)cublas;  /* reserved for future prefill path */

    int seq_len = cache_pos + 1;   /* positions 0..cache_pos inclusive */
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    int gqa_ratio = n_heads / n_kv_heads;

    /* 1. Write current token's k, v into cache */
    {
        int total = n_kv_heads * head_dim;
        int threads = 256, blocks = (total + threads - 1) / threads;
        kv_cache_write_kernel<<<blocks, threads, 0, s>>>(
            k, v, kv_cache_k, kv_cache_v, cache_pos, n_kv_heads, head_dim);
    }

    /* 2. For each query head: scores + softmax + weighted-V */
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / gqa_ratio;
        const float *q_h = q + h * head_dim;
        float *out_h     = out + h * head_dim;
        float *sc        = scores + h * seq_len;

        /* Compute dot products */
        {
            int threads = 128;
            int blocks  = (seq_len + threads - 1) / threads;
            attn_scores_kernel<<<blocks, threads, 0, s>>>(
                q_h, kv_cache_k, sc,
                kv_h, n_kv_heads, head_dim,
                seq_len, cache_pos, sliding_window, logit_soft_cap, inv_sqrt_d);
        }

        /* Softmax over seq_len */
        {
            int threads = 256;
            softmax_kernel<<<1, threads, threads * sizeof(float), s>>>(sc, seq_len);
        }

        /* Weighted sum of values */
        {
            int threads = 256;
            int blocks  = (head_dim + threads - 1) / threads;
            attn_output_kernel<<<blocks, threads, 0, s>>>(
                sc, kv_cache_v, out_h, kv_h, n_kv_heads, head_dim, seq_len);
        }
    }
}

/* ── lm_head projection (vocab_size dot-products) ─────────────────────────
 * We use a custom kernel instead of cuBLAS to avoid large temporary allocs
 * for a single-row GEMV.
 */
__global__ void lm_head_kernel(const float *embed, const float *x,
                                float *logits, int vocab_size, int hidden_size)
{
    int row  = blockIdx.x;
    int lane = threadIdx.x;

    if (row >= vocab_size) return;

    const float *w = embed + (long long)row * hidden_size;
    float acc = 0.0f;
    for (int i = lane; i < hidden_size; i += 32)
        acc += w[i] * x[i];

    /* Warp reduction */
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, offset);

    if (lane == 0) logits[row] = acc;
}

/* ── Weight upload helpers ────────────────────────────────────────────────── */

static void *upload_tensor(const GgufFile *gf, const char *name, size_t *out_bytes) {
    GgufTensorInfo ti;
    if (gguf_find_tensor(gf, name, &ti) != 0) {
        fprintf(stderr, "gemma3_init: tensor '%s' not found in GGUF\n", name);
        return NULL;
    }

    /* Compute byte size */
    size_t n_elems = 1;
    for (uint32_t d = 0; d < ti.ndims; d++) n_elems *= ti.shape[d];

    size_t block_sz = gguf_block_size(ti.dtype);
    int    block_el = gguf_block_elems(ti.dtype);
    size_t n_bytes;
    if (block_el <= 1) {
        n_bytes = n_elems * block_sz;  /* F32/F16: block_sz = elem size */
    } else {
        n_bytes = (n_elems / block_el) * block_sz;
    }

    const void *host_data = gguf_tensor_data(gf, name);
    if (!host_data) {
        fprintf(stderr, "gemma3_init: data for '%s' not mapped\n", name);
        return NULL;
    }

    void *dev;
    CUDA_CHECK(cudaMalloc(&dev, n_bytes));
    CUDA_CHECK(cudaMemcpy(dev, host_data, n_bytes, cudaMemcpyHostToDevice));
    if (out_bytes) *out_bytes = n_bytes;
    return dev;
}

static float *upload_f32_tensor(const GgufFile *gf, const char *name, int expected_n) {
    GgufTensorInfo ti;
    if (gguf_find_tensor(gf, name, &ti) != 0) {
        fprintf(stderr, "gemma3_init: norm tensor '%s' not found\n", name);
        return NULL;
    }
    size_t n_elems = 1;
    for (uint32_t d = 0; d < ti.ndims; d++) n_elems *= ti.shape[d];
    if ((int)n_elems != expected_n) {
        fprintf(stderr, "gemma3_init: '%s' expected %d elements, got %zu\n",
                name, expected_n, n_elems);
        return NULL;
    }
    const void *src = gguf_tensor_data(gf, name);
    float *dev;
    CUDA_CHECK(cudaMalloc(&dev, n_elems * sizeof(float)));
    /* Tensors may be stored as f32 or f16 in GGUF */
    if (ti.dtype == GGUF_TYPE_F32) {
        CUDA_CHECK(cudaMemcpy(dev, src, n_elems * sizeof(float), cudaMemcpyHostToDevice));
    } else if (ti.dtype == GGUF_TYPE_F16) {
        /* Convert f16 → f32 on the host before upload */
        float *tmp = (float *)malloc(n_elems * sizeof(float));
        const uint16_t *h16 = (const uint16_t *)src;
        for (size_t i = 0; i < n_elems; i++) {
            /* Manual f16→f32 conversion */
            uint32_t exp  = (h16[i] >> 10) & 0x1f;
            uint32_t mant = h16[i] & 0x3ff;
            uint32_t sign = h16[i] >> 15;
            uint32_t f32bits;
            if (exp == 0)       f32bits = (sign << 31) | (mant << 13);
            else if (exp == 31) f32bits = (sign << 31) | 0x7f800000u | (mant << 13);
            else                f32bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
            memcpy(&tmp[i], &f32bits, 4);
        }
        CUDA_CHECK(cudaMemcpy(dev, tmp, n_elems * sizeof(float), cudaMemcpyHostToDevice));
        free(tmp);
    } else {
        fprintf(stderr, "gemma3_init: norm tensor '%s' has unexpected dtype %u\n",
                name, ti.dtype);
        cudaFree(dev);
        return NULL;
    }
    return dev;
}

/* ── gemma3_init ──────────────────────────────────────────────────────────── */

extern "C" Gemma3Model *gemma3_init(const GgufFile *gf, const Gemma3Config *cfg,
                                     int gpu_id)
{
    CUDA_CHECK(cudaSetDevice(gpu_id));

    Gemma3Model *m = (Gemma3Model *)calloc(1, sizeof(Gemma3Model));
    if (!m) return NULL;
    m->cfg = *cfg;

    CUBLAS_CHECK(cublasCreate(&m->cublas));
    CUDA_CHECK(cudaStreamCreate(&m->stream));
    CUBLAS_CHECK(cublasSetStream(m->cublas, m->stream));

    int H  = cfg->hidden_size;
    int NL = cfg->n_layers;
    int VS = cfg->vocab_size;

    /* ── Token embeddings ── */
    {
        GgufTensorInfo ti;
        if (gguf_find_tensor(gf, "token_embd.weight", &ti) != 0) {
            fprintf(stderr, "gemma3_init: token_embd.weight not found\n");
            goto fail;
        }
        size_t n_bytes = (size_t)VS * H * sizeof(float);
        CUDA_CHECK(cudaMalloc(&m->embed_tokens, n_bytes));

        const void *src = gguf_tensor_data(gf, "token_embd.weight");
        if (ti.dtype == GGUF_TYPE_F32) {
            CUDA_CHECK(cudaMemcpy(m->embed_tokens, src, n_bytes, cudaMemcpyHostToDevice));
        } else if (ti.dtype == GGUF_TYPE_F16) {
            float *tmp = (float *)malloc(n_bytes);
            const uint16_t *h16 = (const uint16_t *)src;
            for (int i = 0; i < VS * H; i++) {
                uint32_t exp  = (h16[i] >> 10) & 0x1f;
                uint32_t mant = h16[i] & 0x3ff;
                uint32_t sign = h16[i] >> 15;
                uint32_t f32bits;
                if (exp == 0)       f32bits = (sign << 31) | (mant << 13);
                else if (exp == 31) f32bits = (sign << 31) | 0x7f800000u | (mant << 13);
                else                f32bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                memcpy(&tmp[i], &f32bits, 4);
            }
            CUDA_CHECK(cudaMemcpy(m->embed_tokens, tmp, n_bytes, cudaMemcpyHostToDevice));
            free(tmp);
        } else {
            fprintf(stderr, "gemma3_init: unsupported embed dtype %u\n", ti.dtype);
            goto fail;
        }
    }
    /* Gemma 3 ties lm_head to embed_tokens */
    m->lm_head = m->embed_tokens;

    /* ── Final norm ── */
    m->final_norm = upload_f32_tensor(gf, "output_norm.weight", H);
    if (!m->final_norm) goto fail;

    /* ── Layers ── */
    m->layers = (Gemma3Layer *)calloc(NL, sizeof(Gemma3Layer));
    if (!m->layers) goto fail;

    for (int l = 0; l < NL; l++) {
        Gemma3Layer *lay = &m->layers[l];
        char name[256];

#define UPLOAD_PROJ(field, key_fmt, dtype_field) \
        snprintf(name, sizeof(name), key_fmt, l); \
        { GgufTensorInfo _ti; \
          if (gguf_find_tensor(gf, name, &_ti) != 0) { \
              fprintf(stderr, "gemma3_init: tensor '%s' not found\n", name); \
              goto fail; \
          } \
          lay->dtype_field = _ti.dtype; \
          lay->field = upload_tensor(gf, name, NULL); \
          if (!lay->field) goto fail; }

        UPLOAD_PROJ(q,        "blk.%d.attn_q.weight",        q_dtype)
        UPLOAD_PROJ(k,        "blk.%d.attn_k.weight",        k_dtype)
        UPLOAD_PROJ(v,        "blk.%d.attn_v.weight",        v_dtype)
        UPLOAD_PROJ(o,        "blk.%d.attn_output.weight",   o_dtype)
        UPLOAD_PROJ(ffn_gate, "blk.%d.ffn_gate.weight",      gate_dtype)
        UPLOAD_PROJ(ffn_up,   "blk.%d.ffn_up.weight",        up_dtype)
        UPLOAD_PROJ(ffn_down, "blk.%d.ffn_down.weight",      down_dtype)
#undef UPLOAD_PROJ

#define UPLOAD_NORM(field, key_fmt) \
        snprintf(name, sizeof(name), key_fmt, l); \
        lay->field = upload_f32_tensor(gf, name, H); \
        if (!lay->field) goto fail;

        UPLOAD_NORM(attn_norm,      "blk.%d.attn_norm.weight")
        UPLOAD_NORM(post_attn_norm, "blk.%d.post_attention_norm.weight")
        UPLOAD_NORM(ffn_norm,       "blk.%d.ffn_norm.weight")
        UPLOAD_NORM(post_ffn_norm,  "blk.%d.post_ffw_norm.weight")
#undef UPLOAD_NORM

        /* KV cache */
        size_t kv_sz = (size_t)cfg->max_seq_len * cfg->n_kv_heads
                       * cfg->head_dim * sizeof(float);
        CUDA_CHECK(cudaMalloc(&lay->kv_k, kv_sz));
        CUDA_CHECK(cudaMalloc(&lay->kv_v, kv_sz));
    }

    /* ── Working buffers ── */
    int ffn_size = cfg->ffn_size;
    int n_heads  = cfg->n_heads;
    int n_kv     = cfg->n_kv_heads;
    int hd       = cfg->head_dim;
    int max_sl   = cfg->max_seq_len;

    CUDA_CHECK(cudaMalloc(&m->d_x,           H         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_attn_in,     H         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_q,           n_heads * hd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_k,           n_kv    * hd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_v,           n_kv    * hd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_attn_out,    n_heads * hd * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_ffn_in,      H         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_gate,        ffn_size  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_up,          ffn_size  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_ffn_out,     H         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_logits,      VS        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m->d_attn_scores, n_heads * max_sl * sizeof(float)));

    m->cache_pos = 0;
    return m;

fail:
    gemma3_free(m);
    return NULL;
}

/* ── gemma3_free ──────────────────────────────────────────────────────────── */

extern "C" void gemma3_free(Gemma3Model *m) {
    if (!m) return;

#define CFREE(p) do { if (p) { cudaFree(p); (p) = NULL; } } while(0)

    CFREE(m->embed_tokens);
    CFREE(m->final_norm);
    CFREE(m->d_x);
    CFREE(m->d_attn_in);
    CFREE(m->d_q);
    CFREE(m->d_k);
    CFREE(m->d_v);
    CFREE(m->d_attn_out);
    CFREE(m->d_ffn_in);
    CFREE(m->d_gate);
    CFREE(m->d_up);
    CFREE(m->d_ffn_out);
    CFREE(m->d_logits);
    CFREE(m->d_attn_scores);

    if (m->layers) {
        for (int l = 0; l < m->cfg.n_layers; l++) {
            Gemma3Layer *lay = &m->layers[l];
            CFREE(lay->q); CFREE(lay->k); CFREE(lay->v); CFREE(lay->o);
            CFREE(lay->ffn_gate); CFREE(lay->ffn_up); CFREE(lay->ffn_down);
            CFREE(lay->attn_norm); CFREE(lay->post_attn_norm);
            CFREE(lay->ffn_norm);  CFREE(lay->post_ffn_norm);
            CFREE(lay->kv_k); CFREE(lay->kv_v);
        }
        free(m->layers);
    }
#undef CFREE

    if (m->cublas) cublasDestroy(m->cublas);
    if (m->stream) cudaStreamDestroy(m->stream);
    free(m);
}

/* ── gemma3_reset_kv ──────────────────────────────────────────────────────── */

extern "C" void gemma3_reset_kv(Gemma3Model *m) {
    m->cache_pos = 0;
}

/* ── gemma3_forward ───────────────────────────────────────────────────────── */

extern "C" float *gemma3_forward(Gemma3Model *m, int token_id, int pos) {
    const Gemma3Config *cfg = &m->cfg;
    cudaStream_t s = m->stream;
    int H   = cfg->hidden_size;
    int hd  = cfg->head_dim;
    float eps = cfg->rms_norm_eps;

    /* 1. Embed token */
    embed_lookup_cuda(m->embed_tokens, token_id, m->d_x, H, s);

    /* Gemma 3 scales embeddings by sqrt(hidden_size) */
    float embed_scale = sqrtf((float)H);
    scale_cuda(m->d_x, embed_scale, H, s);

    /* 2. Transformer layers */
    for (int l = 0; l < cfg->n_layers; l++) {
        Gemma3Layer *lay = &m->layers[l];

        /* ── Attention sub-layer ── */

        /* Pre-attention RMSNorm (writes into d_attn_in, preserves d_x) */
        CUDA_CHECK(cudaMemcpyAsync(m->d_attn_in, m->d_x, H * sizeof(float),
                                   cudaMemcpyDeviceToDevice, s));
        rms_norm_cuda(m->d_attn_in, lay->attn_norm, H, eps, s);

        /* Q, K, V projections */
        int q_dim = cfg->n_heads    * hd;
        int kv_dim = cfg->n_kv_heads * hd;
        gguf_matvec(lay->q_dtype, lay->q, m->d_attn_in, m->d_q, q_dim,  H, s);
        gguf_matvec(lay->k_dtype, lay->k, m->d_attn_in, m->d_k, kv_dim, H, s);
        gguf_matvec(lay->v_dtype, lay->v, m->d_attn_in, m->d_v, kv_dim, H, s);

        /* RoPE on Q and K */
        rope_cuda(m->d_q, cfg->n_heads,    hd, pos, cfg->rope_theta, s);
        rope_cuda(m->d_k, cfg->n_kv_heads, hd, pos, cfg->rope_theta, s);

        /* GQA attention */
        float *scores = m->d_attn_scores + 0; /* use first n_heads*seq_len section */
        attention_decode_cuda(
            m->d_q, m->d_k, m->d_v,
            lay->kv_k, lay->kv_v,
            scores, m->d_attn_out,
            cfg->n_heads, cfg->n_kv_heads, hd,
            pos, cfg->sliding_window, cfg->logit_soft_cap,
            m->cublas, s);

        /* O projection: d_ffn_out reused as scratch for attn output projection */
        gguf_matvec(lay->o_dtype, lay->o, m->d_attn_out, m->d_ffn_out, H,
                    cfg->n_heads * hd, s);

        /* Post-attention norm */
        rms_norm_cuda(m->d_ffn_out, lay->post_attn_norm, H, eps, s);

        /* Residual add */
        add_cuda(m->d_x, m->d_ffn_out, H, s);

        /* ── FFN sub-layer ── */

        /* Pre-FFN RMSNorm */
        CUDA_CHECK(cudaMemcpyAsync(m->d_ffn_in, m->d_x, H * sizeof(float),
                                   cudaMemcpyDeviceToDevice, s));
        rms_norm_cuda(m->d_ffn_in, lay->ffn_norm, H, eps, s);

        /* Gate and Up projections */
        gguf_matvec(lay->gate_dtype, lay->ffn_gate, m->d_ffn_in, m->d_gate,
                    cfg->ffn_size, H, s);
        gguf_matvec(lay->up_dtype,   lay->ffn_up,   m->d_ffn_in, m->d_up,
                    cfg->ffn_size, H, s);

        /* SwiGLU: gate = silu(gate) * up */
        swiglu_cuda(m->d_gate, m->d_up, cfg->ffn_size, s);

        /* Down projection */
        gguf_matvec(lay->down_dtype, lay->ffn_down, m->d_gate, m->d_ffn_out,
                    H, cfg->ffn_size, s);

        /* Post-FFN norm */
        rms_norm_cuda(m->d_ffn_out, lay->post_ffn_norm, H, eps, s);

        /* Residual add */
        add_cuda(m->d_x, m->d_ffn_out, H, s);
    }

    /* 3. Final RMSNorm */
    rms_norm_cuda(m->d_x, m->final_norm, H, eps, s);

    /* 4. lm_head (tied to embed_tokens) */
    {
        int VS = cfg->vocab_size;
        lm_head_kernel<<<VS, 32, 0, s>>>(m->lm_head, m->d_x, m->d_logits, VS, H);
    }

    /* 5. Final logit soft-cap */
    if (cfg->final_logit_soft_cap > 0.0f)
        softcap_cuda(m->d_logits, cfg->final_logit_soft_cap, cfg->vocab_size, s);

    /* Advance KV cache position */
    m->cache_pos = pos + 1;

    CUDA_CHECK(cudaStreamSynchronize(s));
    return m->d_logits;
}
