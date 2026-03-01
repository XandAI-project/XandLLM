/*
 * engine.c — XandEngine implementation
 *
 * Orchestrates:
 *  1. GGUF file parsing (gguf.c)
 *  2. Gemma 3 config extraction from GGUF metadata
 *  3. Weight upload to GPU and model init (gemma3.cu)
 *  4. Sequential token-by-token forward pass (gemma3_forward)
 *  5. D2H copy of logits to caller-provided CPU buffer
 */
#include "engine.h"
#include "gguf.h"
#include "gemma3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Engine struct ────────────────────────────────────────────────────────── */

struct XandEngine {
    GgufFile    *gf;
    Gemma3Model *model;
    int          vocab_size;
    int          gpu_id;
};

/* ── Config extraction ────────────────────────────────────────────────────── */

static int extract_config(const GgufFile *gf, size_t max_ctx, Gemma3Config *cfg) {
    cfg->hidden_size    = (int)gguf_meta_u32(gf, "gemma3.embedding_length",    0);
    cfg->n_layers       = (int)gguf_meta_u32(gf, "gemma3.block_count",         0);
    cfg->n_heads        = (int)gguf_meta_u32(gf, "gemma3.attention.head_count",0);
    cfg->n_kv_heads     = (int)gguf_meta_u32(gf, "gemma3.attention.head_count_kv", cfg->n_heads);
    cfg->head_dim       = (int)gguf_meta_u32(gf, "gemma3.attention.key_length", 0);
    cfg->ffn_size       = (int)gguf_meta_u32(gf, "gemma3.feed_forward_length", 0);
    cfg->rope_theta     = gguf_meta_f32(gf, "gemma3.rope.freq_base",      10000.0f);
    cfg->sliding_window = (int)gguf_meta_u32(gf, "gemma3.attention.sliding_window", 0);
    cfg->logit_soft_cap      = gguf_meta_f32(gf, "gemma3.attention.logit_softcapping", 0.0f);
    cfg->final_logit_soft_cap = gguf_meta_f32(gf, "gemma3.final_logit_softcapping",   0.0f);
    cfg->rms_norm_eps   = gguf_meta_f32(gf, "gemma3.attention.layer_norm_rms_epsilon", 1e-6f);
    cfg->max_seq_len    = (int)max_ctx;
    cfg->vocab_size     = gguf_vocab_size(gf);

    /* Validate mandatory fields */
    if (cfg->hidden_size == 0 || cfg->n_layers == 0 || cfg->n_heads == 0
        || cfg->head_dim == 0 || cfg->ffn_size == 0) {
        fprintf(stderr, "engine: missing or zero Gemma 3 metadata fields\n");
        fprintf(stderr, "  hidden_size=%d n_layers=%d n_heads=%d head_dim=%d ffn_size=%d\n",
                cfg->hidden_size, cfg->n_layers, cfg->n_heads,
                cfg->head_dim, cfg->ffn_size);
        return -1;
    }
    if (cfg->vocab_size <= 0) {
        /* Fallback: read from tokenizer.ggml.tokens_length */
        cfg->vocab_size = (int)gguf_meta_u32(gf, "tokenizer.ggml.tokens_length", 0);
    }
    if (cfg->vocab_size <= 0) {
        fprintf(stderr, "engine: cannot determine vocabulary size\n");
        return -1;
    }
    return 0;
}

/* ── xandengine_create ────────────────────────────────────────────────────── */

XandEngine *xandengine_create(const char *path, int gpu_id, size_t max_ctx) {
    /* Validate architecture before allocating anything */
    GgufFile *gf = gguf_open(path);
    if (!gf) return NULL;

    const char *arch = gguf_meta_str(gf, "general.architecture");
    if (!arch || (strcmp(arch, "gemma3") != 0 && strcmp(arch, "gemma") != 0)) {
        fprintf(stderr,
            "engine: unsupported architecture '%s' (expected gemma3)\n",
            arch ? arch : "<missing>");
        gguf_close(gf);
        return NULL;
    }

    Gemma3Config cfg;
    if (extract_config(gf, max_ctx > 0 ? max_ctx : 8192, &cfg) != 0) {
        gguf_close(gf);
        return NULL;
    }

    fprintf(stderr,
        "[xandengine] Gemma 3 config: hidden=%d layers=%d heads=%d kv_heads=%d "
        "head_dim=%d ffn=%d vocab=%d max_ctx=%d rope_theta=%.0f\n",
        cfg.hidden_size, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
        cfg.head_dim, cfg.ffn_size, cfg.vocab_size, cfg.max_seq_len,
        (double)cfg.rope_theta);

    Gemma3Model *model = gemma3_init(gf, &cfg, gpu_id);
    if (!model) {
        fprintf(stderr, "engine: gemma3_init failed\n");
        gguf_close(gf);
        return NULL;
    }

    XandEngine *e = (XandEngine *)calloc(1, sizeof(XandEngine));
    if (!e) {
        gemma3_free(model);
        gguf_close(gf);
        return NULL;
    }
    e->gf         = gf;
    e->model      = model;
    e->vocab_size = cfg.vocab_size;
    e->gpu_id     = gpu_id;

    fprintf(stderr, "[xandengine] model loaded successfully on GPU %d\n", gpu_id);
    return e;
}

/* ── xandengine_destroy ───────────────────────────────────────────────────── */

void xandengine_destroy(XandEngine *e) {
    if (!e) return;
    gemma3_free(e->model);
    gguf_close(e->gf);
    free(e);
}

/* ── xandengine_reset_kv ──────────────────────────────────────────────────── */

void xandengine_reset_kv(XandEngine *e) {
    if (e && e->model) gemma3_reset_kv(e->model);
}

/* ── xandengine_forward ───────────────────────────────────────────────────── */

int xandengine_forward(XandEngine *e,
                       const int32_t *tokens, int n_tokens,
                       float *logits_out, int vocab_size)
{
    if (!e || !tokens || n_tokens <= 0 || !logits_out) return -1;
    if (vocab_size != e->vocab_size) {
        fprintf(stderr, "engine: vocab_size mismatch: caller=%d engine=%d\n",
                vocab_size, e->vocab_size);
        return -1;
    }

    Gemma3Model *m = e->model;
    int pos = m->cache_pos;

    /* Run each token sequentially, accumulating KV cache */
    float *d_logits = NULL;
    for (int i = 0; i < n_tokens; i++) {
        d_logits = gemma3_forward(m, (int)tokens[i], pos + i);
        if (!d_logits) return -1;
    }

    /* Copy last token's logits from GPU to caller's CPU buffer */
    /* gemma3_forward already synchronised the stream */
    cudaError_t err = cudaMemcpy(logits_out, d_logits,
                                 (size_t)vocab_size * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "engine: cudaMemcpy logits failed: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

/* ── Accessors ────────────────────────────────────────────────────────────── */

int xandengine_vocab_size(const XandEngine *e) {
    return e ? e->vocab_size : 0;
}

const char *xandengine_arch(const XandEngine *e) {
    (void)e;
    return "gemma3";
}

const char *xandengine_chat_format(const XandEngine *e) {
    (void)e;
    return "gemma";
}
