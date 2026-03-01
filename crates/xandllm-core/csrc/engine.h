/*
 * engine.h — Public C API for the XandLLM GGUF inference engine
 *
 * This is the only header Rust needs to include (via FFI declarations).
 * All implementation details (GGUF parser, CUDA kernels, model weights) are
 * hidden behind this opaque pointer interface.
 *
 * Thread safety: each XandEngine is single-threaded.  Create one engine per
 * concurrent request if you need parallelism.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine handle. */
typedef struct XandEngine XandEngine;

/*
 * Create an engine from a GGUF file.
 *
 * path     — absolute path to the .gguf file
 * gpu_id   — CUDA device index (0 for the first GPU)
 * max_ctx  — maximum context length (number of tokens the KV cache holds)
 *
 * Returns NULL on failure; the error is printed to stderr.
 * Only Gemma 3 GGUF files are supported.
 */
XandEngine *xandengine_create(const char *path, int gpu_id, size_t max_ctx);

/*
 * Free all resources (GPU memory, GGUF mmap, tokenizer data, etc.).
 * Safe to call with NULL.
 */
void xandengine_destroy(XandEngine *e);

/*
 * Reset the KV cache, allowing a new conversation to start without
 * reloading weights.  After this call the engine behaves as if freshly
 * created.
 */
void xandengine_reset_kv(XandEngine *e);

/*
 * Run one forward pass.
 *
 * tokens      — token IDs (int32, CPU buffer)
 * n_tokens    — number of tokens to process (prefill length or 1 for decode)
 * logits_out  — caller-allocated CPU float array of length vocab_size;
 *               receives the logits for the LAST token after the call.
 * vocab_size  — must equal xandengine_vocab_size(e)
 *
 * Returns 0 on success, non-zero on error (message printed to stderr).
 *
 * Note: for n_tokens > 1 (prefill), the function runs each token through the
 * model sequentially while building the KV cache; only the last token's
 * logits are written to logits_out.
 */
int xandengine_forward(XandEngine *e,
                       const int32_t *tokens, int n_tokens,
                       float         *logits_out, int vocab_size);

/* Return the vocabulary size (number of output logit dimensions). */
int xandengine_vocab_size(const XandEngine *e);

/*
 * Return the architecture name of the loaded model.
 * Currently always "gemma3".
 * The returned pointer is valid for the lifetime of the engine.
 */
const char *xandengine_arch(const XandEngine *e);

/*
 * Return the recommended chat-template format string.
 * Currently always "gemma".
 * The returned pointer is valid for the lifetime of the engine.
 */
const char *xandengine_chat_format(const XandEngine *e);

#ifdef __cplusplus
}
#endif
