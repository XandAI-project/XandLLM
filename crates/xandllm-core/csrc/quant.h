/*
 * quant.h — Quantization block structs and CUDA matvec kernels
 *
 * Struct layouts match the GGUF / GGML binary format exactly.
 * All kernels compute  y = W * x  where:
 *   W  is (n_rows × n_cols) stored row-major on the GPU
 *   x  is a float vector (n_cols)  on the GPU
 *   y  is a float vector (n_rows)  on the GPU
 */
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#  include <cuda_fp16.h>
extern "C" {
#endif

/* ── Block sizes (must match gguf.h) ────────────────────────────────────── */
#define QK4_0  32     /* elements per Q4_0 block */
#define QK8_0  32     /* elements per Q8_0 block */
#define QK_K   256    /* elements per K-quant superblock */

/* ── Q4_0 block: 32 weights packed as nibbles + one f16 scale (18 bytes) ── */
typedef struct {
#ifdef __cplusplus
    __half d;
#else
    uint16_t d;       /* f16 scale */
#endif
    uint8_t  qs[16];  /* 32 weights packed as 4-bit nibbles */
} block_q4_0;

/* ── Q8_0 block: 32 int8 weights + one f16 scale (34 bytes) ─────────────── */
typedef struct {
#ifdef __cplusplus
    __half d;
#else
    uint16_t d;       /* f16 scale */
#endif
    int8_t   qs[32];
} block_q8_0;

/*
 * Q4_K superblock: 256 weights, 144 bytes.
 * Layout (GGML spec):
 *   d     : f16   — super-scale
 *   dmin  : f16   — super-min
 *   scales: uint8[12] — packed 6-bit sub-block scales (8 sub-blocks × 12 bits)
 *   qs    : uint8[128] — 256 weights packed as 4-bit nibbles
 */
typedef struct {
#ifdef __cplusplus
    __half d, dmin;
#else
    uint16_t d, dmin;
#endif
    uint8_t  scales[12];
    uint8_t  qs[128];
} block_q4_K;

/* ── GGUF dtype codes (duplicated from gguf.h for C++ headers) ──────────── */
#ifndef GGUF_TYPE_F32
#  define GGUF_TYPE_F32    0
#  define GGUF_TYPE_F16    1
#  define GGUF_TYPE_Q4_0   2
#  define GGUF_TYPE_Q8_0   8
#  define GGUF_TYPE_Q4_K  12
#endif

/* ── CUDA kernel declarations ─────────────────────────────────────────────
 *
 * All kernels expect pointers already on the GPU device.
 * stream may be 0 (default stream).
 */

/*
 * Q4_K matvec: W is (n_rows × n_cols) in block_q4_K layout.
 * n_cols must be a multiple of QK_K (256).
 */
void q4k_matvec_cuda(const block_q4_K *W,
                     const float      *x,
                     float            *y,
                     int n_rows, int n_cols,
                     cudaStream_t stream);

/*
 * Q4_0 matvec: W is (n_rows × n_cols) in block_q4_0 layout.
 * n_cols must be a multiple of QK4_0 (32).
 */
void q4_0_matvec_cuda(const block_q4_0 *W,
                      const float      *x,
                      float            *y,
                      int n_rows, int n_cols,
                      cudaStream_t stream);

/*
 * Q8_0 matvec: W is (n_rows × n_cols) in block_q8_0 layout.
 * n_cols must be a multiple of QK8_0 (32).
 */
void q8_0_matvec_cuda(const block_q8_0 *W,
                      const float      *x,
                      float            *y,
                      int n_rows, int n_cols,
                      cudaStream_t stream);

/*
 * F16 matvec: W is (n_rows × n_cols) f16 values.
 */
void f16_matvec_cuda(const void  *W,   /* __half* */
                     const float *x,
                     float       *y,
                     int n_rows, int n_cols,
                     cudaStream_t stream);

/*
 * Dispatcher — selects the right kernel based on GGUF dtype.
 * dtype is one of GGUF_TYPE_*.
 */
void gguf_matvec(uint32_t      dtype,
                 const void   *W,
                 const float  *x,
                 float        *y,
                 int n_rows, int n_cols,
                 cudaStream_t stream);

#ifdef __cplusplus
}
#endif
