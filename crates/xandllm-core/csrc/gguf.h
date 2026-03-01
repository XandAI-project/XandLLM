/*
 * gguf.h — GGUF v1/v2/v3 binary file parser
 *
 * Supports the subset of metadata types needed to load Gemma 3 GGUF models:
 *   uint8, int8, uint16, int16, uint32, int32, float32,
 *   bool, string, uint64, int64, float64, and homogeneous arrays of the above.
 *
 * All multi-byte integers are little-endian (GGUF spec).
 * Tensor data is memory-mapped for zero-copy weight access.
 */
#pragma once
#ifndef GGUF_H
#define GGUF_H
#include <stddef.h>
#include <stdint.h>

/* ── GGUF dtype codes (matches GGML enum) ────────────────────────────────── */
#define GGUF_TYPE_F32    0
#define GGUF_TYPE_F16    1
#define GGUF_TYPE_Q4_0   2
#define GGUF_TYPE_Q4_1   3
#define GGUF_TYPE_Q5_0   6
#define GGUF_TYPE_Q5_1   7
#define GGUF_TYPE_Q8_0   8
#define GGUF_TYPE_Q8_1   9
#define GGUF_TYPE_Q2_K  10
#define GGUF_TYPE_Q3_K  11
#define GGUF_TYPE_Q4_K  12
#define GGUF_TYPE_Q5_K  13
#define GGUF_TYPE_Q6_K  14
#define GGUF_TYPE_Q8_K  15
#define GGUF_TYPE_BF16  30

/* ── Metadata value types ────────────────────────────────────────────────── */
#define GGUF_META_UINT8    0
#define GGUF_META_INT8     1
#define GGUF_META_UINT16   2
#define GGUF_META_INT16    3
#define GGUF_META_UINT32   4
#define GGUF_META_INT32    5
#define GGUF_META_FLOAT32  6
#define GGUF_META_BOOL     7
#define GGUF_META_STRING   8
#define GGUF_META_ARRAY    9
#define GGUF_META_UINT64  10
#define GGUF_META_INT64   11
#define GGUF_META_FLOAT64 12

/* ── Structs ─────────────────────────────────────────────────────────────── */

/* A single metadata key-value pair. */
typedef struct {
    char   *key;          /* null-terminated, heap-allocated */
    uint32_t value_type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint8_t  bool_;
        char    *str;     /* null-terminated, heap-allocated */
        uint64_t u64;
        int64_t  i64;
        double   f64;
        struct {
            uint32_t  elem_type;
            uint64_t  count;
            void     *data;  /* heap-allocated flat array */
        } array;
    } val;
} GgufMetaKV;

/* Info for one tensor: name, shape, dtype, byte offset into tensor data. */
typedef struct {
    char     name[256];
    uint32_t ndims;
    uint64_t shape[4];   /* shape[0] = innermost (columns), C order */
    uint32_t dtype;
    uint64_t offset;     /* bytes from start of tensor data region */
} GgufTensorInfo;

/* Parsed GGUF file handle.
 * The struct is given a tag (GgufFile) so that forward declarations in other
 * headers ("struct GgufFile;") refer to the same type as the typedef.
 */
typedef struct GgufFile {
    /* Metadata */
    GgufMetaKV   *meta;
    size_t        n_meta;

    /* Tensor catalogue */
    GgufTensorInfo *tensors;
    size_t          n_tensors;

    /* Memory-mapped tensor data */
    void          *mapped;        /* mmap base address */
    size_t         mapped_size;   /* total mmap length */
    size_t         data_offset;   /* byte offset of tensor data within file */

    /* Underlying file descriptor (for munmap on close) */
    int            fd;
} GgufFile;

/* ── Public API ──────────────────────────────────────────────────────────── */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Open and parse a GGUF file.
 * Returns NULL on failure (message printed to stderr).
 * The returned handle owns all heap memory; release with gguf_close().
 */
GgufFile *gguf_open(const char *path);

/* Release all resources held by a GgufFile. */
void gguf_close(GgufFile *f);

/*
 * Return a pointer to the raw tensor data for the named tensor.
 * The pointer is into the memory-mapped region — valid as long as GgufFile
 * is open.  Returns NULL if the tensor is not found.
 */
void *gguf_tensor_data(const GgufFile *f, const char *name);

/*
 * Locate a tensor by name and fill *out_info.
 * Returns 0 on success, -1 if not found.
 */
int gguf_find_tensor(const GgufFile *f, const char *name, GgufTensorInfo *out_info);

/* Metadata accessors — return default_val if key is not present. */
uint64_t    gguf_meta_u64(const GgufFile *f, const char *key, uint64_t default_val);
uint32_t    gguf_meta_u32(const GgufFile *f, const char *key, uint32_t default_val);
int32_t     gguf_meta_i32(const GgufFile *f, const char *key, int32_t  default_val);
float       gguf_meta_f32(const GgufFile *f, const char *key, float    default_val);
const char *gguf_meta_str(const GgufFile *f, const char *key); /* NULL if absent */

/*
 * Return the number of tokens in the GGUF vocabulary
 * (key: "tokenizer.ggml.tokens").
 */
int gguf_vocab_size(const GgufFile *f);

/* Return the byte size of one block for a given GGUF dtype. */
size_t gguf_block_size(uint32_t dtype);

/* Return the number of elements per block for a given GGUF dtype. */
int gguf_block_elems(uint32_t dtype);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GGUF_H */
