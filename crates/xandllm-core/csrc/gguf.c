/*
 * gguf.c — GGUF v1/v2/v3 binary parser implementation
 *
 * Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * Uses POSIX mmap on Linux/macOS and CreateFileMapping on Windows.
 */
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <io.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#  include <sys/mman.h>
#endif

/* ── helpers ─────────────────────────────────────────────────────────────── */

static int read_u8(FILE *fp, uint8_t *out) {
    return fread(out, 1, 1, fp) == 1 ? 0 : -1;
}
static int read_u16(FILE *fp, uint16_t *out) {
    return fread(out, 2, 1, fp) == 1 ? 0 : -1;
}
static int read_u32(FILE *fp, uint32_t *out) {
    return fread(out, 4, 1, fp) == 1 ? 0 : -1;
}
static int read_i32(FILE *fp, int32_t *out) {
    return fread(out, 4, 1, fp) == 1 ? 0 : -1;
}
static int read_u64(FILE *fp, uint64_t *out) {
    return fread(out, 8, 1, fp) == 1 ? 0 : -1;
}
static int read_i64(FILE *fp, int64_t *out) {
    return fread(out, 8, 1, fp) == 1 ? 0 : -1;
}
static int read_f32(FILE *fp, float *out) {
    return fread(out, 4, 1, fp) == 1 ? 0 : -1;
}
static int read_f64(FILE *fp, double *out) {
    return fread(out, 8, 1, fp) == 1 ? 0 : -1;
}

/* Read a GGUF length-prefixed string (uint64 length + bytes, no null term). */
static char *read_string(FILE *fp) {
    uint64_t len;
    if (read_u64(fp, &len) != 0) return NULL;
    if (len > (1 << 24)) {
        fprintf(stderr, "gguf: implausibly long string (%llu bytes)\n",
                (unsigned long long)len);
        return NULL;
    }
    char *s = (char *)malloc(len + 1);
    if (!s) return NULL;
    if (fread(s, 1, len, fp) != len) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

/* ── Block size / element count tables ──────────────────────────────────── */

size_t gguf_block_size(uint32_t dtype) {
    switch (dtype) {
        case GGUF_TYPE_F32:  return 4;
        case GGUF_TYPE_F16:  return 2;
        case GGUF_TYPE_BF16: return 2;
        case GGUF_TYPE_Q4_0: return 2 + 16;          /* d(f16) + 16 nibble bytes = 18 */
        case GGUF_TYPE_Q4_1: return 2 + 2 + 16;      /* d + m + 16 = 20 */
        case GGUF_TYPE_Q5_0: return 2 + 4 + 16;      /* 22 */
        case GGUF_TYPE_Q5_1: return 2 + 2 + 4 + 16;  /* 24 */
        case GGUF_TYPE_Q8_0: return 2 + 32;          /* 34 */
        case GGUF_TYPE_Q8_1: return 4 + 4 + 32;      /* 40 */
        case GGUF_TYPE_Q2_K: return 256/16 + 256/4 + 2 + 2; /* 84 */
        case GGUF_TYPE_Q3_K: return 256/8 + 256/4 + 12 + 2; /* 110 */
        case GGUF_TYPE_Q4_K: return 2 + 2 + 12 + 128; /* 144 */
        case GGUF_TYPE_Q5_K: return 2 + 2 + 12 + 4 + 128; /* 176 */
        case GGUF_TYPE_Q6_K: return 256/2 + 256/4 + 256/16 + 2; /* 210 */
        case GGUF_TYPE_Q8_K: return 4 + 256 + 2*16;  /* 292 */
        default:             return 0;
    }
}

int gguf_block_elems(uint32_t dtype) {
    switch (dtype) {
        case GGUF_TYPE_F32:
        case GGUF_TYPE_F16:
        case GGUF_TYPE_BF16: return 1;
        case GGUF_TYPE_Q4_0:
        case GGUF_TYPE_Q4_1:
        case GGUF_TYPE_Q5_0:
        case GGUF_TYPE_Q5_1:
        case GGUF_TYPE_Q8_0:
        case GGUF_TYPE_Q8_1: return 32;
        case GGUF_TYPE_Q2_K:
        case GGUF_TYPE_Q3_K:
        case GGUF_TYPE_Q4_K:
        case GGUF_TYPE_Q5_K:
        case GGUF_TYPE_Q6_K:
        case GGUF_TYPE_Q8_K: return 256;
        default:             return 0;
    }
}

/* ── Metadata value parser ───────────────────────────────────────────────── */

/*
 * Parse a single metadata value of known type into kv->val.
 * For GGUF_META_ARRAY, allocates kv->val.array.data on the heap.
 * Returns 0 on success, -1 on error.
 */
static int parse_meta_value(FILE *fp, GgufMetaKV *kv, uint32_t vtype);

static int parse_scalar(FILE *fp, GgufMetaKV *kv, uint32_t vtype) {
    switch (vtype) {
        case GGUF_META_UINT8:   return read_u8 (fp, &kv->val.u8);
        case GGUF_META_INT8:    return read_u8 (fp, (uint8_t *)&kv->val.i8);
        case GGUF_META_UINT16:  return read_u16(fp, &kv->val.u16);
        case GGUF_META_INT16:   return read_u16(fp, (uint16_t *)&kv->val.i16);
        case GGUF_META_UINT32:  return read_u32(fp, &kv->val.u32);
        case GGUF_META_INT32:   return read_i32(fp, &kv->val.i32);
        case GGUF_META_FLOAT32: return read_f32(fp, &kv->val.f32);
        case GGUF_META_BOOL: {
            uint8_t b; int r = read_u8(fp, &b);
            kv->val.bool_ = b; return r;
        }
        case GGUF_META_STRING:
            kv->val.str = read_string(fp);
            return kv->val.str ? 0 : -1;
        case GGUF_META_UINT64:  return read_u64(fp, &kv->val.u64);
        case GGUF_META_INT64:   return read_i64(fp, (int64_t *)&kv->val.i64);
        case GGUF_META_FLOAT64: return read_f64(fp, &kv->val.f64);
        default:
            fprintf(stderr, "gguf: unsupported meta type %u\n", vtype);
            return -1;
    }
}

/* Returns byte-size of one element of meta-type t (0 if variable / unknown). */
static size_t meta_elem_bytes(uint32_t t) {
    switch (t) {
        case GGUF_META_UINT8:
        case GGUF_META_INT8:
        case GGUF_META_BOOL:   return 1;
        case GGUF_META_UINT16:
        case GGUF_META_INT16:  return 2;
        case GGUF_META_UINT32:
        case GGUF_META_INT32:
        case GGUF_META_FLOAT32:return 4;
        case GGUF_META_UINT64:
        case GGUF_META_INT64:
        case GGUF_META_FLOAT64:return 8;
        default: return 0;
    }
}

static int parse_meta_value(FILE *fp, GgufMetaKV *kv, uint32_t vtype) {
    kv->value_type = vtype;
    if (vtype == GGUF_META_ARRAY) {
        uint32_t elem_type;
        uint64_t count;
        if (read_u32(fp, &elem_type) != 0) return -1;
        if (read_u64(fp, &count)     != 0) return -1;
        kv->val.array.elem_type = elem_type;
        kv->val.array.count     = count;
        kv->val.array.data      = NULL;

        if (count == 0) return 0;

        /* For fixed-size element types, read all at once. */
        size_t esz = meta_elem_bytes(elem_type);
        if (esz > 0) {
            void *buf = malloc(esz * count);
            if (!buf) return -1;
            if (fread(buf, esz, count, fp) != count) { free(buf); return -1; }
            kv->val.array.data = buf;
        } else if (elem_type == GGUF_META_STRING) {
            char **strs = (char **)calloc(count, sizeof(char *));
            if (!strs) return -1;
            for (uint64_t i = 0; i < count; i++) {
                strs[i] = read_string(fp);
                if (!strs[i]) {
                    for (uint64_t j = 0; j < i; j++) free(strs[j]);
                    free(strs);
                    return -1;
                }
            }
            kv->val.array.data = strs;
        } else {
            /* Unknown array element type — skip gracefully. */
            fprintf(stderr, "gguf: skipping array of unsupported type %u\n", elem_type);
            return -1;
        }
        return 0;
    }
    return parse_scalar(fp, kv, vtype);
}

/* ── Main parse routine ──────────────────────────────────────────────────── */

#define GGUF_MAGIC 0x46554747u  /* "GGUF" little-endian */

GgufFile *gguf_open(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "gguf_open: cannot open '%s': %s\n", path, strerror(errno));
        return NULL;
    }

    /* ── Header ── */
    uint32_t magic;
    if (fread(&magic, 4, 1, fp) != 1 || magic != GGUF_MAGIC) {
        fprintf(stderr, "gguf_open: bad magic in '%s'\n", path);
        fclose(fp); return NULL;
    }
    uint32_t version;
    if (read_u32(fp, &version) != 0 || version < 1 || version > 3) {
        fprintf(stderr, "gguf_open: unsupported version %u\n", version);
        fclose(fp); return NULL;
    }

    uint64_t n_tensors, n_meta;
    if (version >= 2) {
        if (read_u64(fp, &n_tensors) != 0) goto fail;
        if (read_u64(fp, &n_meta)    != 0) goto fail;
    } else {
        uint32_t t32, m32;
        if (read_u32(fp, &t32) != 0) goto fail;
        if (read_u32(fp, &m32) != 0) goto fail;
        n_tensors = t32; n_meta = m32;
    }

    GgufFile *gf = (GgufFile *)calloc(1, sizeof(GgufFile));
    if (!gf) goto fail;

    /* ── Metadata ── */
    gf->n_meta = (size_t)n_meta;
    gf->meta   = (GgufMetaKV *)calloc(n_meta, sizeof(GgufMetaKV));
    if (!gf->meta && n_meta > 0) goto fail_gf;

    for (uint64_t i = 0; i < n_meta; i++) {
        GgufMetaKV *kv = &gf->meta[i];
        kv->key = read_string(fp);
        if (!kv->key) {
            fprintf(stderr, "gguf_open: failed reading meta key %llu\n",
                    (unsigned long long)i);
            goto fail_gf;
        }
        uint32_t vtype;
        if (read_u32(fp, &vtype) != 0) goto fail_gf;
        if (parse_meta_value(fp, kv, vtype) != 0) {
            fprintf(stderr, "gguf_open: failed parsing value for key '%s'\n", kv->key);
            goto fail_gf;
        }
    }

    /* ── Tensor info ── */
    gf->n_tensors = (size_t)n_tensors;
    gf->tensors   = (GgufTensorInfo *)calloc(n_tensors, sizeof(GgufTensorInfo));
    if (!gf->tensors && n_tensors > 0) goto fail_gf;

    for (uint64_t i = 0; i < n_tensors; i++) {
        GgufTensorInfo *ti = &gf->tensors[i];

        char *tname = read_string(fp);
        if (!tname) goto fail_gf;
        strncpy(ti->name, tname, sizeof(ti->name) - 1);
        free(tname);

        uint32_t ndims;
        if (version >= 2) {
            if (read_u32(fp, &ndims) != 0) goto fail_gf;
        } else {
            if (read_u32(fp, &ndims) != 0) goto fail_gf;
        }
        ti->ndims = ndims;
        for (uint32_t d = 0; d < ndims && d < 4; d++) {
            if (version >= 2) {
                if (read_u64(fp, &ti->shape[d]) != 0) goto fail_gf;
            } else {
                uint32_t s32;
                if (read_u32(fp, &s32) != 0) goto fail_gf;
                ti->shape[d] = s32;
            }
        }
        if (read_u32(fp, &ti->dtype)  != 0) goto fail_gf;
        if (read_u64(fp, &ti->offset) != 0) goto fail_gf;
    }

    /* ── Align to 32-byte boundary for tensor data ── */
    {
        long pos = ftell(fp);
        if (pos < 0) goto fail_gf;
        long aligned = ((pos + 31) / 32) * 32;
        fseek(fp, aligned, SEEK_SET);
        gf->data_offset = (size_t)aligned;
    }

    fclose(fp);

    /* ── Memory-map the whole file ── */
#ifdef _WIN32
    {
        HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                                   NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) goto fail_gf;

        LARGE_INTEGER fsize;
        GetFileSizeEx(hFile, &fsize);
        gf->mapped_size = (size_t)fsize.QuadPart;

        HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        CloseHandle(hFile);
        if (!hMap) goto fail_gf;

        gf->mapped = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(hMap);
        if (!gf->mapped) goto fail_gf;
    }
#else
    {
        int fd = open(path, O_RDONLY);
        if (fd < 0) goto fail_gf;
        struct stat st;
        if (fstat(fd, &st) < 0) { close(fd); goto fail_gf; }
        gf->mapped_size = (size_t)st.st_size;
        gf->mapped = mmap(NULL, gf->mapped_size, PROT_READ, MAP_SHARED, fd, 0);
        if (gf->mapped == MAP_FAILED) { close(fd); gf->mapped = NULL; goto fail_gf; }
        gf->fd = fd;
    }
#endif

    return gf;

fail:
    fclose(fp);
    return NULL;

fail_gf:
    fclose(fp);
    gguf_close(gf);
    return NULL;
}

/* ── Free helpers ────────────────────────────────────────────────────────── */

static void free_meta_kv(GgufMetaKV *kv) {
    free(kv->key);
    if (kv->value_type == GGUF_META_STRING) {
        free(kv->val.str);
    } else if (kv->value_type == GGUF_META_ARRAY) {
        if (kv->val.array.elem_type == GGUF_META_STRING && kv->val.array.data) {
            char **strs = (char **)kv->val.array.data;
            for (uint64_t i = 0; i < kv->val.array.count; i++) free(strs[i]);
        }
        free(kv->val.array.data);
    }
}

void gguf_close(GgufFile *f) {
    if (!f) return;

    if (f->meta) {
        for (size_t i = 0; i < f->n_meta; i++) free_meta_kv(&f->meta[i]);
        free(f->meta);
    }
    free(f->tensors);

    if (f->mapped) {
#ifdef _WIN32
        UnmapViewOfFile(f->mapped);
#else
        munmap(f->mapped, f->mapped_size);
        if (f->fd > 0) close(f->fd);
#endif
    }
    free(f);
}

/* ── Accessors ───────────────────────────────────────────────────────────── */

void *gguf_tensor_data(const GgufFile *f, const char *name) {
    for (size_t i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            return (char *)f->mapped + f->data_offset + f->tensors[i].offset;
        }
    }
    return NULL;
}

int gguf_find_tensor(const GgufFile *f, const char *name, GgufTensorInfo *out_info) {
    for (size_t i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            if (out_info) *out_info = f->tensors[i];
            return 0;
        }
    }
    return -1;
}

/* Find a metadata entry by key; returns NULL if absent. */
static const GgufMetaKV *find_meta(const GgufFile *f, const char *key) {
    for (size_t i = 0; i < f->n_meta; i++) {
        if (f->meta[i].key && strcmp(f->meta[i].key, key) == 0)
            return &f->meta[i];
    }
    return NULL;
}

uint64_t gguf_meta_u64(const GgufFile *f, const char *key, uint64_t def) {
    const GgufMetaKV *kv = find_meta(f, key);
    if (!kv) return def;
    switch (kv->value_type) {
        case GGUF_META_UINT8:  return kv->val.u8;
        case GGUF_META_UINT16: return kv->val.u16;
        case GGUF_META_UINT32: return kv->val.u32;
        case GGUF_META_UINT64: return kv->val.u64;
        case GGUF_META_INT32:  return (uint64_t)kv->val.i32;
        case GGUF_META_INT64:  return (uint64_t)kv->val.i64;
        default: return def;
    }
}

uint32_t gguf_meta_u32(const GgufFile *f, const char *key, uint32_t def) {
    return (uint32_t)gguf_meta_u64(f, key, def);
}

int32_t gguf_meta_i32(const GgufFile *f, const char *key, int32_t def) {
    const GgufMetaKV *kv = find_meta(f, key);
    if (!kv) return def;
    switch (kv->value_type) {
        case GGUF_META_INT8:   return kv->val.i8;
        case GGUF_META_INT16:  return kv->val.i16;
        case GGUF_META_INT32:  return kv->val.i32;
        case GGUF_META_INT64:  return (int32_t)kv->val.i64;
        case GGUF_META_UINT32: return (int32_t)kv->val.u32;
        default: return def;
    }
}

float gguf_meta_f32(const GgufFile *f, const char *key, float def) {
    const GgufMetaKV *kv = find_meta(f, key);
    if (!kv) return def;
    switch (kv->value_type) {
        case GGUF_META_FLOAT32: return kv->val.f32;
        case GGUF_META_FLOAT64: return (float)kv->val.f64;
        case GGUF_META_UINT32:  return (float)kv->val.u32;
        case GGUF_META_INT32:   return (float)kv->val.i32;
        default: return def;
    }
}

const char *gguf_meta_str(const GgufFile *f, const char *key) {
    const GgufMetaKV *kv = find_meta(f, key);
    if (!kv || kv->value_type != GGUF_META_STRING) return NULL;
    return kv->val.str;
}

int gguf_vocab_size(const GgufFile *f) {
    const GgufMetaKV *kv = find_meta(f, "tokenizer.ggml.tokens");
    if (!kv || kv->value_type != GGUF_META_ARRAY) return 0;
    return (int)kv->val.array.count;
}
