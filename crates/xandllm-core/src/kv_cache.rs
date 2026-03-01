//! Pre-allocated, fixed-size KV cache — inspired by llama.cpp's ring-buffer
//! approach.
//!
//! ## Design goals (from llama.cpp / vLLM)
//!
//! | Goal | How |
//! |------|-----|
//! | No growing allocations | Pre-allocate `[batch, heads, max_len, head_dim]` once per layer |
//! | Constant-size per step | `Tensor::slice_scatter` writes new K/V at a fixed offset |
//! | Bounded memory | Sliding window: evict oldest tokens when buffer is full |
//! | GPU allocator friendly | All per-layer tensors are the same shape → allocator can pool them |
//!
//! ## Memory model
//!
//! ```text
//! [batch, heads, <──────── max_len ────────>, head_dim]
//!                [■■■■■■■■■■■■■■░░░░░░░░░░░░]
//!                 ^──── filled ────^
//! ```
//!
//! After `filled == max_len` the buffer is full.  On the next update the
//! oldest `new_len` tokens are evicted (shifted left with `narrow + cat`) and
//! the new tokens are written at `max_len - new_len`.  This gives a sliding
//! context window, exactly as in `llama.cpp -c` context rolling.

use candle_core::Tensor;

use crate::error::CoreResult;

// ── Cache entry ───────────────────────────────────────────────────────────────

/// Single-layer key-value pair.  Both tensors have shape
/// `[batch, heads, max_len, head_dim]` and are allocated exactly once.
#[derive(Debug, Clone)]
pub struct KvEntry {
    pub key: Tensor,
    pub value: Tensor,
}

// ── KvCache ───────────────────────────────────────────────────────────────────

/// Pre-allocated, rolling KV cache for a full transformer stack.
///
/// * One `KvEntry` per layer, lazily allocated on first [`update`].
/// * All per-layer tensors share the same fixed shape — no reallocation.
/// * Exceeding `max_len` triggers a sliding-window eviction.
///
/// [`update`]: KvCache::update
#[derive(Debug)]
pub struct KvCache {
    /// Per-layer pre-allocated buffer.  `None` until the layer is first written.
    entries: Vec<Option<KvEntry>>,
    /// Number of token positions currently filled (same for every layer).
    current_len: usize,
    max_len: usize,
}

impl KvCache {
    /// Create an empty cache for `num_layers` transformer layers and a context
    /// window of `max_len` tokens.
    pub fn new(num_layers: usize, max_len: usize) -> Self {
        Self {
            entries: vec![None; num_layers],
            current_len: 0,
            max_len,
        }
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// Write new key/value tensors for `layer_idx` and return the full
    /// accumulated `(key, value)` pair for attention computation.
    ///
    /// `key` and `value` have shape `[batch, heads, new_len, head_dim]`.
    ///
    /// ### First call for a layer
    /// Pre-allocates a zero-filled `[batch, heads, max_len, head_dim]` buffer
    /// and writes the first `new_len` tokens via `Tensor::slice_scatter`.
    ///
    /// ### Subsequent calls
    /// Uses `slice_scatter` to write at the current fill position without
    /// creating any larger tensor — always operates on the same `max_len` shape.
    ///
    /// ### Overflow (sliding window)
    /// When `current_len + new_len > max_len` the oldest tokens are evicted:
    /// the filled portion is shifted left by `evict` positions so the new
    /// tokens fit at the end.  This matches llama.cpp's context-rolling strategy.
    pub fn update(
        &mut self,
        layer_idx: usize,
        key: Tensor,
        value: Tensor,
    ) -> CoreResult<(Tensor, Tensor)> {
        let new_len = key.dim(2)?;

        // --- Handle overflow: evict oldest tokens (sliding window) ------------
        if self.current_len + new_len > self.max_len {
            let evict = (self.current_len + new_len) - self.max_len;
            self.evict_oldest(evict)?;
            // current_len has been updated by evict_oldest
        }

        let write_pos = self.current_len;

        let (k_buf, v_buf) = match self.entries[layer_idx].take() {
            None => {
                // First write: pre-allocate the full fixed-size buffer.
                // Using `from_vec` (zeros) keeps the allocation on the correct device.
                let (batch, heads, _, head_dim) = key.dims4()?;
                let dtype = key.dtype();
                let device = key.device();
                let k_zeros = Tensor::zeros((batch, heads, self.max_len, head_dim), dtype, device)?;
                let v_zeros = Tensor::zeros((batch, heads, self.max_len, head_dim), dtype, device)?;
                (k_zeros, v_zeros)
            }
            Some(prev) => (prev.key, prev.value),
        };

        // `slice_scatter` writes `key`/`value` at `write_pos` along dim 2.
        // The returned tensor has the SAME shape as `k_buf` (max_len wide) —
        // no new allocation size growth, just a fixed-size write.
        let k_out = k_buf.slice_scatter(&key, 2, write_pos)?;
        let v_out = v_buf.slice_scatter(&value, 2, write_pos)?;

        // Store the updated fixed-size buffer.
        self.entries[layer_idx] = Some(KvEntry {
            key: k_out.clone(),
            value: v_out.clone(),
        });

        // Advance the fill pointer so subsequent writes land in the right place.
        let filled = write_pos + new_len;
        self.current_len = filled;

        // Return only the filled portion for attention.
        let k_view = k_out.narrow(2, 0, filled)?;
        let v_view = v_out.narrow(2, 0, filled)?;

        Ok((k_view, v_view))
    }

    // ── Eviction (sliding window) ─────────────────────────────────────────────

    /// Drop the `evict` oldest tokens from every allocated layer.
    ///
    /// Uses `narrow` (zero-copy view) + `slice_scatter` to shift the remaining
    /// tokens to the front of the pre-allocated buffer.
    fn evict_oldest(&mut self, evict: usize) -> CoreResult<()> {
        let keep = self.current_len.saturating_sub(evict);

        for entry in self.entries.iter_mut().flatten() {
            // View of tokens to keep: [batch, heads, keep, head_dim]
            let k_keep = entry.key.narrow(2, evict, keep)?;
            let v_keep = entry.value.narrow(2, evict, keep)?;

            // Write the surviving tokens back to the front of the buffer.
            // This uses slice_scatter at offset 0, so no size change.
            entry.key = entry.key.slice_scatter(&k_keep, 2, 0)?;
            entry.value = entry.value.slice_scatter(&v_keep, 2, 0)?;
        }

        self.current_len = keep;
        Ok(())
    }

    // ── Advance ───────────────────────────────────────────────────────────────

    /// Manually increment the fill pointer by `n` tokens.
    ///
    /// Normally `update` maintains this automatically.  Call `advance` only
    /// when you write to layers independently and track position yourself.
    pub fn advance(&mut self, n: usize) {
        self.current_len = (self.current_len + n).min(self.max_len);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Number of token positions currently in the cache.
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Maximum context length.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is completely full (next write will evict).
    pub fn is_full(&self) -> bool {
        self.current_len >= self.max_len
    }

    // ── Reset ─────────────────────────────────────────────────────────────────

    /// Clear all cached K/V data.  The pre-allocated buffers are kept so the
    /// next write sequence avoids the initial allocation cost.
    ///
    /// Unlike dropping the whole cache, this preserves the GPU memory pages
    /// allocated for each layer, allowing the CUDA allocator to reuse them
    /// immediately for the next generation.
    pub fn clear(&mut self) {
        // Zero out the buffers in-place using slice_scatter with zeros.
        // This is cheaper than freeing + reallocating because CUDA memory
        // pages stay mapped and warm in the allocator pool.
        for entry in self.entries.iter_mut() {
            if let Some(e) = entry {
                if let (Ok(k_shape), Ok(v_shape)) = (e.key.shape().clone().into_dims4(), e.value.shape().clone().into_dims4()) {
                    let dev = e.key.device();
                    let dt  = e.key.dtype();
                    if let (Ok(kz), Ok(vz)) = (
                        Tensor::zeros(k_shape, dt, dev),
                        Tensor::zeros(v_shape, dt, dev),
                    ) {
                        e.key   = kz;
                        e.value = vz;
                    }
                }
            }
        }
        self.current_len = 0;
    }

    /// Like `clear` but releases the GPU memory completely.
    /// Use this when you want to free VRAM (e.g. between model phases).
    pub fn free(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = None;
        }
        self.current_len = 0;
    }
}

// ── Shape helper ──────────────────────────────────────────────────────────────

trait IntoDims4 {
    fn into_dims4(self) -> CoreResult<(usize, usize, usize, usize)>;
}

impl IntoDims4 for candle_core::Shape {
    fn into_dims4(self) -> CoreResult<(usize, usize, usize, usize)> {
        let dims = self.dims();
        if dims.len() == 4 {
            Ok((dims[0], dims[1], dims[2], dims[3]))
        } else {
            Err(crate::error::CoreError::Config {
                field: "kv_cache shape".to_string(),
                reason: format!("Expected 4-D tensor, got {}D", dims.len()),
            })
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn zeros(seq: usize) -> Tensor {
        Tensor::zeros((1usize, 2usize, seq, 4usize), DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn initial_state() {
        let cache = KvCache::new(4, 128);
        assert_eq!(cache.num_layers(), 4);
        assert_eq!(cache.max_len(), 128);
        assert_eq!(cache.current_len(), 0);
        assert!(!cache.is_full());
    }

    #[test]
    fn first_update_allocates_and_returns_filled_view() {
        let mut cache = KvCache::new(2, 64);
        let (k, v) = cache.update(0, zeros(8), zeros(8)).unwrap();
        // View should be the filled portion [batch=1, heads=2, seq=8, head_dim=4]
        assert_eq!(k.dims(), &[1, 2, 8, 4]);
        assert_eq!(v.dims(), &[1, 2, 8, 4]);
    }

    #[test]
    fn subsequent_updates_grow_view() {
        let mut cache = KvCache::new(1, 64);
        cache.update(0, zeros(4), zeros(4)).unwrap();
        let (k, _) = cache.update(0, zeros(1), zeros(1)).unwrap();
        assert_eq!(k.dims()[2], 5); // 4 + 1
    }

    #[test]
    fn advance_accumulates() {
        let mut cache = KvCache::new(4, 1024);
        cache.advance(10);
        assert_eq!(cache.current_len(), 10);
        cache.advance(5);
        assert_eq!(cache.current_len(), 15);
    }

    #[test]
    fn overflow_triggers_sliding_window() {
        let mut cache = KvCache::new(1, 8); // tiny window
        // Fill 8 tokens
        cache.update(0, zeros(4), zeros(4)).unwrap();
        cache.update(0, zeros(4), zeros(4)).unwrap();
        assert_eq!(cache.current_len(), 8);
        // One more token should evict 1, keep 7, write 1 → still 8
        let (k, _) = cache.update(0, zeros(1), zeros(1)).unwrap();
        assert_eq!(cache.current_len(), 8);
        assert_eq!(k.dims()[2], 8);
    }

    #[test]
    fn clear_resets_length_keeps_buffers() {
        let mut cache = KvCache::new(2, 64);
        cache.update(0, zeros(4), zeros(4)).unwrap();
        cache.clear();
        assert_eq!(cache.current_len(), 0);
        // After clear, a new update should start at position 0 again
        let (k, _) = cache.update(0, zeros(3), zeros(3)).unwrap();
        assert_eq!(k.dims()[2], 3);
    }

    #[test]
    fn free_releases_buffers() {
        let mut cache = KvCache::new(2, 64);
        cache.update(0, zeros(4), zeros(4)).unwrap();
        cache.free();
        assert_eq!(cache.current_len(), 0);
        // After free, entries are None — next update re-allocates
        let (k, _) = cache.update(0, zeros(4), zeros(4)).unwrap();
        assert_eq!(k.dims()[2], 4);
    }
}
