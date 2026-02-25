use candle_core::Tensor;

use crate::error::CoreResult;

/// Per-layer key-value cache entry.
#[derive(Debug, Clone)]
pub struct KvEntry {
    pub key: Tensor,
    pub value: Tensor,
}

/// KV-cache for the full transformer stack.
///
/// Stores one `KvEntry` per layer. Entries grow as tokens are appended
/// during autoregressive generation.
#[derive(Debug)]
pub struct KvCache {
    entries: Vec<Option<KvEntry>>,
    num_layers: usize,
    /// Number of tokens currently stored in the cache.
    current_len: usize,
    max_len: usize,
}

impl KvCache {
    /// Create a new, empty KV cache for `num_layers` transformer layers.
    pub fn new(num_layers: usize, max_len: usize) -> Self {
        Self {
            entries: vec![None; num_layers],
            num_layers,
            current_len: 0,
            max_len,
        }
    }

    /// Update the cache for layer `layer_idx` with new key/value tensors.
    ///
    /// If a previous entry exists, the new tensors are concatenated along
    /// the sequence dimension (dim=2 for `[batch, heads, seq, head_dim]`).
    pub fn update(&mut self, layer_idx: usize, key: Tensor, value: Tensor) -> CoreResult<(Tensor, Tensor)> {
        let (k, v) = match self.entries[layer_idx].take() {
            Some(prev) => {
                let k = Tensor::cat(&[&prev.key, &key], 2)?;
                let v = Tensor::cat(&[&prev.value, &value], 2)?;
                (k, v)
            }
            None => (key, value),
        };

        self.entries[layer_idx] = Some(KvEntry {
            key: k.clone(),
            value: v.clone(),
        });

        Ok((k, v))
    }

    /// Returns the current sequence length stored in the cache.
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Increment the stored sequence counter by `n`.
    pub fn advance(&mut self, n: usize) {
        self.current_len += n;
    }

    /// Reset the cache (e.g. for a new request).
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = None;
        }
        self.current_len = 0;
    }

    /// Maximum sequence length supported.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Number of transformer layers this cache covers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_initial_state() {
        let cache = KvCache::new(12, 4096);
        assert_eq!(cache.num_layers(), 12);
        assert_eq!(cache.max_len(), 4096);
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_advance_accumulates() {
        let mut cache = KvCache::new(4, 1024);
        cache.advance(10);
        assert_eq!(cache.current_len(), 10);
        cache.advance(5);
        assert_eq!(cache.current_len(), 15);
    }

    #[test]
    fn test_clear_resets_length() {
        let mut cache = KvCache::new(4, 1024);
        cache.advance(100);
        cache.clear();
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_update_single_layer_returns_tensors() {
        let mut cache = KvCache::new(2, 128);
        let dev = Device::Cpu;
        // Shape [batch=1, heads=2, seq=3, head_dim=4]
        let k = Tensor::zeros((1_usize, 2_usize, 3_usize, 4_usize), candle_core::DType::F32, &dev).unwrap();
        let v = Tensor::zeros((1_usize, 2_usize, 3_usize, 4_usize), candle_core::DType::F32, &dev).unwrap();
        let (k_out, v_out) = cache.update(0, k, v).unwrap();
        // Sequence dim should be 3 (no prior cache)
        assert_eq!(k_out.dims()[2], 3);
        assert_eq!(v_out.dims()[2], 3);
    }

    #[test]
    fn test_update_concatenates_on_second_call() {
        let mut cache = KvCache::new(1, 256);
        let dev = Device::Cpu;
        let make = |seq: usize| {
            Tensor::zeros((1_usize, 1_usize, seq, 8_usize), candle_core::DType::F32, &dev).unwrap()
        };
        cache.update(0, make(4), make(4)).unwrap();
        let (k2, _) = cache.update(0, make(1), make(1)).unwrap();
        // After two updates with seq=4 then seq=1, concatenated seq must be 5
        assert_eq!(k2.dims()[2], 5);
    }

    #[test]
    fn test_clear_removes_entries() {
        let mut cache = KvCache::new(2, 128);
        let dev = Device::Cpu;
        let t = Tensor::zeros((1_usize, 1_usize, 2_usize, 4_usize), candle_core::DType::F32, &dev).unwrap();
        cache.update(0, t.clone(), t).unwrap();
        cache.clear();
        // After clear, updating layer 0 should give seq=2 (no concat with old data)
        let t2 = Tensor::zeros((1_usize, 1_usize, 2_usize, 4_usize), candle_core::DType::F32, &dev).unwrap();
        let (k, _) = cache.update(0, t2.clone(), t2).unwrap();
        assert_eq!(k.dims()[2], 2);
    }
}
