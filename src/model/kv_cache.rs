//! KV cache abstraction supporting FP32 and FP16 storage.
//!
//! FP16 halves memory usage with negligible quality loss for attention.

use half::f16;
use ndarray::Array4;

/// KV cache data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVDtype {
    F32,
    F16,
}

/// Storage backend for the KV cache
enum KVCacheStorage {
    F32(Array4<f32>),
    F16(Vec<u16>), // f16 stored as raw u16 bits
}

/// KV cache that supports FP32 and FP16 storage.
/// FP16 halves memory usage with negligible quality loss.
pub struct KVCache {
    storage: KVCacheStorage,
    shape: [usize; 4], // [n_layers, n_kv_heads, max_seq_len, head_dim]
    dtype: KVDtype,
}

impl KVCache {
    /// Create a new zeroed KV cache
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        dtype: KVDtype,
    ) -> Self {
        let shape = [n_layers, n_kv_heads, max_seq_len, head_dim];
        let storage = match dtype {
            KVDtype::F32 => {
                KVCacheStorage::F32(Array4::zeros((n_layers, n_kv_heads, max_seq_len, head_dim)))
            }
            KVDtype::F16 => {
                let capacity = n_layers * n_kv_heads * max_seq_len * head_dim;
                KVCacheStorage::F16(vec![0u16; capacity])
            }
        };
        Self {
            storage,
            shape,
            dtype,
        }
    }

    /// Get the shape
    pub fn shape(&self) -> &[usize; 4] {
        &self.shape
    }

    fn linear_idx(&self, layer: usize, head: usize, pos: usize, d: usize) -> usize {
        let [_, n_heads, max_seq, head_dim] = self.shape;
        layer * (n_heads * max_seq * head_dim) + head * (max_seq * head_dim) + pos * head_dim + d
    }

    /// Write a single element
    pub fn set(&mut self, layer: usize, head: usize, pos: usize, d: usize, val: f32) {
        let idx = self.linear_idx(layer, head, pos, d);
        match &mut self.storage {
            KVCacheStorage::F32(arr) => arr[[layer, head, pos, d]] = val,
            KVCacheStorage::F16(vec) => vec[idx] = f16::from_f32(val).to_bits(),
        }
    }

    /// Read a single element
    pub fn get(&self, layer: usize, head: usize, pos: usize, d: usize) -> f32 {
        match &self.storage {
            KVCacheStorage::F32(arr) => arr[[layer, head, pos, d]],
            KVCacheStorage::F16(vec) => {
                let idx = self.linear_idx(layer, head, pos, d);
                f16::from_bits(vec[idx]).to_f32()
            }
        }
    }

    /// Write a full head's K or V state at a position.
    /// Copies from f32 slice, converting to f16 if needed.
    pub fn push_head(&mut self, layer: usize, head: usize, pos: usize, state: &[f32]) {
        let head_dim = self.shape[3];
        assert_eq!(state.len(), head_dim, "state length must match head_dim");
        let base_idx = self.linear_idx(layer, head, pos, 0);

        match &mut self.storage {
            KVCacheStorage::F32(arr) => {
                for d in 0..head_dim {
                    arr[[layer, head, pos, d]] = state[d];
                }
            }
            KVCacheStorage::F16(vec) => {
                for (d, &val) in state.iter().enumerate() {
                    vec[base_idx + d] = f16::from_f32(val).to_bits();
                }
            }
        }
    }

    /// Read a slice of the cache for attention: returns [seq_len, head_dim] as f32.
    /// For FP16, this materializes a temporary f32 buffer.
    pub fn get_slice_f32(&self, layer: usize, head: usize, seq_len: usize) -> Vec<f32> {
        let head_dim = self.shape[3];
        let mut out = Vec::with_capacity(seq_len * head_dim);

        match &self.storage {
            KVCacheStorage::F32(arr) => {
                for pos in 0..seq_len {
                    for d in 0..head_dim {
                        out.push(arr[[layer, head, pos, d]]);
                    }
                }
            }
            KVCacheStorage::F16(vec) => {
                for pos in 0..seq_len {
                    for d in 0..head_dim {
                        let idx = self.linear_idx(layer, head, pos, d);
                        out.push(f16::from_bits(vec[idx]).to_f32());
                    }
                }
            }
        }
        out
    }

    /// Get the KV dtype
    pub fn dtype(&self) -> KVDtype {
        self.dtype
    }

    /// Get the total number of elements (for memory reporting).
    pub fn len(&self) -> usize {
        let [a, b, c, d] = self.shape;
        a * b * c * d
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size in bytes (accounting for F32 vs F16)
    pub fn size_bytes(&self) -> usize {
        let elem_bytes = match self.dtype {
            KVDtype::F32 => 4,
            KVDtype::F16 => 2,
        };
        self.len() * elem_bytes
    }
}

impl std::fmt::Debug for KVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KVCache")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .finish()
    }
}
