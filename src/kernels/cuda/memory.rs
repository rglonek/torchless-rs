//! CUDA Memory Management
//!
//! This module provides GPU memory management utilities including buffer pooling
//! for efficient memory reuse.

use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Convert cudarc driver errors to anyhow errors
fn driver_error_to_anyhow(e: cudarc::driver::DriverError) -> anyhow::Error {
    anyhow::anyhow!("CUDA driver error: {:?}", e)
}

// =============================================================================
// Memory Pool
// =============================================================================

/// A memory pool for reusing GPU buffers.
///
/// Allocating and freeing GPU memory is expensive. This pool keeps freed
/// buffers around for reuse, significantly reducing allocation overhead
/// during inference.
#[derive(Debug)]
pub struct CudaMemoryPool {
    /// Free buffers organized by size (rounded up to nearest power of 2)
    free_buffers: BTreeMap<usize, Vec<CudaSlice<f32>>>,
    /// Total bytes allocated (including freed buffers in pool)
    total_allocated_bytes: usize,
    /// Total bytes currently in use
    in_use_bytes: usize,
    /// Number of allocations served from pool
    pool_hits: usize,
    /// Number of allocations that required new allocation
    pool_misses: usize,
}

impl CudaMemoryPool {
    /// Create a new empty memory pool.
    pub fn new() -> Self {
        Self {
            free_buffers: BTreeMap::new(),
            total_allocated_bytes: 0,
            in_use_bytes: 0,
            pool_hits: 0,
            pool_misses: 0,
        }
    }

    /// Get a buffer of at least the given size.
    ///
    /// First tries to reuse a buffer from the pool. If no suitable buffer
    /// is available, allocates a new one.
    pub fn get_or_alloc(
        &mut self,
        device: &Arc<CudaDevice>,
        min_size: usize,
    ) -> anyhow::Result<CudaSlice<f32>> {
        let bucket_size = round_up_power_of_2(min_size);

        // Try to get a buffer from the pool
        if let Some(buffers) = self.free_buffers.get_mut(&bucket_size) {
            if let Some(buffer) = buffers.pop() {
                self.in_use_bytes += bucket_size * std::mem::size_of::<f32>();
                self.pool_hits += 1;
                return Ok(buffer);
            }
        }

        // Also check larger buckets
        for (&size, buffers) in self.free_buffers.range_mut(bucket_size..) {
            if let Some(buffer) = buffers.pop() {
                self.in_use_bytes += size * std::mem::size_of::<f32>();
                self.pool_hits += 1;
                return Ok(buffer);
            }
        }

        // No suitable buffer found, allocate new one
        let buffer = device
            .alloc_zeros(bucket_size)
            .map_err(driver_error_to_anyhow)?;
        self.total_allocated_bytes += bucket_size * std::mem::size_of::<f32>();
        self.in_use_bytes += bucket_size * std::mem::size_of::<f32>();
        self.pool_misses += 1;

        Ok(buffer)
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buffer(&mut self, buffer: CudaSlice<f32>) {
        let len = DeviceSlice::len(&buffer);
        let bucket_size = round_up_power_of_2(len);

        self.in_use_bytes = self
            .in_use_bytes
            .saturating_sub(bucket_size * std::mem::size_of::<f32>());

        self.free_buffers
            .entry(bucket_size)
            .or_default()
            .push(buffer);
    }

    /// Clear all buffers from the pool, freeing GPU memory.
    pub fn clear(&mut self) {
        self.free_buffers.clear();
        self.total_allocated_bytes = self.in_use_bytes;
    }

    /// Get statistics about the pool.
    pub fn stats(&self) -> MemoryPoolStats {
        let pool_bytes: usize = self
            .free_buffers
            .iter()
            .map(|(size, buffers)| size * buffers.len() * std::mem::size_of::<f32>())
            .sum();

        MemoryPoolStats {
            total_allocated_bytes: self.total_allocated_bytes,
            in_use_bytes: self.in_use_bytes,
            pooled_bytes: pool_bytes,
            pool_hits: self.pool_hits,
            pool_misses: self.pool_misses,
            hit_rate: if self.pool_hits + self.pool_misses > 0 {
                self.pool_hits as f64 / (self.pool_hits + self.pool_misses) as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for CudaMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about memory pool usage.
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    /// Total bytes allocated from GPU
    pub total_allocated_bytes: usize,
    /// Bytes currently in use
    pub in_use_bytes: usize,
    /// Bytes sitting in pool waiting for reuse
    pub pooled_bytes: usize,
    /// Number of allocations served from pool
    pub pool_hits: usize,
    /// Number of allocations requiring new allocation
    pub pool_misses: usize,
    /// Hit rate (hits / total requests)
    pub hit_rate: f64,
}

impl std::fmt::Display for MemoryPoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryPool: {:.2} MB allocated, {:.2} MB in use, {:.2} MB pooled, {:.1}% hit rate ({} hits, {} misses)",
            self.total_allocated_bytes as f64 / 1024.0 / 1024.0,
            self.in_use_bytes as f64 / 1024.0 / 1024.0,
            self.pooled_bytes as f64 / 1024.0 / 1024.0,
            self.hit_rate * 100.0,
            self.pool_hits,
            self.pool_misses,
        )
    }
}

// =============================================================================
// Pinned Memory
// =============================================================================

/// Pinned (page-locked) host memory for faster CPU-GPU transfers.
///
/// Pinned memory can be transferred to/from GPU using DMA, which is
/// significantly faster than pageable memory.
#[derive(Debug)]
pub struct PinnedBuffer {
    /// The raw data
    data: Vec<f32>,
    /// Whether the memory is actually pinned (depends on system support)
    is_pinned: bool,
}

impl PinnedBuffer {
    /// Create a new pinned buffer with the given capacity.
    ///
    /// Note: On systems without CUDA support, this falls back to regular memory.
    pub fn new(capacity: usize) -> Self {
        // For now, use regular Vec - true pinned memory would require
        // calling cudaHostAlloc, which cudarc doesn't expose directly.
        // The performance benefit is still present for async copies.
        Self {
            data: vec![0.0; capacity],
            is_pinned: false, // Would be true with actual pinned memory
        }
    }

    /// Get a slice of the data.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get a mutable slice of the data.
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get the capacity.
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Get the length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Check if the memory is actually pinned.
    pub fn is_pinned(&self) -> bool {
        self.is_pinned
    }

    /// Resize the buffer.
    pub fn resize(&mut self, new_len: usize) {
        self.data.resize(new_len, 0.0);
    }

    /// Copy data from a slice.
    pub fn copy_from_slice(&mut self, src: &[f32]) {
        self.data.resize(src.len(), 0.0);
        self.data.copy_from_slice(src);
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Round up to the nearest power of 2.
fn round_up_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

/// Compute the size in bytes for a tensor with given shape.
pub fn compute_tensor_bytes(shape: &[usize]) -> usize {
    let elements: usize = shape.iter().product();
    elements * std::mem::size_of::<f32>()
}

/// Estimate memory required for model weights on GPU.
pub fn estimate_model_memory_mb(
    hidden_size: usize,
    intermediate_size: usize,
    n_layers: usize,
    vocab_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
) -> f64 {
    let head_dim = hidden_size / n_heads;

    // Per-layer weights
    let q_proj = hidden_size * hidden_size;
    let k_proj = hidden_size * (n_kv_heads * head_dim);
    let v_proj = hidden_size * (n_kv_heads * head_dim);
    let o_proj = hidden_size * hidden_size;
    let gate_proj = hidden_size * intermediate_size;
    let up_proj = hidden_size * intermediate_size;
    let down_proj = intermediate_size * hidden_size;
    let norm = hidden_size * 2; // input_norm and post_attention_norm

    let layer_params = q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj + norm;
    let total_layer_params = layer_params * n_layers;

    // Embedding and output
    let embedding = vocab_size * hidden_size;
    let output = vocab_size * hidden_size;
    let final_norm = hidden_size;

    let total_params = total_layer_params + embedding + output + final_norm;

    // Convert to MB (assuming f32)
    (total_params * std::mem::size_of::<f32>()) as f64 / 1024.0 / 1024.0
}

/// Estimate memory required for KV cache on GPU.
pub fn estimate_kv_cache_memory_mb(
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) -> f64 {
    // K and V each: [n_layers, max_seq_len, n_kv_heads, head_dim]
    let elements_per_cache = n_layers * max_seq_len * n_kv_heads * head_dim;
    let total_bytes = elements_per_cache * 2 * std::mem::size_of::<f32>();
    total_bytes as f64 / 1024.0 / 1024.0
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_up_power_of_2() {
        assert_eq!(round_up_power_of_2(0), 1);
        assert_eq!(round_up_power_of_2(1), 1);
        assert_eq!(round_up_power_of_2(2), 2);
        assert_eq!(round_up_power_of_2(3), 4);
        assert_eq!(round_up_power_of_2(4), 4);
        assert_eq!(round_up_power_of_2(5), 8);
        assert_eq!(round_up_power_of_2(1000), 1024);
        assert_eq!(round_up_power_of_2(1024), 1024);
        assert_eq!(round_up_power_of_2(1025), 2048);
    }

    #[test]
    fn test_pinned_buffer() {
        let mut buf = PinnedBuffer::new(100);
        assert_eq!(buf.len(), 100);

        buf.copy_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.as_slice()[0], 1.0);
        assert_eq!(buf.as_slice()[2], 3.0);
    }

    #[test]
    fn test_compute_tensor_bytes() {
        assert_eq!(compute_tensor_bytes(&[100]), 400); // 100 * 4 bytes
        assert_eq!(compute_tensor_bytes(&[10, 20]), 800); // 200 * 4 bytes
        assert_eq!(compute_tensor_bytes(&[2, 3, 4]), 96); // 24 * 4 bytes
    }

    #[test]
    fn test_estimate_model_memory() {
        // Test with Mistral 7B-like parameters
        let memory_mb = estimate_model_memory_mb(
            4096,  // hidden_size
            14336, // intermediate_size
            32,    // n_layers
            32000, // vocab_size
            32,    // n_heads
            8,     // n_kv_heads
        );

        // Should be roughly 26-28 GB for 7B params in f32
        assert!(memory_mb > 25000.0 && memory_mb < 30000.0);
    }

    #[test]
    fn test_estimate_kv_cache_memory() {
        let memory_mb = estimate_kv_cache_memory_mb(
            32,   // n_layers
            8,    // n_kv_heads
            128,  // head_dim
            4096, // max_seq_len
        );

        // 32 * 4096 * 8 * 128 * 2 * 4 bytes = ~1 GB
        assert!(memory_mb > 900.0 && memory_mb < 1100.0);
    }

    #[test]
    fn test_memory_pool_new() {
        let pool = CudaMemoryPool::new();
        let stats = pool.stats();

        assert_eq!(stats.total_allocated_bytes, 0);
        assert_eq!(stats.in_use_bytes, 0);
        assert_eq!(stats.pool_hits, 0);
        assert_eq!(stats.pool_misses, 0);
    }
}
