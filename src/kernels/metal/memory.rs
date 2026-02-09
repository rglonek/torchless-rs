//! Metal Memory Management
//!
//! This module provides GPU memory management utilities for Metal,
//! including buffer pooling for efficient memory reuse.
//!
//! Apple Silicon uses Unified Memory Architecture (UMA), which means
//! the GPU and CPU share the same memory pool. This eliminates the need
//! for explicit memory copies between CPU and GPU.

use metal::{Buffer, Device, MTLResourceOptions};
use std::collections::BTreeMap;

// =============================================================================
// Memory Pool
// =============================================================================

/// A memory pool for reusing Metal buffers.
///
/// Allocating GPU buffers can be expensive. This pool keeps freed
/// buffers around for reuse, significantly reducing allocation overhead
/// during inference.
///
/// Metal on Apple Silicon uses unified memory, so buffers are accessible
/// from both CPU and GPU without explicit copies.
#[derive(Debug)]
pub struct MetalMemoryPool {
    /// Free buffers organized by size (rounded up to nearest power of 2)
    free_buffers: BTreeMap<usize, Vec<Buffer>>,
    /// Total bytes allocated (including freed buffers in pool)
    total_allocated_bytes: usize,
    /// Total bytes currently in use
    in_use_bytes: usize,
    /// Number of allocations served from pool
    pool_hits: usize,
    /// Number of allocations that required new allocation
    pool_misses: usize,
}

impl MetalMemoryPool {
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
    pub fn get_or_alloc(&mut self, device: &Device, min_elements: usize) -> anyhow::Result<Buffer> {
        let min_size = min_elements * std::mem::size_of::<f32>();
        let bucket_size = round_up_power_of_2(min_size);

        // Try to get a buffer from the pool
        if let Some(buffers) = self.free_buffers.get_mut(&bucket_size) {
            if let Some(buffer) = buffers.pop() {
                self.in_use_bytes += bucket_size;
                self.pool_hits += 1;
                return Ok(buffer);
            }
        }

        // Also check larger buckets
        for (&size, buffers) in self.free_buffers.range_mut(bucket_size..) {
            if let Some(buffer) = buffers.pop() {
                self.in_use_bytes += size;
                self.pool_hits += 1;
                return Ok(buffer);
            }
        }

        // No suitable buffer found, allocate new one
        let buffer = device.new_buffer(bucket_size as u64, MTLResourceOptions::StorageModeShared);

        self.total_allocated_bytes += bucket_size;
        self.in_use_bytes += bucket_size;
        self.pool_misses += 1;

        Ok(buffer)
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buffer(&mut self, buffer: Buffer) {
        let size = buffer.length() as usize;
        let bucket_size = round_up_power_of_2(size);

        self.in_use_bytes = self.in_use_bytes.saturating_sub(bucket_size);

        self.free_buffers
            .entry(bucket_size)
            .or_default()
            .push(buffer);
    }

    /// Clear all buffers from the pool, freeing memory.
    pub fn clear(&mut self) {
        self.free_buffers.clear();
        self.total_allocated_bytes = self.in_use_bytes;
    }

    /// Get statistics about the pool.
    pub fn stats(&self) -> MetalMemoryPoolStats {
        let pool_bytes: usize = self
            .free_buffers
            .iter()
            .map(|(size, buffers)| size * buffers.len())
            .sum();

        MetalMemoryPoolStats {
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

impl Default for MetalMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about memory pool usage.
#[derive(Debug, Clone)]
pub struct MetalMemoryPoolStats {
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

impl std::fmt::Display for MetalMemoryPoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MetalMemoryPool: {:.2} MB allocated, {:.2} MB in use, {:.2} MB pooled, {:.1}% hit rate ({} hits, {} misses)",
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
// Buffer Allocation Helpers
// =============================================================================

/// Create a new Metal buffer with the given data.
pub fn create_buffer_with_data(device: &Device, data: &[f32]) -> Buffer {
    let size = data.len() * std::mem::size_of::<f32>();
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        size as u64,
        MTLResourceOptions::StorageModeShared,
    );
    buffer
}

/// Create a new zero-initialized Metal buffer.
pub fn create_buffer_zeros(device: &Device, len: usize) -> Buffer {
    let size = len * std::mem::size_of::<f32>();
    let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);

    // Zero the buffer
    unsafe {
        std::ptr::write_bytes(buffer.contents() as *mut u8, 0, size);
    }

    buffer
}

/// Copy data from a buffer to a slice.
pub fn copy_buffer_to_slice(buffer: &Buffer, dst: &mut [f32]) {
    let src = buffer.contents() as *const f32;
    let len = dst
        .len()
        .min(buffer.length() as usize / std::mem::size_of::<f32>());
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), len);
    }
}

/// Copy data from a slice to a buffer.
pub fn copy_slice_to_buffer(src: &[f32], buffer: &Buffer) {
    let dst = buffer.contents() as *mut f32;
    let len = src
        .len()
        .min(buffer.length() as usize / std::mem::size_of::<f32>());
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, len);
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

/// Get total memory available on the Metal device.
pub fn get_device_memory_size(device: &Device) -> u64 {
    // On Apple Silicon, this returns the recommended working set size
    device.recommended_max_working_set_size()
}

/// Estimate memory required for model weights on Metal GPU.
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

/// Estimate memory required for KV cache on Metal GPU.
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
        let pool = MetalMemoryPool::new();
        let stats = pool.stats();

        assert_eq!(stats.total_allocated_bytes, 0);
        assert_eq!(stats.in_use_bytes, 0);
        assert_eq!(stats.pool_hits, 0);
        assert_eq!(stats.pool_misses, 0);
    }
}
