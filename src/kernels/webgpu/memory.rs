//! WebGPU Memory Management
//!
//! This module provides GPU memory management utilities for WebGPU,
//! including buffer pooling for efficient memory reuse.
//!
//! WebGPU uses discrete memory - buffers must be explicitly copied
//! between CPU and GPU via staging buffers for readback.

use std::collections::BTreeMap;
use wgpu::util::DeviceExt;

// =============================================================================
// Memory Pool
// =============================================================================

/// A memory pool for reusing WebGPU buffers.
///
/// Allocating GPU buffers can be expensive. This pool keeps freed
/// buffers around for reuse, significantly reducing allocation overhead
/// during inference.
#[derive(Debug)]
pub struct WebGPUMemoryPool {
    /// Free buffers organized by size (rounded up to nearest power of 2)
    free_buffers: BTreeMap<usize, Vec<wgpu::Buffer>>,
    /// Total bytes allocated (including freed buffers in pool)
    total_allocated_bytes: usize,
    /// Total bytes currently in use
    in_use_bytes: usize,
    /// Number of allocations served from pool
    pool_hits: usize,
    /// Number of allocations that required new allocation
    pool_misses: usize,
}

impl WebGPUMemoryPool {
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
        device: &wgpu::Device,
        min_elements: usize,
        usage: wgpu::BufferUsages,
    ) -> anyhow::Result<wgpu::Buffer> {
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
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WebGPU pool buffer"),
            size: bucket_size as u64,
            usage,
            mapped_at_creation: false,
        });

        self.total_allocated_bytes += bucket_size;
        self.in_use_bytes += bucket_size;
        self.pool_misses += 1;

        Ok(buffer)
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buffer(&mut self, buffer: wgpu::Buffer) {
        let size = buffer.size() as usize;
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
    pub fn stats(&self) -> WebGPUMemoryPoolStats {
        let pool_bytes: usize = self
            .free_buffers
            .iter()
            .map(|(size, buffers)| size * buffers.len())
            .sum();

        WebGPUMemoryPoolStats {
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

impl Default for WebGPUMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about memory pool usage.
#[derive(Debug, Clone)]
pub struct WebGPUMemoryPoolStats {
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

impl std::fmt::Display for WebGPUMemoryPoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WebGPUMemoryPool: {:.2} MB allocated, {:.2} MB in use, {:.2} MB pooled, {:.1}% hit rate ({} hits, {} misses)",
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

/// Buffer usage for compute operations.
fn compute_buffer_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}

/// Create a new WebGPU buffer with the given data.
pub fn create_buffer_with_data(device: &wgpu::Device, data: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("WebGPU buffer with data"),
        contents: bytemuck::cast_slice(data),
        usage: compute_buffer_usage(),
    })
}

/// Create a new zero-initialized WebGPU buffer.
pub fn create_buffer_zeros(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    let data = vec![0.0f32; len];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("WebGPU zero buffer"),
        contents: bytemuck::cast_slice(&data),
        usage: compute_buffer_usage(),
    })
}

/// Read data from a GPU buffer back to CPU.
///
/// Uses a staging buffer and `map_async` with `pollster::block_on` for blocking.
pub fn read_buffer_to_vec(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<f32> {
    let size = (len * std::mem::size_of::<f32>()) as u64;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("WebGPU staging buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("WebGPU readback encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    rx.recv().unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    result
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Round up to the nearest power of 2.
pub fn round_up_power_of_2(n: usize) -> usize {
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

/// Estimate memory required for model weights on WebGPU.
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
    let norm = hidden_size * 2;

    let layer_params = q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj + norm;
    let total_layer_params = layer_params * n_layers;

    let embedding = vocab_size * hidden_size;
    let output = vocab_size * hidden_size;
    let final_norm = hidden_size;

    let total_params = total_layer_params + embedding + output + final_norm;

    (total_params * std::mem::size_of::<f32>()) as f64 / 1024.0 / 1024.0
}

/// Estimate memory required for KV cache on WebGPU.
pub fn estimate_kv_cache_memory_mb(
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) -> f64 {
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
        assert_eq!(compute_tensor_bytes(&[100]), 400);
        assert_eq!(compute_tensor_bytes(&[10, 20]), 800);
        assert_eq!(compute_tensor_bytes(&[2, 3, 4]), 96);
    }

    #[test]
    fn test_estimate_model_memory() {
        let memory_mb = estimate_model_memory_mb(4096, 14336, 32, 32000, 32, 8);
        assert!(memory_mb > 25000.0 && memory_mb < 30000.0);
    }

    #[test]
    fn test_estimate_kv_cache_memory() {
        let memory_mb = estimate_kv_cache_memory_mb(32, 8, 128, 4096);
        assert!(memory_mb > 900.0 && memory_mb < 1100.0);
    }

    #[test]
    fn test_memory_pool_new() {
        let pool = WebGPUMemoryPool::new();
        let stats = pool.stats();

        assert_eq!(stats.total_allocated_bytes, 0);
        assert_eq!(stats.in_use_bytes, 0);
        assert_eq!(stats.pool_hits, 0);
        assert_eq!(stats.pool_misses, 0);
    }
}
