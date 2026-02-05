//! Metal Tensor Types
//!
//! This module provides GPU tensor types that wrap Metal device memory buffers.

use metal::Buffer;
use std::sync::Arc;

/// A tensor stored on Metal device memory.
///
/// This type wraps a Metal `Buffer` and provides shape information.
/// Metal uses unified memory architecture on Apple Silicon, so data
/// can be accessed from both CPU and GPU without explicit copies.
#[derive(Clone)]
pub struct MetalTensor {
    /// Raw GPU buffer
    buffer: Arc<Buffer>,
    /// Shape of the tensor (e.g., [rows, cols] for 2D)
    shape: Vec<usize>,
    /// Total number of elements
    len: usize,
}

impl MetalTensor {
    /// Create a new 1D tensor from Metal buffer.
    pub fn new_1d(buffer: Buffer, len: usize) -> Self {
        Self {
            buffer: Arc::new(buffer),
            shape: vec![len],
            len,
        }
    }

    /// Create a new 2D tensor from Metal buffer.
    pub fn new_2d(buffer: Buffer, rows: usize, cols: usize) -> Self {
        let len = rows * cols;
        Self {
            buffer: Arc::new(buffer),
            shape: vec![rows, cols],
            len,
        }
    }

    /// Create a new 3D tensor from Metal buffer.
    pub fn new_3d(buffer: Buffer, d0: usize, d1: usize, d2: usize) -> Self {
        let len = d0 * d1 * d2;
        Self {
            buffer: Arc::new(buffer),
            shape: vec![d0, d1, d2],
            len,
        }
    }

    /// Create a new tensor with arbitrary shape from Metal buffer.
    pub fn new_with_shape(buffer: Buffer, shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            buffer: Arc::new(buffer),
            shape,
            len,
        }
    }

    /// Get a reference to the underlying Metal buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get shape as (rows, cols) for 2D tensors.
    pub fn shape_2d(&self) -> Option<(usize, usize)> {
        if self.shape.len() == 2 {
            Some((self.shape[0], self.shape[1]))
        } else {
            None
        }
    }

    /// Get number of rows (first dimension).
    pub fn nrows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    /// Get number of columns (second dimension).
    pub fn ncols(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[1]
        } else {
            self.shape.first().copied().unwrap_or(0)
        }
    }

    /// Reshape the tensor (must have same number of elements).
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), &'static str> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len {
            return Err("New shape must have same number of elements");
        }
        self.shape = new_shape;
        Ok(())
    }

    /// Get the memory size in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
    }

    /// Get a slice view of the data (unified memory access).
    ///
    /// # Safety
    /// This is safe on Apple Silicon due to unified memory, but the caller
    /// must ensure no GPU operations are in flight that might modify the data.
    pub fn as_slice(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(self.buffer.contents() as *const f32, self.len)
        }
    }

    /// Get a mutable slice view of the data (unified memory access).
    ///
    /// # Safety
    /// This is safe on Apple Silicon due to unified memory, but the caller
    /// must ensure no GPU operations are in flight that might modify the data.
    pub fn as_slice_mut(&self) -> &mut [f32] {
        unsafe {
            std::slice::from_raw_parts_mut(self.buffer.contents() as *mut f32, self.len)
        }
    }
}

impl std::fmt::Debug for MetalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalTensor")
            .field("shape", &self.shape)
            .field("len", &self.len)
            .field("memory_bytes", &self.memory_bytes())
            .finish()
    }
}

// =============================================================================
// Batch of Metal Tensors
// =============================================================================

/// A batch of tensors on Metal device memory.
///
/// Used for batched operations like batched matrix multiplication.
#[derive(Clone)]
pub struct MetalTensorBatch {
    /// All tensor data stored contiguously
    buffer: Arc<Buffer>,
    /// Shape of each tensor in the batch
    tensor_shape: Vec<usize>,
    /// Number of tensors in the batch
    batch_size: usize,
    /// Elements per tensor
    elements_per_tensor: usize,
}

impl MetalTensorBatch {
    /// Create a new batch from Metal buffer.
    pub fn new(buffer: Buffer, batch_size: usize, tensor_shape: Vec<usize>) -> Self {
        let elements_per_tensor: usize = tensor_shape.iter().product();
        Self {
            buffer: Arc::new(buffer),
            tensor_shape,
            batch_size,
            elements_per_tensor,
        }
    }

    /// Get a reference to the underlying Metal buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get the shape of each tensor.
    pub fn tensor_shape(&self) -> &[usize] {
        &self.tensor_shape
    }

    /// Get elements per tensor.
    pub fn elements_per_tensor(&self) -> usize {
        self.elements_per_tensor
    }

    /// Get total number of elements.
    pub fn total_elements(&self) -> usize {
        self.batch_size * self.elements_per_tensor
    }

    /// Get the memory size in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.total_elements() * std::mem::size_of::<f32>()
    }
}

impl std::fmt::Debug for MetalTensorBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalTensorBatch")
            .field("batch_size", &self.batch_size)
            .field("tensor_shape", &self.tensor_shape)
            .field("total_elements", &self.total_elements())
            .finish()
    }
}

// =============================================================================
// KV Cache on Metal
// =============================================================================

/// Key-Value cache stored on Metal GPU for efficient attention computation.
///
/// Stores the cached keys and values for all layers and all positions.
#[derive(Clone)]
pub struct MetalKVCache {
    /// Key cache: [n_layers, max_seq_len, n_kv_heads, head_dim]
    pub keys: Arc<Buffer>,
    /// Value cache: [n_layers, max_seq_len, n_kv_heads, head_dim]
    pub values: Arc<Buffer>,
    /// Number of layers
    pub n_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of KV heads
    pub n_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Current position in the cache
    pub current_pos: usize,
}

impl MetalKVCache {
    /// Get the total number of elements in one of the caches.
    pub fn cache_elements(&self) -> usize {
        self.n_layers * self.max_seq_len * self.n_kv_heads * self.head_dim
    }

    /// Get memory usage in bytes (for both K and V caches).
    pub fn memory_bytes(&self) -> usize {
        self.cache_elements() * 2 * std::mem::size_of::<f32>()
    }

    /// Get the offset for a specific layer and position.
    pub fn offset(&self, layer: usize, pos: usize) -> usize {
        let per_layer = self.max_seq_len * self.n_kv_heads * self.head_dim;
        let per_pos = self.n_kv_heads * self.head_dim;
        layer * per_layer + pos * per_pos
    }
}

impl std::fmt::Debug for MetalKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalKVCache")
            .field("n_layers", &self.n_layers)
            .field("max_seq_len", &self.max_seq_len)
            .field("n_kv_heads", &self.n_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("current_pos", &self.current_pos)
            .field("memory_mb", &(self.memory_bytes() as f64 / 1024.0 / 1024.0))
            .finish()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn test_metal_tensor_shape_logic() {
        // Test shape computation logic without actual Metal hardware
        let shape_2d = vec![10, 20];
        let len: usize = shape_2d.iter().product();
        assert_eq!(len, 200);

        let shape_3d = vec![5, 10, 20];
        let len: usize = shape_3d.iter().product();
        assert_eq!(len, 1000);
    }

    #[test]
    fn test_kv_cache_offset() {
        // Test offset calculation without real Metal memory
        let _n_layers = 32;
        let max_seq_len = 2048;
        let n_kv_heads = 8;
        let head_dim = 128;

        let per_layer = max_seq_len * n_kv_heads * head_dim;
        let per_pos = n_kv_heads * head_dim;

        // Layer 0, position 0
        let offset = 0 * per_layer + 0 * per_pos;
        assert_eq!(offset, 0);

        // Layer 1, position 0
        let offset = 1 * per_layer + 0 * per_pos;
        assert_eq!(offset, per_layer);

        // Layer 0, position 100
        let offset = 0 * per_layer + 100 * per_pos;
        assert_eq!(offset, 100 * per_pos);
    }
}
