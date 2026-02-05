//! CUDA Tensor Types
//!
//! This module provides GPU tensor types that wrap CUDA device memory.

use cudarc::driver::CudaSlice;

/// A tensor stored on CUDA device memory.
///
/// This type wraps a `CudaSlice<f32>` and provides shape information.
#[derive(Clone)]
pub struct CudaTensor {
    /// Raw GPU memory
    data: CudaSlice<f32>,
    /// Shape of the tensor (e.g., [rows, cols] for 2D)
    shape: Vec<usize>,
    /// Total number of elements
    len: usize,
}

impl CudaTensor {
    /// Create a new 1D tensor from CUDA slice.
    pub fn new_1d(data: CudaSlice<f32>, len: usize) -> Self {
        Self {
            data,
            shape: vec![len],
            len,
        }
    }

    /// Create a new 2D tensor from CUDA slice.
    pub fn new_2d(data: CudaSlice<f32>, rows: usize, cols: usize) -> Self {
        let len = rows * cols;
        Self {
            data,
            shape: vec![rows, cols],
            len,
        }
    }

    /// Create a new 3D tensor from CUDA slice.
    pub fn new_3d(data: CudaSlice<f32>, d0: usize, d1: usize, d2: usize) -> Self {
        let len = d0 * d1 * d2;
        Self {
            data,
            shape: vec![d0, d1, d2],
            len,
        }
    }

    /// Create a new tensor with arbitrary shape from CUDA slice.
    pub fn new_with_shape(data: CudaSlice<f32>, shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self { data, shape, len }
    }

    /// Get a reference to the underlying CUDA slice.
    pub fn data(&self) -> &CudaSlice<f32> {
        &self.data
    }

    /// Get a mutable reference to the underlying CUDA slice.
    pub fn data_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.data
    }

    /// Consume the tensor and return the underlying CUDA slice.
    pub fn into_data(self) -> CudaSlice<f32> {
        self.data
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
}

impl std::fmt::Debug for CudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensor")
            .field("shape", &self.shape)
            .field("len", &self.len)
            .field("memory_bytes", &self.memory_bytes())
            .finish()
    }
}

// =============================================================================
// Batch of CUDA Tensors
// =============================================================================

/// A batch of tensors on CUDA device memory.
///
/// Used for batched operations like batched matrix multiplication.
#[derive(Clone)]
pub struct CudaTensorBatch {
    /// All tensor data stored contiguously
    data: CudaSlice<f32>,
    /// Shape of each tensor in the batch
    tensor_shape: Vec<usize>,
    /// Number of tensors in the batch
    batch_size: usize,
    /// Elements per tensor
    elements_per_tensor: usize,
}

impl CudaTensorBatch {
    /// Create a new batch from CUDA slice.
    pub fn new(data: CudaSlice<f32>, batch_size: usize, tensor_shape: Vec<usize>) -> Self {
        let elements_per_tensor: usize = tensor_shape.iter().product();
        Self {
            data,
            tensor_shape,
            batch_size,
            elements_per_tensor,
        }
    }

    /// Get a reference to the underlying CUDA slice.
    pub fn data(&self) -> &CudaSlice<f32> {
        &self.data
    }

    /// Get a mutable reference to the underlying CUDA slice.
    pub fn data_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.data
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

impl std::fmt::Debug for CudaTensorBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensorBatch")
            .field("batch_size", &self.batch_size)
            .field("tensor_shape", &self.tensor_shape)
            .field("total_elements", &self.total_elements())
            .finish()
    }
}

// =============================================================================
// KV Cache on GPU
// =============================================================================

/// Key-Value cache stored on GPU for efficient attention computation.
///
/// Stores the cached keys and values for all layers and all positions.
#[derive(Clone)]
pub struct CudaKVCache {
    /// Key cache: [n_layers, max_seq_len, n_kv_heads, head_dim]
    pub keys: CudaSlice<f32>,
    /// Value cache: [n_layers, max_seq_len, n_kv_heads, head_dim]
    pub values: CudaSlice<f32>,
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

impl CudaKVCache {
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

impl std::fmt::Debug for CudaKVCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaKVCache")
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
    use super::*;

    #[test]
    fn test_cuda_tensor_shape() {
        // This test doesn't require actual CUDA hardware
        // Just test the shape logic
        
        // We can't create real CudaSlice without a device, but we can test
        // the shape computation logic through other means
        let shape_2d = vec![10, 20];
        let len: usize = shape_2d.iter().product();
        assert_eq!(len, 200);
        
        let shape_3d = vec![5, 10, 20];
        let len: usize = shape_3d.iter().product();
        assert_eq!(len, 1000);
    }

    #[test]
    fn test_kv_cache_offset() {
        // Test offset calculation without real CUDA memory
        let n_layers = 32;
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
