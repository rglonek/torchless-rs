//! WebGPU Tensor Types
//!
//! This module provides GPU tensor types that wrap WebGPU device memory buffers.
//! Unlike Metal's unified memory, WebGPU buffers require explicit copy operations
//! for CPU-GPU data transfer.

use std::sync::Arc;
use wgpu::Buffer;

/// A tensor stored on WebGPU device memory.
///
/// This type wraps a WebGPU `Buffer` and provides shape information.
/// Data must be transferred to/from CPU via staging buffers - there is no
/// unified memory access like Metal on Apple Silicon.
#[derive(Clone)]
pub struct WebGPUTensor {
    /// Raw GPU buffer
    buffer: Arc<Buffer>,
    /// Shape of the tensor (e.g., [rows, cols] for 2D)
    shape: Vec<usize>,
    /// Total number of elements
    len: usize,
}

impl WebGPUTensor {
    /// Create a new 1D tensor from WebGPU buffer.
    pub fn new_1d(buffer: Buffer, len: usize) -> Self {
        Self {
            buffer: Arc::new(buffer),
            shape: vec![len],
            len,
        }
    }

    /// Create a new 2D tensor from WebGPU buffer.
    pub fn new_2d(buffer: Buffer, rows: usize, cols: usize) -> Self {
        let len = rows * cols;
        Self {
            buffer: Arc::new(buffer),
            shape: vec![rows, cols],
            len,
        }
    }

    /// Create a new 3D tensor from WebGPU buffer.
    pub fn new_3d(buffer: Buffer, d0: usize, d1: usize, d2: usize) -> Self {
        let len = d0 * d1 * d2;
        Self {
            buffer: Arc::new(buffer),
            shape: vec![d0, d1, d2],
            len,
        }
    }

    /// Create a new tensor with arbitrary shape from WebGPU buffer.
    pub fn new_with_shape(buffer: Buffer, shape: Vec<usize>) -> Self {
        let len = shape.iter().product();
        Self {
            buffer: Arc::new(buffer),
            shape,
            len,
        }
    }

    /// Get a reference to the underlying WebGPU buffer.
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
}

impl std::fmt::Debug for WebGPUTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGPUTensor")
            .field("shape", &self.shape)
            .field("len", &self.len)
            .field("memory_bytes", &self.memory_bytes())
            .finish()
    }
}
