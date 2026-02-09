//! Tensor Storage Abstraction
//!
//! This module provides a unified tensor interface that can represent data on
//! different devices (CPU, GPU) and in different formats (f32, f16, quantized).
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      TensorStorage                          │
//! │  (Enum over device-specific storage types)                  │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!        ┌─────────────────────┼─────────────────────┐
//!        ▼                     ▼                     ▼
//! ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
//! │ CpuStorage  │       │ CudaStorage │       │ MetalStorage│
//! └─────────────┘       └─────────────┘       └─────────────┘
//! ```
//!
//! # Data Types
//!
//! The storage layer supports multiple data types for memory efficiency:
//! - `F32`: Full precision (4 bytes per element)
//! - `F16`: Half precision (2 bytes per element)
//! - `BF16`: Brain float (2 bytes per element)
//! - `Int8`: 8-bit quantized (1 byte per element + scales)
//! - `Int4`: 4-bit quantized (0.5 bytes per element + scales)
//!
//! # Mixed Precision Support
//!
//! Mixed precision allows using different dtypes for different parts of the model:
//! - Weights: INT4/INT8 for memory efficiency
//! - Activations: FP16/FP32 for numerical stability
//! - KV Cache: FP16 for balance of memory and precision
//! - Attention scores: FP32 for numerical stability

use half::{bf16, f16};
use ndarray::{Array, Ix1, Ix2, Ix3, Ix4};
use std::fmt;

// =============================================================================
// Data Types
// =============================================================================

/// Supported tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// 16-bit brain floating point (truncated f32)
    BF16,
    /// 8-bit signed integer (quantized)
    Int8,
    /// 4-bit signed integer (quantized, packed)
    Int4,
}

impl Dtype {
    /// Returns the size in bytes of a single element (for Int4, returns 0.5 conceptually).
    pub fn element_size(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::Int8 => 1,
            Dtype::Int4 => 1, // Actually 0.5, but stored as packed pairs
        }
    }

    /// Returns true if this dtype requires quantization scales.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Dtype::Int8 | Dtype::Int4)
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dtype::F32 => write!(f, "f32"),
            Dtype::F16 => write!(f, "f16"),
            Dtype::BF16 => write!(f, "bf16"),
            Dtype::Int8 => write!(f, "int8"),
            Dtype::Int4 => write!(f, "int4"),
        }
    }
}

// =============================================================================
// Device Types
// =============================================================================

/// Supported compute devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// CPU with system memory
    #[default]
    Cpu,
    /// NVIDIA GPU via CUDA
    #[cfg(feature = "cuda")]
    Cuda(usize), // device index
    /// Apple Silicon GPU via Metal
    #[cfg(feature = "metal-gpu")]
    Metal,
    /// AMD GPU via ROCm
    #[cfg(feature = "rocm")]
    Rocm(usize), // device index
    /// Cross-platform GPU via OpenCL
    #[cfg(feature = "opencl")]
    OpenCL(usize), // device index
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            Device::Cuda(idx) => write!(f, "cuda:{}", idx),
            #[cfg(feature = "metal-gpu")]
            Device::Metal => write!(f, "metal"),
            #[cfg(feature = "rocm")]
            Device::Rocm(idx) => write!(f, "rocm:{}", idx),
            #[cfg(feature = "opencl")]
            Device::OpenCL(idx) => write!(f, "opencl:{}", idx),
        }
    }
}

// =============================================================================
// CPU Storage Types
// =============================================================================

/// CPU storage for f32 tensors.
#[derive(Debug, Clone)]
pub struct CpuF32Storage {
    pub data: Vec<f32>,
}

/// CPU storage for f16 tensors.
#[derive(Debug, Clone)]
pub struct CpuF16Storage {
    pub data: Vec<f16>,
}

/// CPU storage for bf16 tensors.
#[derive(Debug, Clone)]
pub struct CpuBF16Storage {
    pub data: Vec<bf16>,
}

/// CPU storage for quantized int8 tensors.
#[derive(Debug, Clone)]
pub struct CpuInt8Storage {
    pub data: Vec<i8>,
    /// Quantization scales, one per group
    pub scales: Vec<f32>,
    /// Number of elements per quantization group
    pub group_size: usize,
}

/// CPU storage for quantized int4 tensors (packed, 2 values per byte).
#[derive(Debug, Clone)]
pub struct CpuInt4Storage {
    /// Packed int4 values (2 per byte, lower nibble first)
    pub data: Vec<u8>,
    /// Quantization scales, one per group
    pub scales: Vec<f32>,
    /// Number of elements per quantization group
    pub group_size: usize,
}

impl CpuInt4Storage {
    /// Get a single int4 value at the given index.
    pub fn get(&self, index: usize) -> i8 {
        let byte_idx = index / 2;
        let byte = self.data[byte_idx];
        if index.is_multiple_of(2) {
            // Lower nibble
            ((byte & 0x0F) as i8) - 8
        } else {
            // Upper nibble
            ((byte >> 4) as i8) - 8
        }
    }

    /// Dequantize a single value at the given index.
    pub fn dequantize(&self, index: usize) -> f32 {
        let q_val = self.get(index);
        let group_idx = index / self.group_size;
        let scale = self.scales[group_idx];
        q_val as f32 * scale
    }
}

// =============================================================================
// GPU Storage Wrapper Types
// =============================================================================

/// CUDA f32 storage (GPU).
#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct CudaF32Storage {
    pub tensor: crate::kernels::cuda::CudaTensor,
    /// Device handle needed to copy data back to host in to_f32().
    pub device: std::sync::Arc<cudarc::driver::CudaDevice>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaF32Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaF32Storage")
            .field("tensor", &self.tensor)
            .finish()
    }
}

/// Metal f32 storage (GPU).
#[cfg(feature = "metal-gpu")]
#[derive(Debug, Clone)]
pub struct MetalF32Storage {
    pub tensor: crate::kernels::metal::MetalTensor,
}

/// ROCm f32 storage (GPU).
#[cfg(feature = "rocm")]
#[derive(Clone)]
pub struct RocmF32Storage {
    pub tensor: crate::kernels::rocm::RocmTensor,
    /// Backend handle needed to copy data back to host in to_f32().
    pub backend: std::sync::Arc<crate::kernels::rocm::RocmBackend>,
}

#[cfg(feature = "rocm")]
impl std::fmt::Debug for RocmF32Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmF32Storage")
            .field("tensor", &self.tensor)
            .finish()
    }
}

/// OpenCL f32 storage (GPU).
#[cfg(feature = "opencl")]
#[derive(Debug, Clone)]
pub struct OpenCLF32Storage {
    pub tensor: crate::kernels::opencl::OpenCLTensor,
}

// =============================================================================
// Unified Tensor Storage
// =============================================================================

/// Unified tensor storage that can represent data on any device.
#[derive(Debug, Clone)]
pub enum TensorStorage {
    /// CPU f32 storage
    CpuF32(CpuF32Storage),
    /// CPU f16 storage
    CpuF16(CpuF16Storage),
    /// CPU bf16 storage
    CpuBF16(CpuBF16Storage),
    /// CPU int8 quantized storage
    CpuInt8(CpuInt8Storage),
    /// CPU int4 quantized storage
    CpuInt4(CpuInt4Storage),
    /// CUDA f32 storage (GPU)
    #[cfg(feature = "cuda")]
    CudaF32(CudaF32Storage),
    /// Metal f32 storage (GPU)
    #[cfg(feature = "metal-gpu")]
    MetalF32(MetalF32Storage),
    /// ROCm f32 storage (GPU)
    #[cfg(feature = "rocm")]
    RocmF32(RocmF32Storage),
    /// OpenCL f32 storage (GPU)
    #[cfg(feature = "opencl")]
    OpenCLF32(OpenCLF32Storage),
}

impl TensorStorage {
    /// Get the dtype of this storage.
    pub fn dtype(&self) -> Dtype {
        match self {
            TensorStorage::CpuF32(_) => Dtype::F32,
            TensorStorage::CpuF16(_) => Dtype::F16,
            TensorStorage::CpuBF16(_) => Dtype::BF16,
            TensorStorage::CpuInt8(_) => Dtype::Int8,
            TensorStorage::CpuInt4(_) => Dtype::Int4,
            #[cfg(feature = "cuda")]
            TensorStorage::CudaF32(_) => Dtype::F32,
            #[cfg(feature = "metal-gpu")]
            TensorStorage::MetalF32(_) => Dtype::F32,
            #[cfg(feature = "rocm")]
            TensorStorage::RocmF32(_) => Dtype::F32,
            #[cfg(feature = "opencl")]
            TensorStorage::OpenCLF32(_) => Dtype::F32,
        }
    }

    /// Get the device of this storage.
    pub fn device(&self) -> Device {
        match self {
            TensorStorage::CpuF32(_)
            | TensorStorage::CpuF16(_)
            | TensorStorage::CpuBF16(_)
            | TensorStorage::CpuInt8(_)
            | TensorStorage::CpuInt4(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            TensorStorage::CudaF32(_) => Device::Cuda(0),
            #[cfg(feature = "metal-gpu")]
            TensorStorage::MetalF32(_) => Device::Metal,
            #[cfg(feature = "rocm")]
            TensorStorage::RocmF32(_) => Device::Rocm(0),
            #[cfg(feature = "opencl")]
            TensorStorage::OpenCLF32(_) => Device::OpenCL(0),
        }
    }

    /// Get the number of elements in this storage.
    pub fn len(&self) -> usize {
        match self {
            TensorStorage::CpuF32(s) => s.data.len(),
            TensorStorage::CpuF16(s) => s.data.len(),
            TensorStorage::CpuBF16(s) => s.data.len(),
            TensorStorage::CpuInt8(s) => s.data.len(),
            TensorStorage::CpuInt4(s) => s.data.len() * 2,
            #[cfg(feature = "cuda")]
            TensorStorage::CudaF32(s) => s.tensor.len(),
            #[cfg(feature = "metal-gpu")]
            TensorStorage::MetalF32(s) => s.tensor.len(),
            #[cfg(feature = "rocm")]
            TensorStorage::RocmF32(s) => s.tensor.len(),
            #[cfg(feature = "opencl")]
            TensorStorage::OpenCLF32(s) => s.tensor.len(),
        }
    }

    /// Returns true if this storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create f32 storage from a Vec.
    pub fn from_f32(data: Vec<f32>) -> Self {
        TensorStorage::CpuF32(CpuF32Storage { data })
    }

    /// Create f16 storage from a Vec of f32 (converts to f16).
    pub fn from_f32_as_f16(data: Vec<f32>) -> Self {
        let f16_data: Vec<f16> = data.into_iter().map(f16::from_f32).collect();
        TensorStorage::CpuF16(CpuF16Storage { data: f16_data })
    }

    /// Create bf16 storage from a Vec of f32 (converts to bf16).
    pub fn from_f32_as_bf16(data: Vec<f32>) -> Self {
        let bf16_data: Vec<bf16> = data.into_iter().map(bf16::from_f32).collect();
        TensorStorage::CpuBF16(CpuBF16Storage { data: bf16_data })
    }

    /// Create int8 quantized storage from f32 data.
    pub fn from_f32_quantize_int8(data: Vec<f32>, group_size: usize) -> Self {
        let n_groups = data.len().div_ceil(group_size);
        let mut scales = Vec::with_capacity(n_groups);
        let mut quantized = Vec::with_capacity(data.len());

        for group in data.chunks(group_size) {
            // Find max absolute value in group
            let max_abs = group.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b));

            // Scale to fit in [-127, 127]
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales.push(scale);

            // Quantize values
            for &val in group {
                let q = (val / scale).round().clamp(-127.0, 127.0) as i8;
                quantized.push(q);
            }
        }

        TensorStorage::CpuInt8(CpuInt8Storage {
            data: quantized,
            scales,
            group_size,
        })
    }

    /// Create int4 quantized storage from f32 data.
    pub fn from_f32_quantize_int4(data: Vec<f32>, group_size: usize) -> Self {
        let n_groups = data.len().div_ceil(group_size);
        let mut scales = Vec::with_capacity(n_groups);
        let packed_len = data.len().div_ceil(2);
        let mut packed = Vec::with_capacity(packed_len);

        for group in data.chunks(group_size) {
            // Find max absolute value in group
            let max_abs = group.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b));

            // Scale to fit in [-7, 7] (we use offset encoding: stored as [0, 15], represent -8 to 7)
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales.push(scale);
        }

        // Pack values into nibbles
        for pair in data.chunks(2) {
            let q0 = quantize_int4(pair[0], &scales, 0, group_size);
            let q1 = if pair.len() > 1 {
                quantize_int4(pair[1], &scales, 1, group_size)
            } else {
                8 // Zero in offset encoding
            };
            // Pack: lower nibble = first value, upper nibble = second value
            packed.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }

        TensorStorage::CpuInt4(CpuInt4Storage {
            data: packed,
            scales,
            group_size,
        })
    }

    /// Dequantize to f32 Vec.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            TensorStorage::CpuF32(s) => s.data.clone(),
            TensorStorage::CpuF16(s) => s.data.iter().map(|x| x.to_f32()).collect(),
            TensorStorage::CpuBF16(s) => s.data.iter().map(|x| x.to_f32()).collect(),
            TensorStorage::CpuInt8(s) => {
                let mut result = Vec::with_capacity(s.data.len());
                for (i, &q) in s.data.iter().enumerate() {
                    let group_idx = i / s.group_size;
                    let scale = s.scales[group_idx];
                    result.push(q as f32 * scale);
                }
                result
            }
            TensorStorage::CpuInt4(s) => {
                let len = s.data.len() * 2;
                let mut result = Vec::with_capacity(len);
                for i in 0..len {
                    result.push(s.dequantize(i));
                }
                result
            }
            #[cfg(feature = "cuda")]
            TensorStorage::CudaF32(s) => self
                .cuda_to_f32(s)
                .expect("Failed to copy CUDA tensor to host"),
            #[cfg(feature = "metal-gpu")]
            TensorStorage::MetalF32(s) => s.tensor.as_slice().to_vec(),
            #[cfg(feature = "rocm")]
            TensorStorage::RocmF32(s) => s
                .backend
                .copy_tensor_to_host(&s.tensor)
                .expect("Failed to copy ROCm tensor to host"),
            #[cfg(feature = "opencl")]
            TensorStorage::OpenCLF32(s) => s
                .tensor
                .to_vec()
                .expect("Failed to copy OpenCL tensor to host"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_to_f32(&self, s: &CudaF32Storage) -> anyhow::Result<Vec<f32>> {
        use cudarc::driver::DriverError;
        s.device
            .dtoh_sync_copy(s.tensor.data())
            .map_err(|e: DriverError| anyhow::anyhow!("CUDA copy failed: {:?}", e))
    }

    /// Get as f32 slice (only valid for CpuF32 storage).
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            TensorStorage::CpuF32(s) => Some(&s.data),
            _ => None,
        }
    }

    /// Get as mutable f32 slice (only valid for CpuF32 storage).
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            TensorStorage::CpuF32(s) => Some(&mut s.data),
            _ => None,
        }
    }

    /// Get as f16 slice (only valid for CpuF16 storage).
    pub fn as_f16_slice(&self) -> Option<&[f16]> {
        match self {
            TensorStorage::CpuF16(s) => Some(&s.data),
            _ => None,
        }
    }

    /// Get as bf16 slice (only valid for CpuBF16 storage).
    pub fn as_bf16_slice(&self) -> Option<&[bf16]> {
        match self {
            TensorStorage::CpuBF16(s) => Some(&s.data),
            _ => None,
        }
    }

    /// Convert storage to a different dtype.
    pub fn cast(&self, target_dtype: Dtype) -> Self {
        if self.dtype() == target_dtype {
            return self.clone();
        }

        let f32_data = self.to_f32();
        match target_dtype {
            Dtype::F32 => TensorStorage::from_f32(f32_data),
            Dtype::F16 => TensorStorage::from_f32_as_f16(f32_data),
            Dtype::BF16 => TensorStorage::from_f32_as_bf16(f32_data),
            Dtype::Int8 => TensorStorage::from_f32_quantize_int8(f32_data, 64),
            Dtype::Int4 => TensorStorage::from_f32_quantize_int4(f32_data, 32),
        }
    }
}

// Helper function to quantize a single value to int4
fn quantize_int4(val: f32, scales: &[f32], idx: usize, group_size: usize) -> u8 {
    let group_idx = idx / group_size;
    let scale = scales.get(group_idx).copied().unwrap_or(1.0);
    let q = (val / scale).round().clamp(-8.0, 7.0) as i8;
    // Convert from [-8, 7] to [0, 15] for storage
    (q + 8) as u8
}

// =============================================================================
// Unified Tensor
// =============================================================================

/// A unified tensor type that can hold data on any device.
#[derive(Debug, Clone)]
pub struct UnifiedTensor {
    /// The underlying storage
    pub storage: TensorStorage,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Strides for each dimension
    pub strides: Vec<usize>,
}

impl UnifiedTensor {
    /// Create a new tensor from storage and shape.
    pub fn new(storage: TensorStorage, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Self {
            storage,
            shape,
            strides,
        }
    }

    /// Create a tensor from f32 data.
    pub fn from_f32(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let storage = TensorStorage::from_f32(data);
        Self::new(storage, shape)
    }

    /// Create a tensor from f32 data, stored as f16.
    pub fn from_f32_as_f16(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let storage = TensorStorage::from_f32_as_f16(data);
        Self::new(storage, shape)
    }

    /// Create a tensor from f32 data, quantized to int8.
    pub fn from_f32_quantize_int8(data: Vec<f32>, shape: Vec<usize>, group_size: usize) -> Self {
        let storage = TensorStorage::from_f32_quantize_int8(data, group_size);
        Self::new(storage, shape)
    }

    /// Create a tensor from f32 data, quantized to int4.
    pub fn from_f32_quantize_int4(data: Vec<f32>, shape: Vec<usize>, group_size: usize) -> Self {
        let storage = TensorStorage::from_f32_quantize_int4(data, group_size);
        Self::new(storage, shape)
    }

    /// Get the dtype of this tensor.
    pub fn dtype(&self) -> Dtype {
        self.storage.dtype()
    }

    /// Get the device of this tensor.
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Get the shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the number of rows (for 2D tensors).
    pub fn nrows(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[0]
        } else {
            1
        }
    }

    /// Get the number of columns (for 2D tensors).
    pub fn ncols(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[1]
        } else {
            self.shape.first().copied().unwrap_or(0)
        }
    }

    /// Convert to ndarray Array1 (dequantizes if needed).
    pub fn to_array1(&self) -> Array<f32, Ix1> {
        let data = self.storage.to_f32();
        Array::from_vec(data)
    }

    /// Convert to ndarray Array2 (dequantizes if needed).
    pub fn to_array2(&self) -> Array<f32, Ix2> {
        assert!(self.shape.len() == 2, "Tensor must be 2D");
        let data = self.storage.to_f32();
        Array::from_shape_vec((self.shape[0], self.shape[1]), data).expect("Shape mismatch")
    }

    /// Convert to ndarray Array3 (dequantizes if needed).
    pub fn to_array3(&self) -> Array<f32, Ix3> {
        assert!(self.shape.len() == 3, "Tensor must be 3D");
        let data = self.storage.to_f32();
        Array::from_shape_vec((self.shape[0], self.shape[1], self.shape[2]), data)
            .expect("Shape mismatch")
    }

    /// Convert to ndarray Array4 (dequantizes if needed).
    pub fn to_array4(&self) -> Array<f32, Ix4> {
        assert!(self.shape.len() == 4, "Tensor must be 4D");
        let data = self.storage.to_f32();
        Array::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2], self.shape[3]),
            data,
        )
        .expect("Shape mismatch")
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match &self.storage {
            TensorStorage::CpuF32(s) => s.data.len() * 4,
            TensorStorage::CpuF16(s) => s.data.len() * 2,
            TensorStorage::CpuBF16(s) => s.data.len() * 2,
            TensorStorage::CpuInt8(s) => s.data.len() + s.scales.len() * 4,
            TensorStorage::CpuInt4(s) => s.data.len() + s.scales.len() * 4,
            #[cfg(feature = "cuda")]
            TensorStorage::CudaF32(s) => s.tensor.memory_bytes(),
            #[cfg(feature = "metal-gpu")]
            TensorStorage::MetalF32(s) => s.tensor.memory_bytes(),
            #[cfg(feature = "rocm")]
            TensorStorage::RocmF32(s) => s.tensor.memory_bytes(),
            #[cfg(feature = "opencl")]
            TensorStorage::OpenCLF32(s) => s.tensor.memory_bytes(),
        }
    }

    /// Convert this tensor to a different dtype.
    pub fn cast(&self, target_dtype: Dtype) -> Self {
        Self::new(self.storage.cast(target_dtype), self.shape.clone())
    }

    /// Get the compression ratio compared to f32.
    pub fn compression_ratio(&self) -> f32 {
        let f32_bytes = self.numel() * 4;
        f32_bytes as f32 / self.memory_bytes() as f32
    }
}

// =============================================================================
// Mixed Precision Configuration
// =============================================================================

/// Configuration for mixed precision inference.
///
/// Mixed precision allows using different data types for different parts of the model
/// to balance memory usage, speed, and numerical precision.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Data type for model weights (dense layers).
    /// Use INT4/INT8 for memory efficiency.
    pub weights_dtype: Dtype,

    /// Data type for embedding weights.
    /// Often kept at higher precision for quality.
    pub embedding_dtype: Dtype,

    /// Data type for activations (intermediate computations).
    /// FP16/FP32 for numerical stability.
    pub activation_dtype: Dtype,

    /// Data type for KV cache storage.
    /// FP16 is common for balance of memory and precision.
    pub kv_cache_dtype: Dtype,

    /// Data type for attention score computation.
    /// FP32 is recommended for numerical stability.
    pub attention_dtype: Dtype,

    /// Data type for output logits.
    pub output_dtype: Dtype,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

impl MixedPrecisionConfig {
    /// Full precision configuration (FP32 everywhere).
    /// Best quality but highest memory usage.
    pub fn full_precision() -> Self {
        Self {
            weights_dtype: Dtype::F32,
            embedding_dtype: Dtype::F32,
            activation_dtype: Dtype::F32,
            kv_cache_dtype: Dtype::F32,
            attention_dtype: Dtype::F32,
            output_dtype: Dtype::F32,
        }
    }

    /// Half precision configuration (FP16 everywhere).
    /// Good balance of speed and quality.
    pub fn half_precision() -> Self {
        Self {
            weights_dtype: Dtype::F16,
            embedding_dtype: Dtype::F16,
            activation_dtype: Dtype::F16,
            kv_cache_dtype: Dtype::F16,
            attention_dtype: Dtype::F32, // Keep attention in FP32 for stability
            output_dtype: Dtype::F32,
        }
    }

    /// Balanced mixed precision (INT8 weights, FP16 activations).
    /// Good memory efficiency with minimal quality loss.
    pub fn balanced() -> Self {
        Self {
            weights_dtype: Dtype::Int8,
            embedding_dtype: Dtype::F16,
            activation_dtype: Dtype::F16,
            kv_cache_dtype: Dtype::F16,
            attention_dtype: Dtype::F32,
            output_dtype: Dtype::F32,
        }
    }

    /// Aggressive quantization (INT4 weights, FP16 activations).
    /// Maximum memory efficiency, some quality loss.
    pub fn aggressive() -> Self {
        Self {
            weights_dtype: Dtype::Int4,
            embedding_dtype: Dtype::F16,
            activation_dtype: Dtype::F16,
            kv_cache_dtype: Dtype::F16,
            attention_dtype: Dtype::F32,
            output_dtype: Dtype::F32,
        }
    }

    /// Memory-constrained configuration (INT4 weights, INT8 KV cache).
    /// For running larger models on limited hardware.
    pub fn memory_constrained() -> Self {
        Self {
            weights_dtype: Dtype::Int4,
            embedding_dtype: Dtype::Int8,
            activation_dtype: Dtype::F16,
            kv_cache_dtype: Dtype::Int8,
            attention_dtype: Dtype::F32,
            output_dtype: Dtype::F32,
        }
    }

    /// Estimate memory usage for a model with given parameters.
    pub fn estimate_memory_mb(&self, params: ModelSizeParams) -> f32 {
        let weights_bytes =
            params.total_weight_params as f32 * self.weights_dtype.element_size() as f32;
        let embedding_bytes =
            params.embedding_params as f32 * self.embedding_dtype.element_size() as f32;
        let kv_cache_bytes =
            params.kv_cache_elements as f32 * self.kv_cache_dtype.element_size() as f32;
        let activation_bytes =
            params.max_activation_elements as f32 * self.activation_dtype.element_size() as f32;

        (weights_bytes + embedding_bytes + kv_cache_bytes + activation_bytes) / (1024.0 * 1024.0)
    }
}

/// Parameters describing model size for memory estimation.
#[derive(Debug, Clone)]
pub struct ModelSizeParams {
    /// Total number of weight parameters (excluding embeddings).
    pub total_weight_params: usize,
    /// Number of embedding parameters.
    pub embedding_params: usize,
    /// Number of elements in KV cache (n_layers * 2 * n_kv_heads * head_dim * max_seq_len).
    pub kv_cache_elements: usize,
    /// Maximum number of activation elements at any point.
    pub max_activation_elements: usize,
}

impl ModelSizeParams {
    /// Create params for a typical 7B parameter model.
    pub fn mistral_7b(max_seq_len: usize) -> Self {
        Self {
            total_weight_params: 7_000_000_000,
            embedding_params: 32_000 * 4096, // vocab_size * hidden_size
            kv_cache_elements: 32 * 2 * 8 * 128 * max_seq_len, // n_layers * 2 * n_kv_heads * head_dim * seq_len
            max_activation_elements: 4096 * 4,                 // hidden_size * batch_size
        }
    }

    /// Create params for a 13B parameter model.
    pub fn llama_13b(max_seq_len: usize) -> Self {
        Self {
            total_weight_params: 13_000_000_000,
            embedding_params: 32_000 * 5120,
            kv_cache_elements: 40 * 2 * 40 * 128 * max_seq_len,
            max_activation_elements: 5120 * 4,
        }
    }
}

/// Compute strides for a given shape (row-major order).
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// =============================================================================
// Memory Transfer Traits
// =============================================================================

/// Trait for types that can be transferred between devices.
pub trait DeviceTransfer {
    /// Transfer this tensor to the specified device.
    fn to_device(&self, device: Device) -> anyhow::Result<UnifiedTensor>;

    /// Transfer this tensor to CPU.
    fn to_cpu(&self) -> anyhow::Result<UnifiedTensor> {
        self.to_device(Device::Cpu)
    }
}

impl DeviceTransfer for UnifiedTensor {
    fn to_device(&self, device: Device) -> anyhow::Result<UnifiedTensor> {
        if self.device() == device {
            return Ok(self.clone());
        }
        match device {
            Device::Cpu => {
                let f32_data = self.storage.to_f32();
                let storage = TensorStorage::from_f32(f32_data);
                Ok(UnifiedTensor::new(storage, self.shape.clone()))
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(device_idx) => {
                use crate::kernels::cuda::CudaBackend;
                let backend = CudaBackend::with_device(device_idx)?;
                let f32_data = self.storage.to_f32();
                let tensor = if self.ndim() == 1 {
                    let arr = ndarray::Array1::from_vec(f32_data);
                    backend.to_device_1d(&arr)?
                } else {
                    let size: usize = self.shape[1..].iter().product();
                    let arr = ndarray::Array2::from_shape_vec((self.shape[0], size), f32_data)?;
                    backend.to_device_2d(&arr)?
                };
                let storage = TensorStorage::CudaF32(CudaF32Storage {
                    tensor,
                    device: backend.device().clone(),
                });
                Ok(UnifiedTensor::new(storage, self.shape.clone()))
            }
            #[cfg(feature = "metal-gpu")]
            Device::Metal => {
                use crate::kernels::metal::MetalBackend;
                let backend = MetalBackend::new()?;
                let f32_data = self.storage.to_f32();
                let tensor = if self.ndim() == 1 {
                    let arr = ndarray::Array1::from_vec(f32_data);
                    backend.to_device_1d(&arr)?
                } else {
                    let size: usize = self.shape[1..].iter().product();
                    let arr = ndarray::Array2::from_shape_vec((self.shape[0], size), f32_data)?;
                    backend.to_device_2d(&arr)?
                };
                let storage = TensorStorage::MetalF32(MetalF32Storage { tensor });
                Ok(UnifiedTensor::new(storage, self.shape.clone()))
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(device_idx) => {
                use crate::kernels::rocm::RocmBackend;
                let backend = RocmBackend::with_device(device_idx)?;
                let f32_data = self.storage.to_f32();
                let tensor = if self.ndim() == 1 {
                    let arr = ndarray::Array1::from_vec(f32_data);
                    backend.to_device_1d(&arr)?
                } else {
                    let size: usize = self.shape[1..].iter().product();
                    let arr = ndarray::Array2::from_shape_vec((self.shape[0], size), f32_data)?;
                    backend.to_device_2d(&arr)?
                };
                let storage = TensorStorage::RocmF32(RocmF32Storage {
                    tensor,
                    backend: std::sync::Arc::new(backend),
                });
                Ok(UnifiedTensor::new(storage, self.shape.clone()))
            }
            #[cfg(feature = "opencl")]
            Device::OpenCL(device_idx) => {
                use crate::kernels::opencl::OpenCLBackend;
                let backend = OpenCLBackend::with_device_index(device_idx)?;
                let f32_data = self.storage.to_f32();
                let tensor = if self.ndim() == 1 {
                    let arr = ndarray::Array1::from_vec(f32_data);
                    backend.to_device_1d(&arr)?
                } else {
                    let size: usize = self.shape[1..].iter().product();
                    let arr = ndarray::Array2::from_shape_vec((self.shape[0], size), f32_data)?;
                    backend.to_device_2d(&arr)?
                };
                let storage = TensorStorage::OpenCLF32(OpenCLF32Storage { tensor });
                Ok(UnifiedTensor::new(storage, self.shape.clone()))
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(Dtype::F32.element_size(), 4);
        assert_eq!(Dtype::F16.element_size(), 2);
        assert_eq!(Dtype::Int8.element_size(), 1);
    }

    #[test]
    fn test_f32_storage() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let storage = TensorStorage::from_f32(data.clone());

        assert_eq!(storage.dtype(), Dtype::F32);
        assert_eq!(storage.device(), Device::Cpu);
        assert_eq!(storage.len(), 4);
        assert_eq!(storage.to_f32(), data);
    }

    #[test]
    fn test_f16_storage() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let storage = TensorStorage::from_f32_as_f16(data.clone());

        assert_eq!(storage.dtype(), Dtype::F16);
        assert_eq!(storage.len(), 4);

        // Check roundtrip (may have small precision loss)
        let recovered = storage.to_f32();
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_int8_quantization() {
        let data = vec![0.5, 1.0, -0.5, -1.0, 0.25, 0.75, -0.25, -0.75];
        let storage = TensorStorage::from_f32_quantize_int8(data.clone(), 4);

        assert_eq!(storage.dtype(), Dtype::Int8);
        assert_eq!(storage.len(), 8);

        // Check dequantization (should be close to original)
        let recovered = storage.to_f32();
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.02, "Expected {} but got {}", a, b);
        }
    }

    #[test]
    fn test_int4_quantization() {
        let data = vec![0.5, 1.0, -0.5, -1.0];
        let storage = TensorStorage::from_f32_quantize_int4(data.clone(), 4);

        assert_eq!(storage.dtype(), Dtype::Int4);

        // Check dequantization (lower precision expected)
        let recovered = storage.to_f32();
        for (a, b) in data.iter().zip(recovered.iter().take(4)) {
            assert!((a - b).abs() < 0.2, "Expected {} but got {}", a, b);
        }
    }

    #[test]
    fn test_unified_tensor() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = UnifiedTensor::from_f32(data, vec![2, 3]);

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.nrows(), 2);
        assert_eq!(tensor.ncols(), 3);
    }

    #[test]
    fn test_tensor_to_array2() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = UnifiedTensor::from_f32(data, vec![2, 3]);

        let arr = tensor.to_array2();
        assert_eq!(arr.dim(), (2, 3));
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 2]], 6.0);
    }

    #[test]
    fn test_memory_bytes() {
        let data = vec![1.0, 2.0, 3.0, 4.0];

        let f32_tensor = UnifiedTensor::from_f32(data.clone(), vec![4]);
        assert_eq!(f32_tensor.memory_bytes(), 16); // 4 * 4 bytes

        let f16_tensor = UnifiedTensor::from_f32_as_f16(data.clone(), vec![4]);
        assert_eq!(f16_tensor.memory_bytes(), 8); // 4 * 2 bytes
    }

    // GPU device transfer round-trip tests. Run with e.g.:
    //   cargo test --features cuda test_device_transfer_cuda -- --ignored
    // These are ignored by default so that `cargo test` does not require GPU hardware.

    #[test]
    #[cfg(feature = "cuda")]
    #[ignore = "requires CUDA GPU; run with cargo test --features cuda -- --ignored"]
    fn test_device_transfer_cuda() {
        use super::DeviceTransfer;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = UnifiedTensor::from_f32(data.clone(), vec![4]);

        let gpu_tensor = tensor.to_device(Device::Cuda(0)).unwrap();
        assert_eq!(gpu_tensor.device(), Device::Cuda(0));

        let cpu_tensor = gpu_tensor.to_cpu().unwrap();
        assert_eq!(cpu_tensor.storage.to_f32(), data);
    }

    #[test]
    #[cfg(feature = "metal-gpu")]
    #[ignore = "requires Metal GPU; run with cargo test --features metal-gpu -- --ignored"]
    fn test_device_transfer_metal() {
        use super::DeviceTransfer;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = UnifiedTensor::from_f32(data.clone(), vec![4]);

        let gpu_tensor = tensor.to_device(Device::Metal).unwrap();
        assert_eq!(gpu_tensor.device(), Device::Metal);

        let cpu_tensor = gpu_tensor.to_cpu().unwrap();
        assert_eq!(cpu_tensor.storage.to_f32(), data);
    }

    #[test]
    #[cfg(feature = "rocm")]
    #[ignore = "requires ROCm GPU; run with cargo test --features rocm -- --ignored"]
    fn test_device_transfer_rocm() {
        use super::DeviceTransfer;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = UnifiedTensor::from_f32(data.clone(), vec![4]);

        let gpu_tensor = tensor.to_device(Device::Rocm(0)).unwrap();
        assert_eq!(gpu_tensor.device(), Device::Rocm(0));

        let cpu_tensor = gpu_tensor.to_cpu().unwrap();
        assert_eq!(cpu_tensor.storage.to_f32(), data);
    }

    #[test]
    #[cfg(feature = "opencl")]
    #[ignore = "requires OpenCL; run with cargo test --features opencl -- --ignored"]
    fn test_device_transfer_opencl() {
        use super::DeviceTransfer;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = UnifiedTensor::from_f32(data.clone(), vec![4]);

        let gpu_tensor = tensor.to_device(Device::OpenCL(0)).unwrap();
        assert_eq!(gpu_tensor.device(), Device::OpenCL(0));

        let cpu_tensor = gpu_tensor.to_cpu().unwrap();
        assert_eq!(cpu_tensor.storage.to_f32(), data);
    }
}
