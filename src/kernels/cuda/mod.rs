//! CUDA Backend for NVIDIA GPUs
//!
//! This module provides GPU-accelerated inference using NVIDIA CUDA via the cudarc crate.
//! It implements the `KernelBackend` trait for seamless integration with the rest of the system.
//!
//! # Features
//!
//! - cuBLAS integration for optimized matrix operations
//! - Custom CUDA kernels for activation functions (SiLU, RMSNorm, softmax)
//! - GPU memory management with buffer pooling
//! - Asynchronous execution with CUDA streams
//! - CPU-GPU data transfer utilities
//!
//! # Usage
//!
//! ```ignore
//! use torchless::kernels::cuda::CudaBackend;
//!
//! // Check if CUDA is available
//! if CudaBackend::is_available() {
//!     let backend = CudaBackend::new()?;
//!     println!("Using CUDA device: {}", backend.device_name());
//! }
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      CudaBackend                            │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │ CudaDevice   │  │ CudaStream   │  │ CublasHandle │     │
//! │  └──────────────┘  └──────────────┘  └──────────────┘     │
//! │          │                │                 │              │
//! │          └────────────────┼─────────────────┘              │
//! │                           ▼                                │
//! │  ┌──────────────────────────────────────────────────────┐ │
//! │  │                  CudaKernels                          │ │
//! │  │  - RMSNorm      - Softmax       - SiLU               │ │
//! │  │  - RoPE         - Attention     - Weighted Sum       │ │
//! │  └──────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use std::sync::Arc;

pub mod kernels;
pub mod memory;
pub mod tensor;

pub use memory::CudaMemoryPool;
pub use tensor::CudaTensor;

use crate::kernels::backend::KernelBackend;
use crate::tensor::{Tensor1, Tensor2};

// =============================================================================
// Error Handling
// =============================================================================

/// Convert cudarc driver errors to anyhow errors
fn driver_error_to_anyhow(e: cudarc::driver::DriverError) -> anyhow::Error {
    anyhow::anyhow!("CUDA driver error: {:?}", e)
}

/// Convert cudarc cublas errors to anyhow errors
fn cublas_error_to_anyhow(e: cudarc::cublas::result::CublasError) -> anyhow::Error {
    anyhow::anyhow!("cuBLAS error: {:?}", e)
}

/// Convert cudarc nvrtc errors to anyhow errors
fn compile_error_to_anyhow(e: cudarc::nvrtc::CompileError) -> anyhow::Error {
    anyhow::anyhow!("CUDA compile error: {:?}", e)
}

// =============================================================================
// CUDA Backend Implementation
// =============================================================================

/// CUDA backend for NVIDIA GPUs.
///
/// This backend uses cuBLAS for matrix operations and custom CUDA kernels
/// for other operations like RMSNorm, softmax, and SiLU.
#[derive(Clone)]
pub struct CudaBackend {
    /// CUDA device handle
    device: Arc<CudaDevice>,
    /// cuBLAS handle for matrix operations
    cublas: Arc<CudaBlas>,
    /// Compiled CUDA kernels
    kernels: Arc<CudaKernels>,
    /// Memory pool for buffer reuse (for future optimization)
    #[allow(dead_code)]
    memory_pool: Arc<std::sync::Mutex<CudaMemoryPool>>,
    /// Device index
    device_index: usize,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device_index", &self.device_index)
            .finish()
    }
}

/// Container for compiled CUDA kernels
struct CudaKernels {
    /// RMSNorm kernel
    rmsnorm: cudarc::driver::CudaFunction,
    /// Softmax kernel  
    softmax: cudarc::driver::CudaFunction,
    /// SiLU activation kernel
    silu: cudarc::driver::CudaFunction,
    /// RoPE (Rotary Position Embedding) kernel
    rope: cudarc::driver::CudaFunction,
    /// Attention scores kernel (Q @ K^T)
    attention_scores: cudarc::driver::CudaFunction,
    /// Weighted sum kernel (softmax @ V)
    weighted_sum: cudarc::driver::CudaFunction,
    /// Element-wise multiply kernel (for future use)
    #[allow(dead_code)]
    elementwise_mul: cudarc::driver::CudaFunction,
    /// Element-wise add kernel (for future use)
    #[allow(dead_code)]
    elementwise_add: cudarc::driver::CudaFunction,
}

impl CudaBackend {
    /// Check if CUDA is available on this system.
    pub fn is_available() -> bool {
        CudaDevice::new(0).is_ok()
    }

    /// Get the number of available CUDA devices.
    pub fn device_count() -> usize {
        cudarc::driver::result::device::get_count().unwrap_or(0) as usize
    }

    /// Create a new CUDA backend on device 0.
    pub fn new() -> anyhow::Result<Self> {
        Self::with_device(0)
    }

    /// Create a new CUDA backend on the specified device.
    pub fn with_device(device_index: usize) -> anyhow::Result<Self> {
        let device = CudaDevice::new(device_index).map_err(driver_error_to_anyhow)?;

        // Create cuBLAS handle
        let cublas = CudaBlas::new(device.clone()).map_err(cublas_error_to_anyhow)?;
        let cublas = Arc::new(cublas);

        // Compile and load custom kernels
        let kernels = compile_kernels(&device)?;
        let kernels = Arc::new(kernels);

        // Initialize memory pool
        let memory_pool = Arc::new(std::sync::Mutex::new(CudaMemoryPool::new()));

        Ok(Self {
            device,
            cublas,
            kernels,
            memory_pool,
            device_index,
        })
    }

    /// Get the name of the CUDA device.
    pub fn device_name(&self) -> String {
        // Note: cudarc doesn't expose device name directly, return index-based name
        format!("CUDA Device {}", self.device_index)
    }

    /// Get the device index.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Get a reference to the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get a reference to the cuBLAS handle.
    pub fn cublas(&self) -> &Arc<CudaBlas> {
        &self.cublas
    }

    /// Transfer a 1D array from CPU to GPU.
    pub fn to_device_1d(&self, data: &Array1<f32>) -> anyhow::Result<CudaTensor> {
        let slice = data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Array must be contiguous"))?;
        let gpu_data = self
            .device
            .htod_sync_copy(slice)
            .map_err(driver_error_to_anyhow)?;
        Ok(CudaTensor::new_1d(gpu_data, data.len()))
    }

    /// Transfer a 2D array from CPU to GPU.
    pub fn to_device_2d(&self, data: &Array2<f32>) -> anyhow::Result<CudaTensor> {
        let slice = data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Array must be contiguous"))?;
        let (rows, cols) = data.dim();
        let gpu_data = self
            .device
            .htod_sync_copy(slice)
            .map_err(driver_error_to_anyhow)?;
        Ok(CudaTensor::new_2d(gpu_data, rows, cols))
    }

    /// Transfer data from GPU to CPU as 1D array.
    pub fn to_host_1d(&self, tensor: &CudaTensor) -> anyhow::Result<Array1<f32>> {
        let data = self
            .device
            .dtoh_sync_copy(tensor.data())
            .map_err(driver_error_to_anyhow)?;
        Ok(Array1::from_vec(data))
    }

    /// Transfer data from GPU to CPU as 2D array.
    pub fn to_host_2d(&self, tensor: &CudaTensor) -> anyhow::Result<Array2<f32>> {
        let (rows, cols) = tensor
            .shape_2d()
            .ok_or_else(|| anyhow::anyhow!("Tensor is not 2D"))?;
        let data = self
            .device
            .dtoh_sync_copy(tensor.data())
            .map_err(driver_error_to_anyhow)?;
        Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| anyhow::anyhow!("Shape error: {}", e))
    }

    /// Allocate GPU memory for a given size.
    pub fn alloc(&self, len: usize) -> anyhow::Result<CudaSlice<f32>> {
        self.device.alloc_zeros(len).map_err(driver_error_to_anyhow)
    }

    /// Synchronize the device (wait for all operations to complete).
    pub fn synchronize(&self) -> anyhow::Result<()> {
        self.device.synchronize().map_err(driver_error_to_anyhow)
    }

    // =========================================================================
    // GPU Kernel Launches
    // =========================================================================

    /// Launch RMSNorm kernel on GPU.
    fn launch_rmsnorm(
        &self,
        x: &mut CudaTensor,
        weight: &CudaTensor,
        eps: f32,
    ) -> anyhow::Result<()> {
        let n = x.len();
        let block_size: u32 = 256;
        let num_blocks: u32 = 1; // RMSNorm is computed per-vector

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: block_size * std::mem::size_of::<f32>() as u32,
        };

        unsafe {
            self.kernels
                .rmsnorm
                .clone()
                .launch(cfg, (x.data_mut(), weight.data(), n as i32, eps))
                .map_err(driver_error_to_anyhow)?;
        }

        Ok(())
    }

    /// Launch softmax kernel on GPU.
    fn launch_softmax(&self, x: &mut CudaTensor) -> anyhow::Result<()> {
        let n = x.len();
        let block_size: u32 = 256;
        let num_blocks: u32 = 1; // Softmax is computed per-vector

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: block_size * std::mem::size_of::<f32>() as u32 * 2,
        };

        unsafe {
            self.kernels
                .softmax
                .clone()
                .launch(cfg, (x.data_mut(), n as i32))
                .map_err(driver_error_to_anyhow)?;
        }

        Ok(())
    }

    /// Launch SiLU kernel on GPU.
    fn launch_silu(&self, x: &CudaTensor, out: &mut CudaTensor) -> anyhow::Result<()> {
        let n = x.len();
        let block_size: u32 = 256;
        let num_blocks = n.div_ceil(block_size as usize) as u32;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .silu
                .clone()
                .launch(cfg, (x.data(), out.data_mut(), n as i32))
                .map_err(driver_error_to_anyhow)?;
        }

        Ok(())
    }

    /// Launch RoPE kernel on GPU.
    fn launch_rope(
        &self,
        x: &mut CudaTensor,
        cos: &CudaTensor,
        sin: &CudaTensor,
        n_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<()> {
        let half = head_dim / 2;
        let block_size: u32 = 256;
        let total_elements = n_heads * half;
        let num_blocks = total_elements.div_ceil(block_size as usize) as u32;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .rope
                .clone()
                .launch(
                    cfg,
                    (
                        x.data_mut(),
                        cos.data(),
                        sin.data(),
                        n_heads as i32,
                        head_dim as i32,
                        half as i32,
                    ),
                )
                .map_err(driver_error_to_anyhow)?;
        }

        Ok(())
    }

    /// Launch attention scores kernel: scores = Q @ K^T * scale
    fn launch_attention_scores(
        &self,
        query: &CudaTensor,
        keys: &CudaTensor,
        scores: &mut CudaTensor,
        scale: f32,
        seq_len: usize,
        head_dim: usize,
    ) -> anyhow::Result<()> {
        let block_size: u32 = 256;
        let num_blocks = seq_len.div_ceil(block_size as usize) as u32;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .attention_scores
                .clone()
                .launch(
                    cfg,
                    (
                        query.data(),
                        keys.data(),
                        scores.data_mut(),
                        seq_len as i32,
                        head_dim as i32,
                        scale,
                    ),
                )
                .map_err(driver_error_to_anyhow)?;
        }

        Ok(())
    }

    /// Launch weighted sum kernel: out = weights @ matrix
    fn launch_weighted_sum(
        &self,
        weights: &CudaTensor,
        matrix: &CudaTensor,
        out: &mut CudaTensor,
        n: usize,
        d: usize,
    ) -> anyhow::Result<()> {
        let block_size: u32 = 256;
        let num_blocks = d.div_ceil(block_size as usize) as u32;

        let cfg = LaunchConfig {
            block_dim: (block_size, 1, 1),
            grid_dim: (num_blocks, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.kernels
                .weighted_sum
                .clone()
                .launch(
                    cfg,
                    (
                        weights.data(),
                        matrix.data(),
                        out.data_mut(),
                        n as i32,
                        d as i32,
                    ),
                )
                .map_err(driver_error_to_anyhow)?;
        }

        Ok(())
    }
}

impl KernelBackend for CudaBackend {
    type Tensor1 = Tensor1;
    type Tensor2 = Tensor2;

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn is_available() -> bool {
        CudaBackend::is_available()
    }

    fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        // Use cuBLAS GEMV for matrix-vector multiplication
        // y = alpha * A * x + beta * y
        // where A is m x n, x is n, y is m

        let (m, n) = w.dim();

        // Transfer to GPU
        let w_gpu = self
            .to_device_2d(w)
            .expect("Failed to transfer weights to GPU");
        let x_gpu = self
            .to_device_1d(x)
            .expect("Failed to transfer input to GPU");
        let mut y_gpu: CudaSlice<f32> = self
            .device
            .alloc_zeros(m)
            .expect("Failed to allocate output");

        // cuBLAS uses column-major, but our arrays are row-major
        // For row-major A @ x, we compute A^T @ x in column-major
        unsafe {
            use cudarc::cublas::sys::cublasOperation_t;

            let cfg = GemvConfig {
                trans: cublasOperation_t::CUBLAS_OP_T, // Transpose because row-major
                m: n as i32,                           // rows of A^T = cols of A
                n: m as i32,                           // cols of A^T = rows of A
                alpha: 1.0f32,
                lda: n as i32, // leading dimension (columns in row-major)
                incx: 1,       // stride of x
                beta: 0.0f32,
                incy: 1, // stride of y
            };

            self.cublas
                .gemv(cfg, w_gpu.data(), x_gpu.data(), &mut y_gpu)
                .expect("cuBLAS GEMV failed");
        }

        // Transfer back to CPU
        let result = self
            .device
            .dtoh_sync_copy(&y_gpu)
            .expect("Failed to transfer result to CPU");
        Array1::from_vec(result)
    }

    fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
        let result = self.matmul_vec(w, x);
        out.assign(&result);
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        // Use cuBLAS GEMM for matrix-matrix multiplication
        // C = alpha * op(A) * op(B) + beta * C

        let (m, k1) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k1, k2, "Matrix dimensions must match for multiplication");
        let k = k1;

        // Transfer to GPU
        let a_gpu = self.to_device_2d(a).expect("Failed to transfer A to GPU");
        let b_gpu = self.to_device_2d(b).expect("Failed to transfer B to GPU");
        let mut c_gpu: CudaSlice<f32> = self
            .device
            .alloc_zeros(m * n)
            .expect("Failed to allocate output");

        unsafe {
            use cudarc::cublas::sys::cublasOperation_t;

            // For row-major matrices, we need to compute appropriately
            // cuBLAS expects column-major, so we swap the order: C^T = B^T @ A^T
            // Then reading C row-major gives us A @ B
            let cfg = GemmConfig {
                transa: cublasOperation_t::CUBLAS_OP_N,
                transb: cublasOperation_t::CUBLAS_OP_N,
                m: n as i32,
                n: m as i32,
                k: k as i32,
                alpha: 1.0f32,
                lda: n as i32,
                ldb: k as i32,
                beta: 0.0f32,
                ldc: n as i32,
            };

            self.cublas
                .gemm(cfg, b_gpu.data(), a_gpu.data(), &mut c_gpu)
                .expect("cuBLAS GEMM failed");
        }

        // Transfer back to CPU
        let result = self
            .device
            .dtoh_sync_copy(&c_gpu)
            .expect("Failed to transfer result to CPU");
        Array2::from_shape_vec((m, n), result).expect("Shape mismatch")
    }

    fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
        let mut x_gpu = self.to_device_1d(x).expect("Failed to transfer x to GPU");
        let weight_gpu = self
            .to_device_1d(weight)
            .expect("Failed to transfer weight to GPU");

        self.launch_rmsnorm(&mut x_gpu, &weight_gpu, eps)
            .expect("RMSNorm kernel failed");

        let result = self
            .to_host_1d(&x_gpu)
            .expect("Failed to transfer result to CPU");
        x.assign(&result);
    }

    fn softmax(&self, x: &mut Array1<f32>) {
        let mut x_gpu = self.to_device_1d(x).expect("Failed to transfer x to GPU");

        self.launch_softmax(&mut x_gpu)
            .expect("Softmax kernel failed");

        let result = self
            .to_host_1d(&x_gpu)
            .expect("Failed to transfer result to CPU");
        x.assign(&result);
    }

    fn softmax_view(&self, x: &mut ArrayViewMut1<f32>) {
        // For views, we need to copy to owned array, process, and copy back
        let mut owned = x.to_owned();
        self.softmax(&mut owned);
        x.assign(&owned);
    }

    fn silu(&self, x: &Array1<f32>) -> Array1<f32> {
        let x_gpu = self.to_device_1d(x).expect("Failed to transfer x to GPU");
        let mut out_gpu = CudaTensor::new_1d(
            self.device
                .alloc_zeros(x.len())
                .expect("Failed to allocate output"),
            x.len(),
        );

        self.launch_silu(&x_gpu, &mut out_gpu)
            .expect("SiLU kernel failed");

        self.to_host_1d(&out_gpu)
            .expect("Failed to transfer result to CPU")
    }

    fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
        let (n_heads, head_dim) = x.dim();

        let mut x_gpu = self.to_device_2d(x).expect("Failed to transfer x to GPU");
        let cos_gpu = self
            .to_device_1d(cos)
            .expect("Failed to transfer cos to GPU");
        let sin_gpu = self
            .to_device_1d(sin)
            .expect("Failed to transfer sin to GPU");

        self.launch_rope(&mut x_gpu, &cos_gpu, &sin_gpu, n_heads, head_dim)
            .expect("RoPE kernel failed");

        let result = self
            .to_host_2d(&x_gpu)
            .expect("Failed to transfer result to CPU");
        x.assign(&result);
    }

    fn compute_attention_scores(
        &self,
        query: ArrayView1<f32>,
        keys: ArrayView2<f32>,
        scores: &mut ArrayViewMut1<f32>,
        scale: f32,
    ) {
        let query_owned = query.to_owned();
        let keys_owned = keys.to_owned();
        let seq_len = keys.nrows();
        let head_dim = keys.ncols();

        let query_gpu = self
            .to_device_1d(&query_owned)
            .expect("Failed to transfer query to GPU");
        let keys_gpu = self
            .to_device_2d(&keys_owned)
            .expect("Failed to transfer keys to GPU");
        let mut scores_gpu = CudaTensor::new_1d(
            self.device
                .alloc_zeros(seq_len)
                .expect("Failed to allocate scores"),
            seq_len,
        );

        self.launch_attention_scores(
            &query_gpu,
            &keys_gpu,
            &mut scores_gpu,
            scale,
            seq_len,
            head_dim,
        )
        .expect("Attention scores kernel failed");

        let result = self
            .to_host_1d(&scores_gpu)
            .expect("Failed to transfer result to CPU");
        scores.assign(&result);
    }

    fn weighted_sum_rows(
        &self,
        weights: ArrayView1<f32>,
        matrix: ArrayView2<f32>,
        out: &mut ArrayViewMut1<f32>,
    ) {
        let weights_owned = weights.to_owned();
        let matrix_owned = matrix.to_owned();
        let (n, d) = matrix.dim();

        let weights_gpu = self
            .to_device_1d(&weights_owned)
            .expect("Failed to transfer weights to GPU");
        let matrix_gpu = self
            .to_device_2d(&matrix_owned)
            .expect("Failed to transfer matrix to GPU");
        let mut out_gpu = CudaTensor::new_1d(
            self.device
                .alloc_zeros(d)
                .expect("Failed to allocate output"),
            d,
        );

        self.launch_weighted_sum(&weights_gpu, &matrix_gpu, &mut out_gpu, n, d)
            .expect("Weighted sum kernel failed");

        let result = self
            .to_host_1d(&out_gpu)
            .expect("Failed to transfer result to CPU");
        out.assign(&result);
    }
}

// =============================================================================
// Kernel Compilation
// =============================================================================

/// CUDA kernel source code
const CUDA_KERNELS_SOURCE: &str = r#"
// RMSNorm kernel
// x = x * weight / sqrt(mean(x^2) + eps)
extern "C" __global__ void rmsnorm_kernel(float* x, const float* weight, int n, float eps) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Compute sum of squares in parallel
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float val = x[i];
        local_sum += val * val;
    }
    
    // Store in shared memory
    shared[tid] = local_sum;
    __syncthreads();
    
    // Reduction to compute total sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute RMS
    float rms = sqrtf(shared[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;
    __syncthreads();
    
    // Apply normalization and weight
    for (int i = tid; i < n; i += block_size) {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

// Softmax kernel with numerical stability
extern "C" __global__ void softmax_kernel(float* x, int n) {
    extern __shared__ float shared[];
    float* max_shared = shared;
    float* sum_shared = &shared[blockDim.x];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Find max in parallel
    float local_max = -INFINITY;
    for (int i = tid; i < n; i += block_size) {
        local_max = fmaxf(local_max, x[i]);
    }
    max_shared[tid] = local_max;
    __syncthreads();
    
    // Reduction for max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = max_shared[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float exp_val = expf(x[i] - max_val);
        x[i] = exp_val;
        local_sum += exp_val;
    }
    sum_shared[tid] = local_sum;
    __syncthreads();
    
    // Reduction for sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }
    float sum = sum_shared[0];
    float inv_sum = 1.0f / sum;
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < n; i += block_size) {
        x[i] *= inv_sum;
    }
}

// SiLU activation kernel: y = x * sigmoid(x) = x / (1 + exp(-x))
extern "C" __global__ void silu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + expf(-val));
    }
}

// RoPE (Rotary Position Embedding) kernel
// Uses half-split layout: rotate dim i with dim i + head_dim/2
extern "C" __global__ void rope_kernel(
    float* x,           // [n_heads, head_dim]
    const float* cos,   // [half]
    const float* sin,   // [half]
    int n_heads,
    int head_dim,
    int half
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * half;
    
    if (idx < total) {
        int h = idx / half;      // head index
        int i = idx % half;      // position within half
        
        int base = h * head_dim;
        float xi = x[base + i];
        float yi = x[base + i + half];
        float c = cos[i];
        float s = sin[i];
        
        // Rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
        x[base + i] = xi * c - yi * s;
        x[base + i + half] = xi * s + yi * c;
    }
}

// Attention scores kernel: scores[i] = keys[i, :].dot(query) * scale
extern "C" __global__ void attention_scores_kernel(
    const float* query,   // [head_dim]
    const float* keys,    // [seq_len, head_dim]
    float* scores,        // [seq_len]
    int seq_len,
    int head_dim,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < seq_len) {
        float dot = 0.0f;
        const float* key_row = keys + i * head_dim;
        for (int j = 0; j < head_dim; j++) {
            dot += query[j] * key_row[j];
        }
        scores[i] = dot * scale;
    }
}

// Weighted sum kernel: out[j] = sum_i(weights[i] * matrix[i, j])
extern "C" __global__ void weighted_sum_kernel(
    const float* weights,  // [n]
    const float* matrix,   // [n, d]
    float* out,            // [d]
    int n,
    int d
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < d) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += weights[i] * matrix[i * d + j];
        }
        out[j] = sum;
    }
}

// Element-wise multiply kernel
extern "C" __global__ void elementwise_mul_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Element-wise add kernel
extern "C" __global__ void elementwise_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
"#;

/// Compile CUDA kernels and load them into the device.
fn compile_kernels(device: &Arc<CudaDevice>) -> anyhow::Result<CudaKernels> {
    // Compile CUDA source to PTX
    let ptx = compile_ptx(CUDA_KERNELS_SOURCE).map_err(compile_error_to_anyhow)?;

    // Load module into device
    device
        .load_ptx(
            ptx,
            "torchless_kernels",
            &[
                "rmsnorm_kernel",
                "softmax_kernel",
                "silu_kernel",
                "rope_kernel",
                "attention_scores_kernel",
                "weighted_sum_kernel",
                "elementwise_mul_kernel",
                "elementwise_add_kernel",
            ],
        )
        .map_err(driver_error_to_anyhow)?;

    // Get function handles
    let rmsnorm = device
        .get_func("torchless_kernels", "rmsnorm_kernel")
        .ok_or_else(|| anyhow::anyhow!("Failed to get rmsnorm_kernel"))?;
    let softmax = device
        .get_func("torchless_kernels", "softmax_kernel")
        .ok_or_else(|| anyhow::anyhow!("Failed to get softmax_kernel"))?;
    let silu = device
        .get_func("torchless_kernels", "silu_kernel")
        .ok_or_else(|| anyhow::anyhow!("Failed to get silu_kernel"))?;
    let rope = device
        .get_func("torchless_kernels", "rope_kernel")
        .ok_or_else(|| anyhow::anyhow!("Failed to get rope_kernel"))?;
    let attention_scores = device
        .get_func("torchless_kernels", "attention_scores_kernel")
        .ok_or_else(|| anyhow::anyhow!("Failed to get attention_scores_kernel"))?;
    let weighted_sum = device
        .get_func("torchless_kernels", "weighted_sum_kernel")
        .ok_or_else(|| anyhow::anyhow!("Failed to get weighted_sum_kernel"))?;
    let elementwise_mul = device
        .get_func("torchless_kernels", "elementwise_mul_kernel")
        .ok_or_else(|| anyhow::anyhow!("Failed to get elementwise_mul_kernel"))?;
    let elementwise_add = device
        .get_func("torchless_kernels", "elementwise_add_kernel")
        .ok_or_else(|| anyhow::anyhow!("Failed to get elementwise_add_kernel"))?;

    Ok(CudaKernels {
        rmsnorm,
        softmax,
        silu,
        rope,
        attention_scores,
        weighted_sum,
        elementwise_mul,
        elementwise_add,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test just checks the availability function doesn't panic
        let available = CudaBackend::is_available();
        println!("CUDA available: {}", available);
    }

    #[test]
    #[ignore] // Only run when CUDA is available
    fn test_cuda_backend_creation() {
        if !CudaBackend::is_available() {
            return;
        }

        let backend = CudaBackend::new().expect("Failed to create CUDA backend");
        assert_eq!(backend.name(), "cuda");
    }

    #[test]
    #[ignore] // Only run when CUDA is available
    fn test_cuda_matmul_vec() {
        if !CudaBackend::is_available() {
            return;
        }

        let backend = CudaBackend::new().expect("Failed to create CUDA backend");

        let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = backend.matmul_vec(&w, &x);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-5); // 1*1 + 2*2 + 3*3 = 14
        assert!((result[1] - 32.0).abs() < 1e-5); // 4*1 + 5*2 + 6*3 = 32
    }

    #[test]
    #[ignore] // Only run when CUDA is available
    fn test_cuda_softmax() {
        if !CudaBackend::is_available() {
            return;
        }

        let backend = CudaBackend::new().expect("Failed to create CUDA backend");

        let mut x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        backend.softmax(&mut x);

        // Check that softmax sums to 1
        let sum: f32 = x.sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check that larger inputs have larger probabilities
        assert!(x[2] > x[1]);
        assert!(x[1] > x[0]);
    }

    #[test]
    #[ignore] // Only run when CUDA is available
    fn test_cuda_silu() {
        if !CudaBackend::is_available() {
            return;
        }

        let backend = CudaBackend::new().expect("Failed to create CUDA backend");

        let x = Array1::from_vec(vec![0.0, 1.0, -1.0]);
        let result = backend.silu(&x);

        assert_eq!(result.len(), 3);
        // SiLU(0) = 0
        assert!(result[0].abs() < 1e-5);
        // SiLU(1) ≈ 0.731
        assert!((result[1] - 0.731).abs() < 0.01);
        // SiLU(-1) ≈ -0.269
        assert!((result[2] - (-0.269)).abs() < 0.01);
    }
}
