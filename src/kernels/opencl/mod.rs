//! OpenCL Backend for Cross-Platform GPU Support
//!
//! This module provides GPU-accelerated inference using the OpenCL framework.
//! It implements the `KernelBackend` trait for seamless integration with the rest of the system.
//!
//! # Features
//!
//! - Cross-platform GPU support (NVIDIA, AMD, Intel, and other OpenCL devices)
//! - Custom OpenCL compute kernels for activation functions (SiLU, RMSNorm, softmax)
//! - GPU memory management with buffer pooling
//! - Works as a fallback when vendor-specific APIs are unavailable
//!
//! # Usage
//!
//! ```ignore
//! use torchless::kernels::opencl::OpenCLBackend;
//!
//! // Check if OpenCL is available
//! if OpenCLBackend::is_available() {
//!     let backend = OpenCLBackend::new()?;
//!     println!("Using OpenCL device: {}", backend.device_name());
//! }
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      OpenCLBackend                          │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │ Context      │  │ Queue        │  │ Program      │     │
//! │  └──────────────┘  └──────────────┘  └──────────────┘     │
//! │          │                │                 │              │
//! │          └────────────────┼─────────────────┘              │
//! │                           ▼                                │
//! │  ┌──────────────────────────────────────────────────────┐ │
//! │  │                  OpenCL Compute Kernels              │ │
//! │  │  - RMSNorm      - Softmax       - SiLU               │ │
//! │  │  - RoPE         - Attention     - Weighted Sum       │ │
//! │  └──────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Requirements
//!
//! - OpenCL 1.2+ compatible GPU and driver
//! - OpenCL runtime (vendor-specific: NVIDIA, AMD, Intel, etc.)

pub mod kernels;
pub mod memory;
pub mod tensor;

pub use memory::OpenCLMemoryPool;
pub use tensor::OpenCLTensor;

use crate::kernels::backend::KernelBackend;
use crate::tensor::{Tensor1, Tensor2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};
use std::sync::Arc;

// =============================================================================
// OpenCL Backend Implementation
// =============================================================================

/// OpenCL backend for cross-platform GPU support.
///
/// This backend uses OpenCL compute kernels for transformer operations and
/// works on any GPU with OpenCL support, including NVIDIA, AMD, and Intel devices.
#[derive(Clone)]
pub struct OpenCLBackend {
    /// OpenCL context (kept for lifetime management)
    #[allow(dead_code)]
    context: Arc<Context>,
    /// Command queue for GPU operations
    queue: Arc<Queue>,
    /// Compiled program with all kernels
    program: Arc<Program>,
    /// Memory pool for buffer reuse
    #[allow(dead_code)]
    memory_pool: Arc<std::sync::Mutex<OpenCLMemoryPool>>,
    /// Device name
    device_name: String,
    /// Platform name
    platform_name: String,
    /// Device index
    device_index: usize,
}

impl std::fmt::Debug for OpenCLBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenCLBackend")
            .field("device_name", &self.device_name)
            .field("platform_name", &self.platform_name)
            .field("device_index", &self.device_index)
            .finish()
    }
}

impl OpenCLBackend {
    /// Check if OpenCL is available on this system.
    pub fn is_available() -> bool {
        // Try to get any OpenCL platform
        Platform::list().first().is_some()
    }

    /// Get a list of available OpenCL devices.
    pub fn list_devices() -> Vec<(String, String, usize)> {
        let mut devices = Vec::new();
        for (platform_idx, platform) in Platform::list().iter().enumerate() {
            if let Ok(platform_name) = platform.name() {
                if let Ok(device_list) = Device::list_all(platform) {
                    for (device_idx, device) in device_list.iter().enumerate() {
                        if let Ok(device_name) = device.name() {
                            devices.push((platform_name.clone(), device_name, platform_idx * 100 + device_idx));
                        }
                    }
                }
            }
        }
        devices
    }

    /// Create a new OpenCL backend using the first available GPU device.
    pub fn new() -> anyhow::Result<Self> {
        Self::with_device_index(0)
    }

    /// Create a new OpenCL backend using a specific device index.
    pub fn with_device_index(device_index: usize) -> anyhow::Result<Self> {
        // Find all devices across all platforms
        let mut all_devices: Vec<(Platform, Device)> = Vec::new();
        for platform in Platform::list() {
            if let Ok(devices) = Device::list_all(&platform) {
                for device in devices {
                    all_devices.push((platform.clone(), device));
                }
            }
        }

        if all_devices.is_empty() {
            anyhow::bail!("No OpenCL devices available");
        }

        let idx = device_index.min(all_devices.len() - 1);
        let (platform, device) = &all_devices[idx];

        let platform_name = platform.name()
            .unwrap_or_else(|_| "Unknown Platform".to_string());
        let device_name = device.name()
            .unwrap_or_else(|_| "Unknown Device".to_string());

        // Create context and queue
        let context = Context::builder()
            .platform(platform.clone())
            .devices(device.clone())
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create OpenCL context: {}", e))?;

        let queue = Queue::new(&context, device.clone(), None)
            .map_err(|e| anyhow::anyhow!("Failed to create OpenCL queue: {}", e))?;

        // Compile kernel program
        let program = Program::builder()
            .src(kernels::OPENCL_KERNELS_SOURCE)
            .devices(device.clone())
            .build(&context)
            .map_err(|e| anyhow::anyhow!("Failed to compile OpenCL kernels: {}", e))?;

        // Initialize memory pool
        let memory_pool = Arc::new(std::sync::Mutex::new(OpenCLMemoryPool::new()));

        Ok(Self {
            context: Arc::new(context),
            queue: Arc::new(queue),
            program: Arc::new(program),
            memory_pool,
            device_name,
            platform_name,
            device_index: idx,
        })
    }

    /// Get the name of the OpenCL device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get the name of the OpenCL platform.
    pub fn platform_name(&self) -> &str {
        &self.platform_name
    }

    /// Get a reference to the command queue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Create a buffer from CPU data.
    pub fn create_buffer(&self, data: &[f32]) -> anyhow::Result<Buffer<f32>> {
        memory::create_buffer_with_data(&self.queue, data)
    }

    /// Create a zero-initialized buffer.
    pub fn create_buffer_zeros(&self, len: usize) -> anyhow::Result<Buffer<f32>> {
        memory::create_buffer_zeros(&self.queue, len)
    }

    /// Transfer a 1D array to GPU buffer.
    pub fn to_device_1d(&self, data: &Array1<f32>) -> anyhow::Result<OpenCLTensor> {
        let slice = data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Array must be contiguous"))?;
        let buffer = self.create_buffer(slice)?;
        Ok(OpenCLTensor::new_1d(buffer, data.len()))
    }

    /// Transfer a 2D array to GPU buffer.
    pub fn to_device_2d(&self, data: &Array2<f32>) -> anyhow::Result<OpenCLTensor> {
        let slice = data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Array must be contiguous"))?;
        let (rows, cols) = data.dim();
        let buffer = self.create_buffer(slice)?;
        Ok(OpenCLTensor::new_2d(buffer, rows, cols))
    }

    /// Transfer data from GPU to CPU as 1D array.
    pub fn to_host_1d(&self, tensor: &OpenCLTensor) -> anyhow::Result<Array1<f32>> {
        let data = tensor.to_vec()?;
        Ok(Array1::from_vec(data))
    }

    /// Transfer data from GPU to CPU as 2D array.
    pub fn to_host_2d(&self, tensor: &OpenCLTensor) -> anyhow::Result<Array2<f32>> {
        let (rows, cols) = tensor
            .shape_2d()
            .ok_or_else(|| anyhow::anyhow!("Tensor is not 2D"))?;
        let data = tensor.to_vec()?;
        Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| anyhow::anyhow!("Shape error: {}", e))
    }

    /// Wait for all GPU operations to complete.
    pub fn synchronize(&self) -> anyhow::Result<()> {
        self.queue.finish()
            .map_err(|e| anyhow::anyhow!("Failed to synchronize OpenCL queue: {}", e))
    }

    // =========================================================================
    // GPU Kernel Launches
    // =========================================================================

    /// Launch RMSNorm kernel on GPU.
    fn launch_rmsnorm(&self, x: &Buffer<f32>, weight: &Buffer<f32>, n: usize, eps: f32) -> anyhow::Result<()> {
        let block_size = 256.max(n.next_power_of_two()).min(1024);

        let kernel = Kernel::builder()
            .program(&self.program)
            .name(kernels::kernel_names::RMSNORM)
            .queue((*self.queue).clone())
            .global_work_size(block_size)
            .local_work_size(block_size)
            .arg(x)
            .arg(weight)
            .arg(&(n as i32))
            .arg(&eps)
            .arg_local::<f32>(block_size)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build rmsnorm kernel: {}", e))?;

        unsafe {
            kernel.enq()
                .map_err(|e| anyhow::anyhow!("Failed to execute rmsnorm kernel: {}", e))?;
        }
        Ok(())
    }

    /// Launch softmax kernel on GPU.
    fn launch_softmax(&self, x: &Buffer<f32>, n: usize) -> anyhow::Result<()> {
        let block_size = 256.max(n.next_power_of_two()).min(1024);

        let kernel = Kernel::builder()
            .program(&self.program)
            .name(kernels::kernel_names::SOFTMAX)
            .queue((*self.queue).clone())
            .global_work_size(block_size)
            .local_work_size(block_size)
            .arg(x)
            .arg(&(n as i32))
            .arg_local::<f32>(block_size * 2) // max_shared + sum_shared
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build softmax kernel: {}", e))?;

        unsafe {
            kernel.enq()
                .map_err(|e| anyhow::anyhow!("Failed to execute softmax kernel: {}", e))?;
        }
        Ok(())
    }

    /// Launch SiLU kernel on GPU.
    fn launch_silu(&self, x: &Buffer<f32>, y: &Buffer<f32>, n: usize) -> anyhow::Result<()> {
        let local_size = 256;
        let global_size = ((n + local_size - 1) / local_size) * local_size;

        let kernel = Kernel::builder()
            .program(&self.program)
            .name(kernels::kernel_names::SILU)
            .queue((*self.queue).clone())
            .global_work_size(global_size)
            .local_work_size(local_size)
            .arg(x)
            .arg(y)
            .arg(&(n as i32))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build silu kernel: {}", e))?;

        unsafe {
            kernel.enq()
                .map_err(|e| anyhow::anyhow!("Failed to execute silu kernel: {}", e))?;
        }
        Ok(())
    }

    /// Launch RoPE kernel on GPU.
    fn launch_rope(
        &self,
        x: &Buffer<f32>,
        cos: &Buffer<f32>,
        sin: &Buffer<f32>,
        n_heads: usize,
        head_dim: usize,
    ) -> anyhow::Result<()> {
        let half = head_dim / 2;
        let total = n_heads * half;
        let local_size = 256;
        let global_size = ((total + local_size - 1) / local_size) * local_size;

        let kernel = Kernel::builder()
            .program(&self.program)
            .name(kernels::kernel_names::ROPE)
            .queue((*self.queue).clone())
            .global_work_size(global_size)
            .local_work_size(local_size)
            .arg(x)
            .arg(cos)
            .arg(sin)
            .arg(&(n_heads as i32))
            .arg(&(head_dim as i32))
            .arg(&(half as i32))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build rope kernel: {}", e))?;

        unsafe {
            kernel.enq()
                .map_err(|e| anyhow::anyhow!("Failed to execute rope kernel: {}", e))?;
        }
        Ok(())
    }

    /// Launch attention scores kernel on GPU.
    fn launch_attention_scores(
        &self,
        query: &Buffer<f32>,
        keys: &Buffer<f32>,
        scores: &Buffer<f32>,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) -> anyhow::Result<()> {
        let local_size = 256;
        let global_size = ((seq_len + local_size - 1) / local_size) * local_size;

        let kernel = Kernel::builder()
            .program(&self.program)
            .name(kernels::kernel_names::ATTENTION_SCORES)
            .queue((*self.queue).clone())
            .global_work_size(global_size)
            .local_work_size(local_size)
            .arg(query)
            .arg(keys)
            .arg(scores)
            .arg(&(seq_len as i32))
            .arg(&(head_dim as i32))
            .arg(&scale)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build attention_scores kernel: {}", e))?;

        unsafe {
            kernel.enq()
                .map_err(|e| anyhow::anyhow!("Failed to execute attention_scores kernel: {}", e))?;
        }
        Ok(())
    }

    /// Launch weighted sum kernel on GPU.
    fn launch_weighted_sum(&self, weights: &Buffer<f32>, matrix: &Buffer<f32>, out: &Buffer<f32>, n: usize, d: usize) -> anyhow::Result<()> {
        let local_size = 256;
        let global_size = ((d + local_size - 1) / local_size) * local_size;

        let kernel = Kernel::builder()
            .program(&self.program)
            .name(kernels::kernel_names::WEIGHTED_SUM)
            .queue((*self.queue).clone())
            .global_work_size(global_size)
            .local_work_size(local_size)
            .arg(weights)
            .arg(matrix)
            .arg(out)
            .arg(&(n as i32))
            .arg(&(d as i32))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build weighted_sum kernel: {}", e))?;

        unsafe {
            kernel.enq()
                .map_err(|e| anyhow::anyhow!("Failed to execute weighted_sum kernel: {}", e))?;
        }
        Ok(())
    }

    /// Launch matrix-vector multiplication kernel on GPU.
    fn launch_matmul_vec(&self, weights: &Buffer<f32>, x: &Buffer<f32>, out: &Buffer<f32>, rows: usize, cols: usize) -> anyhow::Result<()> {
        let local_size = 256;
        let global_size = ((rows + local_size - 1) / local_size) * local_size;

        let kernel = Kernel::builder()
            .program(&self.program)
            .name(kernels::kernel_names::MATMUL_VEC)
            .queue((*self.queue).clone())
            .global_work_size(global_size)
            .local_work_size(local_size)
            .arg(weights)
            .arg(x)
            .arg(out)
            .arg(&(rows as i32))
            .arg(&(cols as i32))
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build matmul_vec kernel: {}", e))?;

        unsafe {
            kernel.enq()
                .map_err(|e| anyhow::anyhow!("Failed to execute matmul_vec kernel: {}", e))?;
        }
        Ok(())
    }
}

// =============================================================================
// KernelBackend Implementation
// =============================================================================

impl KernelBackend for OpenCLBackend {
    type Tensor1 = Tensor1;
    type Tensor2 = Tensor2;

    fn name(&self) -> &'static str {
        "opencl"
    }

    fn is_available() -> bool {
        OpenCLBackend::is_available()
    }

    fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        let (rows, cols) = w.dim();

        // Create GPU buffers
        let w_buffer = self.create_buffer(
            w.as_slice().expect("Weight matrix must be contiguous"),
        ).expect("Failed to create weight buffer");
        let x_buffer = self.create_buffer(
            x.as_slice().expect("Input vector must be contiguous"),
        ).expect("Failed to create input buffer");
        let out_buffer = self.create_buffer_zeros(rows)
            .expect("Failed to create output buffer");

        // Launch kernel
        self.launch_matmul_vec(&w_buffer, &x_buffer, &out_buffer, rows, cols)
            .expect("Failed to launch matmul_vec kernel");

        // Synchronize and read result
        self.synchronize().expect("Failed to synchronize");
        let mut result = vec![0.0f32; rows];
        out_buffer.read(&mut result).enq().expect("Failed to read result");
        Array1::from_vec(result)
    }

    fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
        let result = self.matmul_vec(w, x);
        out.assign(&result);
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        // For now, use simple CPU implementation
        // TODO: Implement tiled GPU matmul kernel
        let (_m, k1) = a.dim();
        let (k2, _n) = b.dim();
        assert_eq!(k1, k2, "Matrix dimensions must match for multiplication");

        // Fall back to CPU implementation for now
        a.dot(b)
    }

    fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
        let n = x.len();

        // Create GPU buffer with data
        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"))
            .expect("Failed to create x buffer");
        let weight_buffer = self.create_buffer(weight.as_slice().expect("weight must be contiguous"))
            .expect("Failed to create weight buffer");

        // Launch kernel
        self.launch_rmsnorm(&x_buffer, &weight_buffer, n, eps)
            .expect("Failed to launch rmsnorm kernel");

        // Synchronize and copy result back
        self.synchronize().expect("Failed to synchronize");
        x_buffer.read(x.as_slice_mut().expect("x must be contiguous")).enq()
            .expect("Failed to read result");
    }

    fn softmax(&self, x: &mut Array1<f32>) {
        let n = x.len();

        // Create GPU buffer with data
        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"))
            .expect("Failed to create x buffer");

        // Launch kernel
        self.launch_softmax(&x_buffer, n)
            .expect("Failed to launch softmax kernel");

        // Synchronize and copy result back
        self.synchronize().expect("Failed to synchronize");
        x_buffer.read(x.as_slice_mut().expect("x must be contiguous")).enq()
            .expect("Failed to read result");
    }

    fn softmax_view(&self, x: &mut ArrayViewMut1<f32>) {
        // For views, we need to copy to owned array, process, and copy back
        let mut owned = x.to_owned();
        self.softmax(&mut owned);
        x.assign(&owned);
    }

    fn silu(&self, x: &Array1<f32>) -> Array1<f32> {
        let n = x.len();

        // Create GPU buffers
        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"))
            .expect("Failed to create x buffer");
        let out_buffer = self.create_buffer_zeros(n)
            .expect("Failed to create output buffer");

        // Launch kernel
        self.launch_silu(&x_buffer, &out_buffer, n)
            .expect("Failed to launch silu kernel");

        // Synchronize and read result
        self.synchronize().expect("Failed to synchronize");
        let mut result = vec![0.0f32; n];
        out_buffer.read(&mut result).enq().expect("Failed to read result");
        Array1::from_vec(result)
    }

    fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
        let (n_heads, head_dim) = x.dim();

        // Create GPU buffers
        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"))
            .expect("Failed to create x buffer");
        let cos_buffer = self.create_buffer(cos.as_slice().expect("cos must be contiguous"))
            .expect("Failed to create cos buffer");
        let sin_buffer = self.create_buffer(sin.as_slice().expect("sin must be contiguous"))
            .expect("Failed to create sin buffer");

        // Launch kernel
        self.launch_rope(&x_buffer, &cos_buffer, &sin_buffer, n_heads, head_dim)
            .expect("Failed to launch rope kernel");

        // Synchronize and copy result back
        self.synchronize().expect("Failed to synchronize");
        x_buffer.read(x.as_slice_mut().expect("x must be contiguous")).enq()
            .expect("Failed to read result");
    }

    fn compute_attention_scores(
        &self,
        query: ArrayView1<f32>,
        keys: ArrayView2<f32>,
        scores: &mut ArrayViewMut1<f32>,
        scale: f32,
    ) {
        let seq_len = keys.nrows();
        let head_dim = keys.ncols();

        // Create GPU buffers
        let query_owned = query.to_owned();
        let keys_owned = keys.to_owned();

        let query_buffer = self.create_buffer(
            query_owned.as_slice().expect("query must be contiguous"),
        ).expect("Failed to create query buffer");
        let keys_buffer = self.create_buffer(
            keys_owned.as_slice().expect("keys must be contiguous"),
        ).expect("Failed to create keys buffer");
        let scores_buffer = self.create_buffer_zeros(seq_len)
            .expect("Failed to create scores buffer");

        // Launch kernel
        self.launch_attention_scores(
            &query_buffer,
            &keys_buffer,
            &scores_buffer,
            seq_len,
            head_dim,
            scale,
        ).expect("Failed to launch attention_scores kernel");

        // Synchronize and copy result back
        self.synchronize().expect("Failed to synchronize");
        let mut result = vec![0.0f32; seq_len];
        scores_buffer.read(&mut result).enq().expect("Failed to read result");
        scores.assign(&Array1::from_vec(result));
    }

    fn weighted_sum_rows(
        &self,
        weights: ArrayView1<f32>,
        matrix: ArrayView2<f32>,
        out: &mut ArrayViewMut1<f32>,
    ) {
        let (n, d) = matrix.dim();

        // Create GPU buffers
        let weights_owned = weights.to_owned();
        let matrix_owned = matrix.to_owned();

        let weights_buffer = self.create_buffer(
            weights_owned.as_slice().expect("weights must be contiguous"),
        ).expect("Failed to create weights buffer");
        let matrix_buffer = self.create_buffer(
            matrix_owned.as_slice().expect("matrix must be contiguous"),
        ).expect("Failed to create matrix buffer");
        let out_buffer = self.create_buffer_zeros(d)
            .expect("Failed to create output buffer");

        // Launch kernel
        self.launch_weighted_sum(&weights_buffer, &matrix_buffer, &out_buffer, n, d)
            .expect("Failed to launch weighted_sum kernel");

        // Synchronize and copy result back
        self.synchronize().expect("Failed to synchronize");
        let mut result = vec![0.0f32; d];
        out_buffer.read(&mut result).enq().expect("Failed to read result");
        out.assign(&Array1::from_vec(result));
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opencl_availability() {
        // This test just checks the availability function doesn't panic
        let available = OpenCLBackend::is_available();
        println!("OpenCL available: {}", available);
    }

    #[test]
    fn test_list_devices() {
        let devices = OpenCLBackend::list_devices();
        println!("Available OpenCL devices:");
        for (platform, device, idx) in &devices {
            println!("  [{}] {} - {}", idx, platform, device);
        }
    }

    #[test]
    #[ignore] // Only run when OpenCL is available
    fn test_opencl_backend_creation() {
        if !OpenCLBackend::is_available() {
            return;
        }

        let backend = OpenCLBackend::new().expect("Failed to create OpenCL backend");
        assert_eq!(backend.name(), "opencl");
        println!("OpenCL device: {} ({})", backend.device_name(), backend.platform_name());
    }

    #[test]
    #[ignore] // Only run when OpenCL is available
    fn test_opencl_matmul_vec() {
        if !OpenCLBackend::is_available() {
            return;
        }

        let backend = OpenCLBackend::new().expect("Failed to create OpenCL backend");

        let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = backend.matmul_vec(&w, &x);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-5); // 1*1 + 2*2 + 3*3 = 14
        assert!((result[1] - 32.0).abs() < 1e-5); // 4*1 + 5*2 + 6*3 = 32
    }

    #[test]
    #[ignore] // Only run when OpenCL is available
    fn test_opencl_softmax() {
        if !OpenCLBackend::is_available() {
            return;
        }

        let backend = OpenCLBackend::new().expect("Failed to create OpenCL backend");

        let mut x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        backend.softmax(&mut x);

        // Check that softmax sums to 1
        let sum: f32 = x.sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "Softmax sum should be ~1.0, got {} (values: {:?})",
            sum,
            x
        );

        // Check that larger inputs have larger probabilities
        assert!(x[2] > x[1], "x[2]={} should be > x[1]={}", x[2], x[1]);
        assert!(x[1] > x[0], "x[1]={} should be > x[0]={}", x[1], x[0]);
    }

    #[test]
    #[ignore] // Only run when OpenCL is available
    fn test_opencl_silu() {
        if !OpenCLBackend::is_available() {
            return;
        }

        let backend = OpenCLBackend::new().expect("Failed to create OpenCL backend");

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

    #[test]
    #[ignore] // Only run when OpenCL is available
    fn test_opencl_rmsnorm() {
        if !OpenCLBackend::is_available() {
            return;
        }

        let backend = OpenCLBackend::new().expect("Failed to create OpenCL backend");

        let mut x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let weight = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let eps = 1e-5;

        backend.rmsnorm(&mut x, &weight, eps);

        // RMS = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.739
        // Normalized values should be x / rms
        let expected_rms = (30.0f32 / 4.0 + eps).sqrt();
        assert!((x[0] - 1.0 / expected_rms).abs() < 0.01);
        assert!((x[1] - 2.0 / expected_rms).abs() < 0.01);
    }
}
