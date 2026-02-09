//! Metal Backend for Apple Silicon GPUs
//!
//! This module provides GPU-accelerated inference using Apple's Metal framework.
//! It implements the `KernelBackend` trait for seamless integration with the rest of the system.
//!
//! # Features
//!
//! - Metal Performance Shaders (MPS) integration for optimized matrix operations
//! - Custom Metal compute shaders for activation functions (SiLU, RMSNorm, softmax)
//! - Unified Memory Architecture support (zero-copy CPU-GPU data sharing)
//! - GPU memory management with buffer pooling
//! - Asynchronous execution with Metal command queues
//!
//! # Usage
//!
//! ```ignore
//! use torchless::kernels::metal::MetalBackend;
//!
//! // Check if Metal is available
//! if MetalBackend::is_available() {
//!     let backend = MetalBackend::new()?;
//!     println!("Using Metal device: {}", backend.device_name());
//! }
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      MetalBackend                           │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │ Device       │  │ CommandQueue │  │ Library      │     │
//! │  └──────────────┘  └──────────────┘  └──────────────┘     │
//! │          │                │                 │              │
//! │          └────────────────┼─────────────────┘              │
//! │                           ▼                                │
//! │  ┌──────────────────────────────────────────────────────┐ │
//! │  │                  Metal Compute Shaders                │ │
//! │  │  - RMSNorm      - Softmax       - SiLU               │ │
//! │  │  - RoPE         - Attention     - Weighted Sum       │ │
//! │  └──────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Requirements
//!
//! - macOS 10.14+ or iOS 12+
//! - Apple Silicon (M1/M2/M3) or AMD/Intel GPU with Metal support
//! - metal-rs crate with appropriate features

pub mod memory;
pub mod shaders;
pub mod tensor;

pub use memory::MetalMemoryPool;
pub use tensor::MetalTensor;

use crate::kernels::backend::KernelBackend;
use crate::tensor::{Tensor1, Tensor2};
use metal::{Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library, MTLSize};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use std::sync::Arc;

// =============================================================================
// Metal Backend Implementation
// =============================================================================

/// Metal backend for Apple Silicon and other Metal-capable GPUs.
///
/// This backend uses Metal compute shaders for custom operations and
/// leverages Apple's unified memory architecture for efficient CPU-GPU
/// data sharing without explicit memory copies.
#[derive(Clone)]
pub struct MetalBackend {
    /// Metal device handle
    device: Arc<Device>,
    /// Command queue for GPU operations
    command_queue: Arc<CommandQueue>,
    /// Compiled shader library
    #[allow(dead_code)]
    library: Arc<Library>,
    /// Compiled compute pipelines
    pipelines: Arc<MetalPipelines>,
    /// Memory pool for buffer reuse
    #[allow(dead_code)]
    memory_pool: Arc<std::sync::Mutex<MetalMemoryPool>>,
    /// Device name
    device_name: String,
}

/// Container for compiled Metal compute pipelines
struct MetalPipelines {
    rmsnorm: ComputePipelineState,
    softmax: ComputePipelineState,
    silu: ComputePipelineState,
    rope: ComputePipelineState,
    attention_scores: ComputePipelineState,
    weighted_sum: ComputePipelineState,
    matmul_vec: ComputePipelineState,
    #[allow(dead_code)]
    matmul_vec_tiled: ComputePipelineState,
    #[allow(dead_code)]
    elementwise_mul: ComputePipelineState,
    #[allow(dead_code)]
    elementwise_add: ComputePipelineState,
    #[allow(dead_code)]
    fused_silu_mul: ComputePipelineState,
    #[allow(dead_code)]
    matmul: ComputePipelineState,
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device_name", &self.device_name)
            .finish()
    }
}

impl MetalBackend {
    /// Check if Metal is available on this system.
    pub fn is_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            Device::system_default().is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    /// Create a new Metal backend using the system default device.
    pub fn new() -> anyhow::Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device available"))?;

        let device_name = device.name().to_string();

        // Create command queue
        let command_queue = device.new_command_queue();

        // Compile shader library
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(shaders::METAL_SHADERS_SOURCE, &options)
            .map_err(|e| anyhow::anyhow!("Failed to compile Metal shaders: {}", e))?;

        // Create compute pipelines for all kernels
        let pipelines = Self::create_pipelines(&device, &library)?;

        // Initialize memory pool
        let memory_pool = Arc::new(std::sync::Mutex::new(MetalMemoryPool::new()));

        Ok(Self {
            device: Arc::new(device),
            command_queue: Arc::new(command_queue),
            library: Arc::new(library),
            pipelines: Arc::new(pipelines),
            memory_pool,
            device_name,
        })
    }

    /// Create compute pipelines for all kernels.
    fn create_pipelines(device: &Device, library: &Library) -> anyhow::Result<MetalPipelines> {
        let create_pipeline = |name: &str| -> anyhow::Result<ComputePipelineState> {
            let function = library
                .get_function(name, None)
                .map_err(|e| anyhow::anyhow!("Failed to get function '{}': {}", name, e))?;

            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| anyhow::anyhow!("Failed to create pipeline for '{}': {}", name, e))
        };

        Ok(MetalPipelines {
            rmsnorm: create_pipeline(shaders::kernel_names::RMSNORM)?,
            softmax: create_pipeline(shaders::kernel_names::SOFTMAX)?,
            silu: create_pipeline(shaders::kernel_names::SILU)?,
            rope: create_pipeline(shaders::kernel_names::ROPE)?,
            attention_scores: create_pipeline(shaders::kernel_names::ATTENTION_SCORES)?,
            weighted_sum: create_pipeline(shaders::kernel_names::WEIGHTED_SUM)?,
            matmul_vec: create_pipeline(shaders::kernel_names::MATMUL_VEC)?,
            matmul_vec_tiled: create_pipeline(shaders::kernel_names::MATMUL_VEC_TILED)?,
            elementwise_mul: create_pipeline(shaders::kernel_names::ELEMENTWISE_MUL)?,
            elementwise_add: create_pipeline(shaders::kernel_names::ELEMENTWISE_ADD)?,
            fused_silu_mul: create_pipeline(shaders::kernel_names::FUSED_SILU_MUL)?,
            matmul: create_pipeline(shaders::kernel_names::MATMUL)?,
        })
    }

    /// Get the name of the Metal device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get a reference to the underlying Metal device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a reference to the command queue.
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// Create a buffer from CPU data (unified memory - no copy needed for access).
    pub fn create_buffer(&self, data: &[f32]) -> Buffer {
        memory::create_buffer_with_data(&self.device, data)
    }

    /// Create a zero-initialized buffer.
    pub fn create_buffer_zeros(&self, len: usize) -> Buffer {
        memory::create_buffer_zeros(&self.device, len)
    }

    /// Transfer a 1D array to GPU buffer.
    pub fn to_device_1d(&self, data: &Array1<f32>) -> anyhow::Result<MetalTensor> {
        let slice = data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Array must be contiguous"))?;
        let buffer = self.create_buffer(slice);
        Ok(MetalTensor::new_1d(buffer, data.len()))
    }

    /// Transfer a 2D array to GPU buffer.
    pub fn to_device_2d(&self, data: &Array2<f32>) -> anyhow::Result<MetalTensor> {
        let slice = data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Array must be contiguous"))?;
        let (rows, cols) = data.dim();
        let buffer = self.create_buffer(slice);
        Ok(MetalTensor::new_2d(buffer, rows, cols))
    }

    /// Transfer data from GPU to CPU as 1D array.
    pub fn to_host_1d(&self, tensor: &MetalTensor) -> Array1<f32> {
        // With unified memory, we can read directly
        let slice = tensor.as_slice();
        Array1::from_vec(slice.to_vec())
    }

    /// Transfer data from GPU to CPU as 2D array.
    pub fn to_host_2d(&self, tensor: &MetalTensor) -> anyhow::Result<Array2<f32>> {
        let (rows, cols) = tensor
            .shape_2d()
            .ok_or_else(|| anyhow::anyhow!("Tensor is not 2D"))?;
        let slice = tensor.as_slice();
        Array2::from_shape_vec((rows, cols), slice.to_vec())
            .map_err(|e| anyhow::anyhow!("Shape error: {}", e))
    }

    /// Wait for all GPU operations to complete.
    pub fn synchronize(&self) {
        let command_buffer = self.command_queue.new_command_buffer();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // =========================================================================
    // GPU Kernel Launches
    // =========================================================================

    /// Launch RMSNorm kernel on GPU.
    fn launch_rmsnorm(&self, x: &Buffer, weight: &Buffer, n: usize, eps: f32) {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.rmsnorm);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            3,
            std::mem::size_of::<f32>() as u64,
            &eps as *const f32 as *const _,
        );

        // Block size must be a power of 2 for correct parallel reduction
        let block_size = 256.max(n.next_power_of_two()).min(1024);
        let shared_mem_size = block_size * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

        let grid_size = MTLSize::new(1, 1, 1);
        let threadgroup_size = MTLSize::new(block_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Launch softmax kernel on GPU.
    fn launch_softmax(&self, x: &Buffer, n: usize) {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.softmax);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );

        // Block size must be a power of 2 for correct parallel reduction
        // Use minimum of 256 threads (or next power of 2 >= n)
        let block_size = 256.max(n.next_power_of_two()).min(1024);
        // Need space for both max and sum shared memory
        let shared_mem_size = block_size * std::mem::size_of::<f32>() * 2;
        encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

        let grid_size = MTLSize::new(1, 1, 1);
        let threadgroup_size = MTLSize::new(block_size as u64, 1, 1);
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Launch SiLU kernel on GPU.
    fn launch_silu(&self, x: &Buffer, y: &Buffer, n: usize) {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.silu);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(y), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );

        let thread_count = n as u64;
        let threadgroup_size = 256u64.min(thread_count);
        let grid_size = MTLSize::new(thread_count, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Launch RoPE kernel on GPU.
    fn launch_rope(&self, x: &Buffer, cos: &Buffer, sin: &Buffer, n_heads: usize, head_dim: usize) {
        let half = head_dim / 2;
        let total = n_heads * half;

        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.rope);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(cos), 0);
        encoder.set_buffer(2, Some(sin), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &(n_heads as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &(head_dim as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<i32>() as u64,
            &(half as i32) as *const i32 as *const _,
        );

        let thread_count = total as u64;
        let threadgroup_size = 256u64.min(thread_count);
        let grid_size = MTLSize::new(thread_count, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Launch attention scores kernel on GPU.
    fn launch_attention_scores(
        &self,
        query: &Buffer,
        keys: &Buffer,
        scores: &Buffer,
        seq_len: usize,
        head_dim: usize,
        scale: f32,
    ) {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.attention_scores);
        encoder.set_buffer(0, Some(query), 0);
        encoder.set_buffer(1, Some(keys), 0);
        encoder.set_buffer(2, Some(scores), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &(seq_len as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &(head_dim as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<f32>() as u64,
            &scale as *const f32 as *const _,
        );

        let thread_count = seq_len as u64;
        let threadgroup_size = 256u64.min(thread_count);
        let grid_size = MTLSize::new(thread_count, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Launch weighted sum kernel on GPU.
    fn launch_weighted_sum(
        &self,
        weights: &Buffer,
        matrix: &Buffer,
        out: &Buffer,
        n: usize,
        d: usize,
    ) {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.weighted_sum);
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(matrix), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &(n as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &(d as i32) as *const i32 as *const _,
        );

        let thread_count = d as u64;
        let threadgroup_size = 256u64.min(thread_count);
        let grid_size = MTLSize::new(thread_count, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Launch matrix-vector multiplication kernel on GPU.
    fn launch_matmul_vec(
        &self,
        weights: &Buffer,
        x: &Buffer,
        out: &Buffer,
        rows: usize,
        cols: usize,
    ) {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.matmul_vec);
        encoder.set_buffer(0, Some(weights), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<i32>() as u64,
            &(rows as i32) as *const i32 as *const _,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<i32>() as u64,
            &(cols as i32) as *const i32 as *const _,
        );

        let thread_count = rows as u64;
        let threadgroup_size = 256u64.min(thread_count);
        let grid_size = MTLSize::new(thread_count, 1, 1);
        let threadgroup = MTLSize::new(threadgroup_size, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

// =============================================================================
// KernelBackend Implementation
// =============================================================================

impl KernelBackend for MetalBackend {
    type Tensor1 = Tensor1;
    type Tensor2 = Tensor2;

    fn name(&self) -> &'static str {
        "metal"
    }

    fn is_available() -> bool {
        MetalBackend::is_available()
    }

    fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        let (rows, cols) = w.dim();

        // Create GPU buffers
        let w_buffer = self.create_buffer(w.as_slice().expect("Weight matrix must be contiguous"));
        let x_buffer = self.create_buffer(x.as_slice().expect("Input vector must be contiguous"));
        let out_buffer = self.create_buffer_zeros(rows);

        // Launch kernel
        self.launch_matmul_vec(&w_buffer, &x_buffer, &out_buffer, rows, cols);

        // Read result (unified memory - direct access)
        let result_slice =
            unsafe { std::slice::from_raw_parts(out_buffer.contents() as *const f32, rows) };
        Array1::from_vec(result_slice.to_vec())
    }

    fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
        let result = self.matmul_vec(w, x);
        out.assign(&result);
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        // For now, use simple implementation
        // TODO: Use Metal Performance Shaders (MPS) for optimized matmul
        let (_m, k1) = a.dim();
        let (k2, _n) = b.dim();
        assert_eq!(k1, k2, "Matrix dimensions must match for multiplication");

        // Fall back to CPU implementation for now
        // MPS integration would require additional setup
        a.dot(b)
    }

    fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
        let n = x.len();

        // Create GPU buffer with data
        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"));
        let weight_buffer =
            self.create_buffer(weight.as_slice().expect("weight must be contiguous"));

        // Launch kernel
        self.launch_rmsnorm(&x_buffer, &weight_buffer, n, eps);

        // Copy result back
        let result_slice =
            unsafe { std::slice::from_raw_parts(x_buffer.contents() as *const f32, n) };
        x.as_slice_mut()
            .expect("x must be contiguous")
            .copy_from_slice(result_slice);
    }

    fn softmax(&self, x: &mut Array1<f32>) {
        let n = x.len();

        // Create GPU buffer with data
        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"));

        // Launch kernel
        self.launch_softmax(&x_buffer, n);

        // Copy result back
        let result_slice =
            unsafe { std::slice::from_raw_parts(x_buffer.contents() as *const f32, n) };
        x.as_slice_mut()
            .expect("x must be contiguous")
            .copy_from_slice(result_slice);
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
        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"));
        let out_buffer = self.create_buffer_zeros(n);

        // Launch kernel
        self.launch_silu(&x_buffer, &out_buffer, n);

        // Read result
        let result_slice =
            unsafe { std::slice::from_raw_parts(out_buffer.contents() as *const f32, n) };
        Array1::from_vec(result_slice.to_vec())
    }

    fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
        let (n_heads, head_dim) = x.dim();

        // Create GPU buffers
        let x_buffer = self.create_buffer(x.as_slice().expect("x must be contiguous"));
        let cos_buffer = self.create_buffer(cos.as_slice().expect("cos must be contiguous"));
        let sin_buffer = self.create_buffer(sin.as_slice().expect("sin must be contiguous"));

        // Launch kernel
        self.launch_rope(&x_buffer, &cos_buffer, &sin_buffer, n_heads, head_dim);

        // Copy result back
        let result_slice = unsafe {
            std::slice::from_raw_parts(x_buffer.contents() as *const f32, n_heads * head_dim)
        };
        x.as_slice_mut()
            .expect("x must be contiguous")
            .copy_from_slice(result_slice);
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

        let query_buffer =
            self.create_buffer(query_owned.as_slice().expect("query must be contiguous"));
        let keys_buffer =
            self.create_buffer(keys_owned.as_slice().expect("keys must be contiguous"));
        let scores_buffer = self.create_buffer_zeros(seq_len);

        // Launch kernel
        self.launch_attention_scores(
            &query_buffer,
            &keys_buffer,
            &scores_buffer,
            seq_len,
            head_dim,
            scale,
        );

        // Copy result back
        let result_slice =
            unsafe { std::slice::from_raw_parts(scores_buffer.contents() as *const f32, seq_len) };
        scores.assign(&Array1::from_vec(result_slice.to_vec()));
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
            weights_owned
                .as_slice()
                .expect("weights must be contiguous"),
        );
        let matrix_buffer =
            self.create_buffer(matrix_owned.as_slice().expect("matrix must be contiguous"));
        let out_buffer = self.create_buffer_zeros(d);

        // Launch kernel
        self.launch_weighted_sum(&weights_buffer, &matrix_buffer, &out_buffer, n, d);

        // Copy result back
        let result_slice =
            unsafe { std::slice::from_raw_parts(out_buffer.contents() as *const f32, d) };
        out.assign(&Array1::from_vec(result_slice.to_vec()));
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_availability() {
        // This test just checks the availability function doesn't panic
        let available = MetalBackend::is_available();
        println!("Metal available: {}", available);
    }

    #[test]
    #[ignore] // Only run when Metal is available
    fn test_metal_backend_creation() {
        if !MetalBackend::is_available() {
            return;
        }

        let backend = MetalBackend::new().expect("Failed to create Metal backend");
        assert_eq!(backend.name(), "metal");
        println!("Metal device: {}", backend.device_name());
    }

    #[test]
    #[ignore] // Only run when Metal is available
    fn test_metal_matmul_vec() {
        if !MetalBackend::is_available() {
            return;
        }

        let backend = MetalBackend::new().expect("Failed to create Metal backend");

        let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = backend.matmul_vec(&w, &x);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-5); // 1*1 + 2*2 + 3*3 = 14
        assert!((result[1] - 32.0).abs() < 1e-5); // 4*1 + 5*2 + 6*3 = 32
    }

    #[test]
    #[ignore] // Only run when Metal is available
    fn test_metal_softmax() {
        if !MetalBackend::is_available() {
            return;
        }

        let backend = MetalBackend::new().expect("Failed to create Metal backend");

        let mut x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        backend.softmax(&mut x);

        // Check that softmax sums to 1
        let sum: f32 = x.sum();
        // Use slightly larger tolerance for GPU floating point
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
    #[ignore] // Only run when Metal is available
    fn test_metal_silu() {
        if !MetalBackend::is_available() {
            return;
        }

        let backend = MetalBackend::new().expect("Failed to create Metal backend");

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
    #[ignore] // Only run when Metal is available
    fn test_metal_rmsnorm() {
        if !MetalBackend::is_available() {
            return;
        }

        let backend = MetalBackend::new().expect("Failed to create Metal backend");

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
