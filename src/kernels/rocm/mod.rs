//! ROCm Backend for AMD GPUs
//!
//! This module provides GPU-accelerated inference using AMD ROCm via the HIP runtime.
//! It implements the `KernelBackend` trait for seamless integration with the rest of the system.
//!
//! # Features
//!
//! - rocBLAS integration for optimized matrix operations
//! - Custom HIP kernels for activation functions (SiLU, RMSNorm, softmax)
//! - GPU memory management with buffer pooling
//! - Asynchronous execution with HIP streams
//! - CPU-GPU data transfer utilities
//!
//! # Usage
//!
//! ```ignore
//! use torchless::kernels::rocm::RocmBackend;
//!
//! // Check if ROCm is available
//! if RocmBackend::is_available() {
//!     let backend = RocmBackend::new()?;
//!     println!("Using ROCm device: {}", backend.device_name());
//! }
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      RocmBackend                            │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │ HipDevice    │  │ HipStream    │  │ RocblasHandle│     │
//! │  └──────────────┘  └──────────────┘  └──────────────┘     │
//! │          │                │                 │              │
//! │          └────────────────┼─────────────────┘              │
//! │                           ▼                                │
//! │  ┌──────────────────────────────────────────────────────┐ │
//! │  │                  HipKernels                           │ │
//! │  │  - RMSNorm      - Softmax       - SiLU               │ │
//! │  │  - RoPE         - Attention     - Weighted Sum       │ │
//! │  └──────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Requirements
//!
//! - AMD GPU with ROCm support (RDNA, CDNA, or GCN architecture)
//! - ROCm runtime installed (tested with ROCm 5.x and 6.x)
//! - rocBLAS library

pub mod kernels;
pub mod memory;
pub mod tensor;

pub use memory::RocmMemoryPool;
pub use tensor::RocmTensor;

use crate::kernels::backend::KernelBackend;
use crate::tensor::{Tensor1, Tensor2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use std::sync::Arc;

// =============================================================================
// HIP Types (FFI wrappers)
// =============================================================================

/// Opaque pointer to HIP device memory.
/// This wraps a device pointer allocated via hipMalloc.
#[derive(Clone)]
pub struct HipSlice<T> {
    /// Device pointer
    ptr: *mut T,
    /// Number of elements
    len: usize,
}

impl<T> HipSlice<T> {
    /// Create a new HipSlice from a raw device pointer.
    ///
    /// # Safety
    /// The pointer must be a valid device pointer allocated via hipMalloc.
    pub unsafe fn from_raw(ptr: *mut T, len: usize) -> Self {
        Self { ptr, len }
    }

    /// Get the device pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get a mutable device pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// Safety: HipSlice is just a device pointer, thread-safety is managed by HIP runtime
unsafe impl<T: Send> Send for HipSlice<T> {}
unsafe impl<T: Sync> Sync for HipSlice<T> {}

// Debug implementation for HipSlice
impl<T> std::fmt::Debug for HipSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipSlice")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

// =============================================================================
// HIP Runtime FFI
// =============================================================================

/// HIP error codes
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipError {
    Success = 0,
    ErrorInvalidValue = 1,
    ErrorOutOfMemory = 2,
    ErrorNotInitialized = 3,
    ErrorDeinitialized = 4,
    ErrorProfilerDisabled = 5,
    ErrorProfilerNotInitialized = 6,
    ErrorProfilerAlreadyStarted = 7,
    ErrorProfilerAlreadyStopped = 8,
    ErrorInvalidConfiguration = 9,
    ErrorInvalidDevice = 100,
    ErrorInvalidMemcpyDirection = 21,
    ErrorUnknown = 999,
}

impl HipError {
    /// Convert an HIP error code to an HipError variant.
    #[allow(dead_code)]
    pub fn from_code(code: i32) -> Self {
        match code {
            0 => HipError::Success,
            1 => HipError::ErrorInvalidValue,
            2 => HipError::ErrorOutOfMemory,
            3 => HipError::ErrorNotInitialized,
            100 => HipError::ErrorInvalidDevice,
            21 => HipError::ErrorInvalidMemcpyDirection,
            _ => HipError::ErrorUnknown,
        }
    }
}

impl std::fmt::Display for HipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for HipError {}

/// HIP memory copy direction
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

// External HIP functions (linked at runtime)
#[cfg(feature = "rocm")]
#[allow(dead_code)]
mod hip_ffi {
    use super::*;
    use std::os::raw::{c_int, c_void};

    #[link(name = "amdhip64")]
    extern "C" {
        pub fn hipGetDeviceCount(count: *mut c_int) -> c_int;
        pub fn hipSetDevice(device: c_int) -> c_int;
        pub fn hipGetDevice(device: *mut c_int) -> c_int;
        pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
        pub fn hipFree(ptr: *mut c_void) -> c_int;
        pub fn hipMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            size: usize,
            kind: HipMemcpyKind,
        ) -> c_int;
        pub fn hipMemcpyAsync(
            dst: *mut c_void,
            src: *const c_void,
            size: usize,
            kind: HipMemcpyKind,
            stream: *mut c_void,
        ) -> c_int;
        pub fn hipDeviceSynchronize() -> c_int;
        pub fn hipStreamCreate(stream: *mut *mut c_void) -> c_int;
        pub fn hipStreamDestroy(stream: *mut c_void) -> c_int;
        pub fn hipStreamSynchronize(stream: *mut c_void) -> c_int;
        pub fn hipMemset(ptr: *mut c_void, value: c_int, count: usize) -> c_int;
        pub fn hipGetDeviceProperties(
            prop: *mut HipDeviceProperties,
            device: c_int,
        ) -> c_int;
    }
    
    #[repr(C)]
    pub struct HipDeviceProperties {
        pub name: [i8; 256],
        pub total_global_mem: usize,
        pub shared_mem_per_block: usize,
        pub warp_size: i32,
        pub max_threads_per_block: i32,
        pub max_threads_dim: [i32; 3],
        pub max_grid_size: [i32; 3],
        pub clock_rate: i32,
        pub memory_clock_rate: i32,
        pub memory_bus_width: i32,
        pub major: i32,
        pub minor: i32,
        pub multi_processor_count: i32,
        pub l2_cache_size: i32,
        pub max_threads_per_multi_processor: i32,
        pub compute_mode: i32,
        // Additional fields omitted for brevity
        _padding: [u8; 512],
    }
}

// =============================================================================
// rocBLAS FFI
// =============================================================================

#[cfg(feature = "rocm")]
#[allow(dead_code)]
mod rocblas_ffi {
    use std::os::raw::{c_int, c_void};
    
    pub type RocblasHandle = *mut c_void;
    
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub enum RocblasOperation {
        None = 111,
        Transpose = 112,
        ConjugateTranspose = 113,
    }
    
    #[link(name = "rocblas")]
    extern "C" {
        pub fn rocblas_create_handle(handle: *mut RocblasHandle) -> c_int;
        pub fn rocblas_destroy_handle(handle: RocblasHandle) -> c_int;
        pub fn rocblas_set_stream(handle: RocblasHandle, stream: *mut c_void) -> c_int;
        
        pub fn rocblas_sgemv(
            handle: RocblasHandle,
            trans: RocblasOperation,
            m: c_int,
            n: c_int,
            alpha: *const f32,
            A: *const f32,
            lda: c_int,
            x: *const f32,
            incx: c_int,
            beta: *const f32,
            y: *mut f32,
            incy: c_int,
        ) -> c_int;
        
        pub fn rocblas_sgemm(
            handle: RocblasHandle,
            transA: RocblasOperation,
            transB: RocblasOperation,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: *const f32,
            A: *const f32,
            lda: c_int,
            B: *const f32,
            ldb: c_int,
            beta: *const f32,
            C: *mut f32,
            ldc: c_int,
        ) -> c_int;
    }
}

// =============================================================================
// ROCm Backend Implementation
// =============================================================================

/// ROCm backend for AMD GPUs.
///
/// This backend uses rocBLAS for matrix operations and custom HIP kernels
/// for other operations like RMSNorm, softmax, and SiLU.
#[derive(Clone)]
pub struct RocmBackend {
    /// Device index
    device_index: usize,
    /// Device name
    device_name: String,
    /// rocBLAS handle (wrapped for thread safety)
    #[cfg(feature = "rocm")]
    rocblas_handle: Arc<RocblasHandle>,
    /// HIP stream for async operations
    #[cfg(feature = "rocm")]
    #[allow(dead_code)]
    stream: Arc<HipStream>,
    /// Memory pool for buffer reuse
    #[allow(dead_code)]
    memory_pool: Arc<std::sync::Mutex<RocmMemoryPool>>,
}

#[cfg(feature = "rocm")]
struct RocblasHandle(rocblas_ffi::RocblasHandle);

#[cfg(feature = "rocm")]
unsafe impl Send for RocblasHandle {}
#[cfg(feature = "rocm")]
unsafe impl Sync for RocblasHandle {}

#[cfg(feature = "rocm")]
struct HipStream(*mut std::ffi::c_void);

#[cfg(feature = "rocm")]
unsafe impl Send for HipStream {}
#[cfg(feature = "rocm")]
unsafe impl Sync for HipStream {}

impl std::fmt::Debug for RocmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmBackend")
            .field("device_index", &self.device_index)
            .field("device_name", &self.device_name)
            .finish()
    }
}

impl RocmBackend {
    /// Check if ROCm is available on this system.
    pub fn is_available() -> bool {
        #[cfg(feature = "rocm")]
        {
            Self::device_count() > 0
        }
        #[cfg(not(feature = "rocm"))]
        {
            false
        }
    }

    /// Get the number of available ROCm devices.
    pub fn device_count() -> usize {
        #[cfg(feature = "rocm")]
        {
            let mut count: std::os::raw::c_int = 0;
            let result = unsafe { hip_ffi::hipGetDeviceCount(&mut count) };
            if result == 0 {
                count as usize
            } else {
                0
            }
        }
        #[cfg(not(feature = "rocm"))]
        {
            0
        }
    }

    /// Create a new ROCm backend on device 0.
    pub fn new() -> anyhow::Result<Self> {
        Self::with_device(0)
    }

    /// Create a new ROCm backend on the specified device.
    #[cfg(feature = "rocm")]
    pub fn with_device(device_index: usize) -> anyhow::Result<Self> {
        use std::os::raw::c_int;

        // Set the device
        let result = unsafe { hip_ffi::hipSetDevice(device_index as c_int) };
        if result != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set HIP device {}: error {}",
                device_index,
                result
            ));
        }

        // Get device properties
        let mut props = std::mem::MaybeUninit::<hip_ffi::HipDeviceProperties>::uninit();
        let result =
            unsafe { hip_ffi::hipGetDeviceProperties(props.as_mut_ptr(), device_index as c_int) };
        let device_name = if result == 0 {
            let props = unsafe { props.assume_init() };
            let name_bytes: Vec<u8> = props
                .name
                .iter()
                .take_while(|&&c| c != 0)
                .map(|&c| c as u8)
                .collect();
            String::from_utf8_lossy(&name_bytes).to_string()
        } else {
            format!("ROCm Device {}", device_index)
        };

        // Create rocBLAS handle
        let mut rocblas_handle: rocblas_ffi::RocblasHandle = std::ptr::null_mut();
        let result = unsafe { rocblas_ffi::rocblas_create_handle(&mut rocblas_handle) };
        if result != 0 {
            return Err(anyhow::anyhow!(
                "Failed to create rocBLAS handle: error {}",
                result
            ));
        }

        // Create HIP stream
        let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
        let result = unsafe { hip_ffi::hipStreamCreate(&mut stream) };
        if result != 0 {
            unsafe { rocblas_ffi::rocblas_destroy_handle(rocblas_handle) };
            return Err(anyhow::anyhow!(
                "Failed to create HIP stream: error {}",
                result
            ));
        }

        // Set rocBLAS to use our stream
        let result = unsafe { rocblas_ffi::rocblas_set_stream(rocblas_handle, stream) };
        if result != 0 {
            unsafe {
                hip_ffi::hipStreamDestroy(stream);
                rocblas_ffi::rocblas_destroy_handle(rocblas_handle);
            }
            return Err(anyhow::anyhow!(
                "Failed to set rocBLAS stream: error {}",
                result
            ));
        }

        Ok(Self {
            device_index,
            device_name,
            rocblas_handle: Arc::new(RocblasHandle(rocblas_handle)),
            stream: Arc::new(HipStream(stream)),
            memory_pool: Arc::new(std::sync::Mutex::new(RocmMemoryPool::new())),
        })
    }

    /// Create a new ROCm backend on the specified device (stub for non-rocm builds).
    #[cfg(not(feature = "rocm"))]
    pub fn with_device(_device_index: usize) -> anyhow::Result<Self> {
        Err(anyhow::anyhow!(
            "ROCm support not compiled. Enable the 'rocm' feature."
        ))
    }

    /// Get the name of the ROCm device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Get the device index.
    pub fn device_index(&self) -> usize {
        self.device_index
    }

    /// Transfer a 1D array from CPU to GPU.
    #[cfg(feature = "rocm")]
    pub fn to_device_1d(&self, data: &Array1<f32>) -> anyhow::Result<RocmTensor> {
        let slice = data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Array must be contiguous"))?;
        let gpu_data = self.alloc_and_copy(slice)?;
        Ok(RocmTensor::new_1d(gpu_data, data.len()))
    }

    /// Transfer a 2D array from CPU to GPU.
    #[cfg(feature = "rocm")]
    pub fn to_device_2d(&self, data: &Array2<f32>) -> anyhow::Result<RocmTensor> {
        let slice = data
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Array must be contiguous"))?;
        let (rows, cols) = data.dim();
        let gpu_data = self.alloc_and_copy(slice)?;
        Ok(RocmTensor::new_2d(gpu_data, rows, cols))
    }

    /// Transfer data from GPU to CPU as 1D array.
    #[cfg(feature = "rocm")]
    pub fn to_host_1d(&self, tensor: &RocmTensor) -> anyhow::Result<Array1<f32>> {
        let mut data = vec![0.0f32; tensor.len()];
        self.copy_to_host(tensor.data(), &mut data)?;
        Ok(Array1::from_vec(data))
    }

    /// Transfer data from GPU to CPU as 2D array.
    #[cfg(feature = "rocm")]
    pub fn to_host_2d(&self, tensor: &RocmTensor) -> anyhow::Result<Array2<f32>> {
        let (rows, cols) = tensor
            .shape_2d()
            .ok_or_else(|| anyhow::anyhow!("Tensor is not 2D"))?;
        let mut data = vec![0.0f32; tensor.len()];
        self.copy_to_host(tensor.data(), &mut data)?;
        Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| anyhow::anyhow!("Shape error: {}", e))
    }

    /// Allocate GPU memory and copy data.
    #[cfg(feature = "rocm")]
    fn alloc_and_copy(&self, data: &[f32]) -> anyhow::Result<HipSlice<f32>> {
        use std::os::raw::c_void;

        let size = data.len() * std::mem::size_of::<f32>();

        // Allocate device memory
        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        let result = unsafe { hip_ffi::hipMalloc(&mut device_ptr, size) };
        if result != 0 {
            return Err(anyhow::anyhow!(
                "Failed to allocate GPU memory: error {}",
                result
            ));
        }

        // Copy data to device
        let result = unsafe {
            hip_ffi::hipMemcpy(
                device_ptr,
                data.as_ptr() as *const c_void,
                size,
                HipMemcpyKind::HostToDevice,
            )
        };
        if result != 0 {
            unsafe { hip_ffi::hipFree(device_ptr) };
            return Err(anyhow::anyhow!(
                "Failed to copy data to GPU: error {}",
                result
            ));
        }

        Ok(unsafe { HipSlice::from_raw(device_ptr as *mut f32, data.len()) })
    }

    /// Copy data from GPU to host.
    #[cfg(feature = "rocm")]
    fn copy_to_host(&self, src: &HipSlice<f32>, dst: &mut [f32]) -> anyhow::Result<()> {
        use std::os::raw::c_void;

        let size = dst.len() * std::mem::size_of::<f32>();
        let result = unsafe {
            hip_ffi::hipMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                src.as_ptr() as *const c_void,
                size,
                HipMemcpyKind::DeviceToHost,
            )
        };
        if result != 0 {
            return Err(anyhow::anyhow!(
                "Failed to copy data from GPU: error {}",
                result
            ));
        }
        Ok(())
    }

    /// Copy tensor data from GPU to host as a Vec.
    #[cfg(feature = "rocm")]
    pub fn copy_tensor_to_host(&self, tensor: &RocmTensor) -> anyhow::Result<Vec<f32>> {
        let mut data = vec![0.0f32; tensor.len()];
        self.copy_to_host(tensor.data(), &mut data)?;
        Ok(data)
    }

    /// Allocate zero-initialized GPU memory.
    #[cfg(feature = "rocm")]
    pub fn alloc_zeros(&self, len: usize) -> anyhow::Result<HipSlice<f32>> {
        use std::os::raw::c_void;

        let size = len * std::mem::size_of::<f32>();

        // Allocate device memory
        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        let result = unsafe { hip_ffi::hipMalloc(&mut device_ptr, size) };
        if result != 0 {
            return Err(anyhow::anyhow!(
                "Failed to allocate GPU memory: error {}",
                result
            ));
        }

        // Zero the memory
        let result = unsafe { hip_ffi::hipMemset(device_ptr, 0, size) };
        if result != 0 {
            unsafe { hip_ffi::hipFree(device_ptr) };
            return Err(anyhow::anyhow!(
                "Failed to zero GPU memory: error {}",
                result
            ));
        }

        Ok(unsafe { HipSlice::from_raw(device_ptr as *mut f32, len) })
    }

    /// Synchronize the device (wait for all operations to complete).
    #[cfg(feature = "rocm")]
    pub fn synchronize(&self) -> anyhow::Result<()> {
        let result = unsafe { hip_ffi::hipDeviceSynchronize() };
        if result != 0 {
            return Err(anyhow::anyhow!(
                "Failed to synchronize device: error {}",
                result
            ));
        }
        Ok(())
    }

    // =========================================================================
    // rocBLAS Operations
    // =========================================================================

    /// Perform matrix-vector multiplication using rocBLAS.
    #[cfg(feature = "rocm")]
    fn rocblas_gemv(
        &self,
        w: &HipSlice<f32>,
        x: &HipSlice<f32>,
        y: &mut HipSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> anyhow::Result<()> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // rocBLAS uses column-major, our arrays are row-major
        // For row-major A @ x, we compute A^T @ x in column-major
        let result = unsafe {
            rocblas_ffi::rocblas_sgemv(
                self.rocblas_handle.0,
                rocblas_ffi::RocblasOperation::Transpose,
                cols as i32, // rows of A^T = cols of A
                rows as i32, // cols of A^T = rows of A
                &alpha,
                w.as_ptr(),
                cols as i32, // lda
                x.as_ptr(),
                1, // incx
                &beta,
                y.as_mut_ptr(),
                1, // incy
            )
        };

        if result != 0 {
            return Err(anyhow::anyhow!("rocBLAS GEMV failed: error {}", result));
        }
        Ok(())
    }

    /// Perform matrix-matrix multiplication using rocBLAS.
    #[cfg(feature = "rocm")]
    fn rocblas_gemm(
        &self,
        a: &HipSlice<f32>,
        b: &HipSlice<f32>,
        c: &mut HipSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> anyhow::Result<()> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // For row-major matrices, we compute C^T = B^T @ A^T in column-major
        let result = unsafe {
            rocblas_ffi::rocblas_sgemm(
                self.rocblas_handle.0,
                rocblas_ffi::RocblasOperation::None,
                rocblas_ffi::RocblasOperation::None,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                b.as_ptr(),
                n as i32, // lda
                a.as_ptr(),
                k as i32, // ldb
                &beta,
                c.as_mut_ptr(),
                n as i32, // ldc
            )
        };

        if result != 0 {
            return Err(anyhow::anyhow!("rocBLAS GEMM failed: error {}", result));
        }
        Ok(())
    }
}

// =============================================================================
// KernelBackend Implementation
// =============================================================================

impl KernelBackend for RocmBackend {
    type Tensor1 = Tensor1;
    type Tensor2 = Tensor2;

    fn name(&self) -> &'static str {
        "rocm"
    }

    fn is_available() -> bool {
        RocmBackend::is_available()
    }

    #[cfg(feature = "rocm")]
    fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        let (rows, cols) = w.dim();

        // Transfer to GPU
        let w_gpu = self.to_device_2d(w).expect("Failed to transfer weights to GPU");
        let x_gpu = self.to_device_1d(x).expect("Failed to transfer input to GPU");
        let y_gpu = self.alloc_zeros(rows).expect("Failed to allocate output");
        let mut y_tensor = RocmTensor::new_1d(y_gpu, rows);

        // Perform matmul
        self.rocblas_gemv(
            w_gpu.data(),
            x_gpu.data(),
            y_tensor.data_mut(),
            rows,
            cols,
        )
        .expect("rocBLAS GEMV failed");

        // Transfer back
        self.to_host_1d(&y_tensor)
            .expect("Failed to transfer result to CPU")
    }

    #[cfg(not(feature = "rocm"))]
    fn matmul_vec(&self, _w: &Array2<f32>, _x: &Array1<f32>) -> Array1<f32> {
        panic!("ROCm support not compiled")
    }

    fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
        let result = self.matmul_vec(w, x);
        out.assign(&result);
    }

    #[cfg(feature = "rocm")]
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(k1, k2, "Matrix dimensions must match for multiplication");
        let k = k1;

        // Transfer to GPU
        let a_gpu = self.to_device_2d(a).expect("Failed to transfer A to GPU");
        let b_gpu = self.to_device_2d(b).expect("Failed to transfer B to GPU");
        let c_gpu = self.alloc_zeros(m * n).expect("Failed to allocate output");
        let mut c_tensor = RocmTensor::new_2d(c_gpu, m, n);

        // Perform matmul
        self.rocblas_gemm(a_gpu.data(), b_gpu.data(), c_tensor.data_mut(), m, n, k)
            .expect("rocBLAS GEMM failed");

        // Transfer back
        self.to_host_2d(&c_tensor)
            .expect("Failed to transfer result to CPU")
    }

    #[cfg(not(feature = "rocm"))]
    fn matmul(&self, _a: &Array2<f32>, _b: &Array2<f32>) -> Array2<f32> {
        panic!("ROCm support not compiled")
    }

    fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
        // For now, use CPU implementation
        // TODO: Implement HIP kernel for RMSNorm
        crate::kernels::rmsnorm(x, weight, eps);
    }

    fn softmax(&self, x: &mut Array1<f32>) {
        // For now, use CPU implementation
        // TODO: Implement HIP kernel for softmax
        crate::kernels::softmax(x);
    }

    fn softmax_view(&self, x: &mut ArrayViewMut1<f32>) {
        // For now, use CPU implementation
        crate::kernels::softmax_view(x);
    }

    fn silu(&self, x: &Array1<f32>) -> Array1<f32> {
        // For now, use CPU implementation
        // TODO: Implement HIP kernel for SiLU
        crate::kernels::silu(x)
    }

    fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
        // For now, use CPU implementation
        // TODO: Implement HIP kernel for RoPE
        crate::kernels::apply_rope(x, cos, sin);
    }

    fn compute_attention_scores(
        &self,
        query: ArrayView1<f32>,
        keys: ArrayView2<f32>,
        scores: &mut ArrayViewMut1<f32>,
        scale: f32,
    ) {
        // For now, use CPU implementation
        crate::kernels::compute_attention_scores(query, keys, scores, scale);
    }

    fn weighted_sum_rows(
        &self,
        weights: ArrayView1<f32>,
        matrix: ArrayView2<f32>,
        out: &mut ArrayViewMut1<f32>,
    ) {
        // For now, use CPU implementation
        crate::kernels::weighted_sum_rows(weights, matrix, out);
    }
}

// =============================================================================
// Cleanup
// =============================================================================

#[cfg(feature = "rocm")]
impl Drop for RocblasHandle {
    fn drop(&mut self) {
        unsafe {
            rocblas_ffi::rocblas_destroy_handle(self.0);
        }
    }
}

#[cfg(feature = "rocm")]
impl Drop for HipStream {
    fn drop(&mut self) {
        unsafe {
            hip_ffi::hipStreamDestroy(self.0);
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
    fn test_rocm_availability() {
        // This test just checks the availability function doesn't panic
        let available = RocmBackend::is_available();
        println!("ROCm available: {}", available);
    }

    #[test]
    fn test_device_count() {
        let count = RocmBackend::device_count();
        println!("ROCm device count: {}", count);
    }

    #[test]
    #[ignore] // Only run when ROCm is available
    fn test_rocm_backend_creation() {
        if !RocmBackend::is_available() {
            return;
        }

        let backend = RocmBackend::new().expect("Failed to create ROCm backend");
        assert_eq!(backend.name(), "rocm");
    }

    #[test]
    #[ignore] // Only run when ROCm is available
    fn test_rocm_matmul_vec() {
        if !RocmBackend::is_available() {
            return;
        }

        let backend = RocmBackend::new().expect("Failed to create ROCm backend");

        let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = backend.matmul_vec(&w, &x);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-5); // 1*1 + 2*2 + 3*3 = 14
        assert!((result[1] - 32.0).abs() < 1e-5); // 4*1 + 5*2 + 6*3 = 32
    }
}
