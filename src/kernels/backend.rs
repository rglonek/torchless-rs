//! Backend Abstraction Layer
//!
//! This module provides a trait-based abstraction for compute backends,
//! allowing the same model code to run on CPU, CUDA, Metal, ROCm, or OpenCL.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    KernelBackend Trait                       │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!        ┌─────────────────────┼─────────────────────┐
//!        ▼                     ▼                     ▼
//! ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
//! │ CpuBackend  │       │ CudaBackend │       │ MetalBackend│
//! └─────────────┘       └─────────────┘       └─────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use torchless::kernels::backend::{Backend, init_backend, BackendPreference};
//!
//! // Auto-select best available backend
//! let backend = init_backend(BackendPreference::Auto)?;
//!
//! // Use the backend for operations
//! let output = backend.matmul_vec(&weights, &input);
//! ```

use crate::tensor::{Tensor1, Tensor2};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use std::fmt::Debug;

/// Trait for compute backend implementations.
///
/// Each backend provides optimized implementations of the core operations
/// needed for transformer inference. Backends are responsible for managing
/// their own memory and device state.
pub trait KernelBackend: Send + Sync + Debug {
    /// The tensor type used by this backend.
    /// For CPU, this is ndarray. For GPU backends, this would be device tensors.
    type Tensor1: Clone;
    type Tensor2: Clone;

    /// Returns the name of this backend (e.g., "cpu", "cuda", "metal").
    fn name(&self) -> &'static str;

    /// Returns true if this backend is available on the current system.
    fn is_available() -> bool
    where
        Self: Sized;

    // =========================================================================
    // Matrix Operations
    // =========================================================================

    /// Matrix-vector multiplication: out = W @ x
    /// W: (n, d), x: (d,) -> out: (n,)
    fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32>;

    /// Matrix-vector multiplication with pre-allocated output: out = W @ x
    /// W: (n, d), x: (d,) -> out: (n,)
    fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>);

    /// Matrix-matrix multiplication: C = A @ B
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;

    // =========================================================================
    // Normalization Operations
    // =========================================================================

    /// RMSNorm: x = x * weight / sqrt(mean(x^2) + eps)
    fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32);

    /// Softmax: x = exp(x - max(x)) / sum(exp(x - max(x)))
    fn softmax(&self, x: &mut Array1<f32>);

    /// Softmax for array views (avoids allocation)
    fn softmax_view(&self, x: &mut ArrayViewMut1<f32>);

    // =========================================================================
    // Activation Functions
    // =========================================================================

    /// SiLU (Swish) activation: out = x / (1 + exp(-x))
    fn silu(&self, x: &Array1<f32>) -> Array1<f32>;

    // =========================================================================
    // Positional Encoding
    // =========================================================================

    /// Apply Rotary Position Embedding (RoPE) to query/key tensors.
    /// x: [n_heads, head_dim], cos/sin: [head_dim/2]
    fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>);

    // =========================================================================
    // Attention Operations
    // =========================================================================

    /// Compute attention scores: scores[i] = keys[i, :].dot(query) * scale
    /// query: (d,), keys: (n, d) -> scores: (n,)
    fn compute_attention_scores(
        &self,
        query: ArrayView1<f32>,
        keys: ArrayView2<f32>,
        scores: &mut ArrayViewMut1<f32>,
        scale: f32,
    );

    /// Compute weighted sum of rows: out[j] = sum_i(weights[i] * matrix[i, j])
    /// weights: (n,), matrix: (n, d) -> out: (d,)
    fn weighted_sum_rows(
        &self,
        weights: ArrayView1<f32>,
        matrix: ArrayView2<f32>,
        out: &mut ArrayViewMut1<f32>,
    );
}

/// CPU backend implementation using ndarray and optional SIMD/parallel features.
#[derive(Debug, Clone, Default)]
pub struct CpuBackend {
    /// Whether to use SIMD optimizations when available
    pub use_simd: bool,
    /// Whether to use parallel processing when available
    pub use_parallel: bool,
}

impl CpuBackend {
    /// Create a new CPU backend with default settings (auto-detect features).
    pub fn new() -> Self {
        Self {
            use_simd: cfg!(feature = "simd"),
            use_parallel: cfg!(feature = "parallel"),
        }
    }

    /// Create a CPU backend with specific feature flags.
    pub fn with_features(use_simd: bool, use_parallel: bool) -> Self {
        Self {
            use_simd,
            use_parallel,
        }
    }
}

impl KernelBackend for CpuBackend {
    type Tensor1 = Tensor1;
    type Tensor2 = Tensor2;

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn is_available() -> bool {
        true // CPU is always available
    }

    fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        if self.use_parallel {
            #[cfg(feature = "parallel")]
            {
                return super::matmul_vec_parallel(w, x);
            }
        }
        super::matmul_vec(w, x)
    }

    fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
        if self.use_parallel {
            #[cfg(feature = "parallel")]
            {
                super::matmul_vec_into_parallel(w, x, out);
                return;
            }
        }
        super::matmul_vec_into(w, x, out);
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        super::matmul(a, b)
    }

    fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
        if self.use_simd {
            #[cfg(feature = "simd")]
            {
                super::rmsnorm_simd(x, weight, eps);
                return;
            }
        }
        super::rmsnorm(x, weight, eps);
    }

    fn softmax(&self, x: &mut Array1<f32>) {
        if self.use_simd {
            #[cfg(feature = "simd")]
            {
                super::softmax_simd(x);
                return;
            }
        }
        super::softmax(x);
    }

    fn softmax_view(&self, x: &mut ArrayViewMut1<f32>) {
        if self.use_simd {
            #[cfg(feature = "simd")]
            {
                super::softmax_view_simd(x);
                return;
            }
        }
        super::softmax_view(x);
    }

    fn silu(&self, x: &Array1<f32>) -> Array1<f32> {
        if self.use_simd {
            #[cfg(feature = "simd")]
            {
                return super::silu_simd(x);
            }
        }
        super::silu(x)
    }

    fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
        if self.use_simd {
            #[cfg(feature = "simd")]
            {
                super::apply_rope_simd(x, cos, sin);
                return;
            }
        }
        super::apply_rope(x, cos, sin);
    }

    fn compute_attention_scores(
        &self,
        query: ArrayView1<f32>,
        keys: ArrayView2<f32>,
        scores: &mut ArrayViewMut1<f32>,
        scale: f32,
    ) {
        if self.use_parallel {
            #[cfg(feature = "parallel")]
            {
                super::compute_attention_scores_parallel(query, keys, scores, scale);
                return;
            }
        }
        super::compute_attention_scores(query, keys, scores, scale);
    }

    fn weighted_sum_rows(
        &self,
        weights: ArrayView1<f32>,
        matrix: ArrayView2<f32>,
        out: &mut ArrayViewMut1<f32>,
    ) {
        if self.use_parallel {
            #[cfg(feature = "parallel")]
            {
                super::weighted_sum_rows_parallel(weights, matrix, out);
                return;
            }
        }
        super::weighted_sum_rows(weights, matrix, out);
    }
}

// =============================================================================
// CUDA Backend
// =============================================================================

// The CUDA backend is implemented in a separate module for cleaner organization.
// Re-export the CudaBackend type for use in this module.
#[cfg(feature = "cuda")]
pub use super::cuda::CudaBackend;

// The ROCm backend is implemented in a separate module for AMD GPU support.
// Re-export the RocmBackend type for use in this module.
#[cfg(feature = "rocm")]
pub use super::rocm::RocmBackend;

// =============================================================================
// ROCm Backend
// =============================================================================

// The ROCm backend is implemented in a separate module (super::rocm).
// It provides GPU acceleration for AMD GPUs using the HIP runtime and rocBLAS.

// =============================================================================
// Metal Backend
// =============================================================================

// The Metal backend is implemented in a separate module for Apple Silicon GPU support.
// Re-export the MetalBackend type for use in this module.
#[cfg(feature = "metal-gpu")]
pub use super::metal::MetalBackend;

// =============================================================================
// OpenCL Backend
// =============================================================================

// The OpenCL backend is implemented in a separate module for cross-platform GPU support.
// Re-export the OpenCLBackend type for use in this module.
#[cfg(feature = "opencl")]
pub use super::opencl::OpenCLBackend;

// =============================================================================
// Backend Selection
// =============================================================================

/// Preference for backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Automatically select the best available backend.
    #[default]
    Auto,
    /// Force CPU backend.
    Cpu,
    /// Prefer CUDA (NVIDIA GPU).
    #[cfg(feature = "cuda")]
    Cuda,
    /// Prefer ROCm (AMD GPU).
    #[cfg(feature = "rocm")]
    Rocm,
    /// Prefer Metal (Apple Silicon GPU).
    #[cfg(feature = "metal-gpu")]
    Metal,
    /// Prefer OpenCL (cross-platform GPU).
    #[cfg(feature = "opencl")]
    OpenCL,
}

/// Dynamic backend wrapper that can hold any backend type.
#[derive(Debug)]
pub enum Backend {
    Cpu(CpuBackend),
    #[cfg(feature = "cuda")]
    Cuda(CudaBackend),
    #[cfg(feature = "rocm")]
    Rocm(RocmBackend),
    #[cfg(feature = "metal-gpu")]
    Metal(MetalBackend),
    #[cfg(feature = "opencl")]
    OpenCL(OpenCLBackend),
}

impl Backend {
    /// Get the name of the current backend.
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Cpu(b) => b.name(),
            #[cfg(feature = "cuda")]
            Backend::Cuda(_) => "cuda",
            #[cfg(feature = "rocm")]
            Backend::Rocm(_) => "rocm",
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(_) => "metal",
            #[cfg(feature = "opencl")]
            Backend::OpenCL(_) => "opencl",
        }
    }

    /// Get a reference to the CPU backend if this is a CPU backend.
    pub fn as_cpu(&self) -> Option<&CpuBackend> {
        match self {
            Backend::Cpu(b) => Some(b),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// Get a reference to the ROCm backend if this is a ROCm backend.
    #[cfg(feature = "rocm")]
    pub fn as_rocm(&self) -> Option<&RocmBackend> {
        match self {
            Backend::Rocm(b) => Some(b),
            _ => None,
        }
    }

    /// Get a reference to the Metal backend if this is a Metal backend.
    #[cfg(feature = "metal-gpu")]
    pub fn as_metal(&self) -> Option<&MetalBackend> {
        match self {
            Backend::Metal(b) => Some(b),
            _ => None,
        }
    }

    /// Get a reference to the OpenCL backend if this is an OpenCL backend.
    #[cfg(feature = "opencl")]
    pub fn as_opencl(&self) -> Option<&OpenCLBackend> {
        match self {
            Backend::OpenCL(b) => Some(b),
            _ => None,
        }
    }
}

// Implement KernelBackend for Backend enum to allow dynamic dispatch
impl KernelBackend for Backend {
    type Tensor1 = Tensor1;
    type Tensor2 = Tensor2;

    fn name(&self) -> &'static str {
        Backend::name(self)
    }

    fn is_available() -> bool {
        true // At least CPU is always available
    }

    fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        match self {
            Backend::Cpu(b) => b.matmul_vec(w, x),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b) => b.matmul_vec(w, x),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b) => b.matmul_vec(w, x),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b) => b.matmul_vec(w, x),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b) => b.matmul_vec(w, x),
        }
    }

    fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
        match self {
            Backend::Cpu(b) => b.matmul_vec_into(w, x, out),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b) => b.matmul_vec_into(w, x, out),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b) => b.matmul_vec_into(w, x, out),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b) => b.matmul_vec_into(w, x, out),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b) => b.matmul_vec_into(w, x, out),
        }
    }

    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        match self {
            Backend::Cpu(b_end) => b_end.matmul(a, b),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b_end) => b_end.matmul(a, b),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b_end) => b_end.matmul(a, b),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b_end) => b_end.matmul(a, b),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b_end) => b_end.matmul(a, b),
        }
    }

    fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
        match self {
            Backend::Cpu(b) => b.rmsnorm(x, weight, eps),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b) => b.rmsnorm(x, weight, eps),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b) => b.rmsnorm(x, weight, eps),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b) => b.rmsnorm(x, weight, eps),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b) => b.rmsnorm(x, weight, eps),
        }
    }

    fn softmax(&self, x: &mut Array1<f32>) {
        match self {
            Backend::Cpu(b) => b.softmax(x),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b) => b.softmax(x),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b) => b.softmax(x),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b) => b.softmax(x),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b) => b.softmax(x),
        }
    }

    fn softmax_view(&self, x: &mut ArrayViewMut1<f32>) {
        match self {
            Backend::Cpu(b) => b.softmax_view(x),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b) => b.softmax_view(x),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b) => b.softmax_view(x),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b) => b.softmax_view(x),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b) => b.softmax_view(x),
        }
    }

    fn silu(&self, x: &Array1<f32>) -> Array1<f32> {
        match self {
            Backend::Cpu(b) => b.silu(x),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b) => b.silu(x),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b) => b.silu(x),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b) => b.silu(x),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b) => b.silu(x),
        }
    }

    fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
        match self {
            Backend::Cpu(b) => b.apply_rope(x, cos, sin),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b) => b.apply_rope(x, cos, sin),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b) => b.apply_rope(x, cos, sin),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b) => b.apply_rope(x, cos, sin),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b) => b.apply_rope(x, cos, sin),
        }
    }

    fn compute_attention_scores(
        &self,
        query: ArrayView1<f32>,
        keys: ArrayView2<f32>,
        scores: &mut ArrayViewMut1<f32>,
        scale: f32,
    ) {
        match self {
            Backend::Cpu(b) => b.compute_attention_scores(query, keys, scores, scale),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b) => b.compute_attention_scores(query, keys, scores, scale),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b) => b.compute_attention_scores(query, keys, scores, scale),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b) => b.compute_attention_scores(query, keys, scores, scale),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b) => b.compute_attention_scores(query, keys, scores, scale),
        }
    }

    fn weighted_sum_rows(
        &self,
        weights: ArrayView1<f32>,
        matrix: ArrayView2<f32>,
        out: &mut ArrayViewMut1<f32>,
    ) {
        match self {
            Backend::Cpu(b) => b.weighted_sum_rows(weights, matrix, out),
            #[cfg(feature = "cuda")]
            Backend::Cuda(b) => b.weighted_sum_rows(weights, matrix, out),
            #[cfg(feature = "rocm")]
            Backend::Rocm(b) => b.weighted_sum_rows(weights, matrix, out),
            #[cfg(feature = "metal-gpu")]
            Backend::Metal(b) => b.weighted_sum_rows(weights, matrix, out),
            #[cfg(feature = "opencl")]
            Backend::OpenCL(b) => b.weighted_sum_rows(weights, matrix, out),
        }
    }
}

/// Initialize a backend based on preference.
///
/// This function will attempt to initialize the preferred backend, falling back
/// to CPU if the preferred backend is not available.
///
/// # Example
///
/// ```ignore
/// use torchless::kernels::backend::{init_backend, BackendPreference};
///
/// // Auto-select best available backend
/// let backend = init_backend(BackendPreference::Auto)?;
/// println!("Using backend: {}", backend.name());
/// ```
pub fn init_backend(preference: BackendPreference) -> anyhow::Result<Backend> {
    match preference {
        BackendPreference::Auto => {
            // Try GPU backends in order of preference
            #[cfg(feature = "cuda")]
            {
                if CudaBackend::is_available() {
                    if let Ok(backend) = CudaBackend::new() {
                        return Ok(Backend::Cuda(backend));
                    }
                }
            }

            #[cfg(feature = "rocm")]
            {
                if RocmBackend::is_available() {
                    if let Ok(backend) = RocmBackend::new() {
                        return Ok(Backend::Rocm(backend));
                    }
                }
            }

            #[cfg(feature = "metal-gpu")]
            {
                if MetalBackend::is_available() {
                    if let Ok(backend) = MetalBackend::new() {
                        return Ok(Backend::Metal(backend));
                    }
                }
            }

            #[cfg(feature = "opencl")]
            {
                if OpenCLBackend::is_available() {
                    if let Ok(backend) = OpenCLBackend::new() {
                        return Ok(Backend::OpenCL(backend));
                    }
                }
            }

            // Fall back to CPU
            Ok(Backend::Cpu(CpuBackend::new()))
        }
        BackendPreference::Cpu => Ok(Backend::Cpu(CpuBackend::new())),
        #[cfg(feature = "cuda")]
        BackendPreference::Cuda => Ok(Backend::Cuda(CudaBackend::new()?)),
        #[cfg(feature = "rocm")]
        BackendPreference::Rocm => Ok(Backend::Rocm(RocmBackend::new()?)),
        #[cfg(feature = "metal-gpu")]
        BackendPreference::Metal => Ok(Backend::Metal(MetalBackend::new()?)),
        #[cfg(feature = "opencl")]
        BackendPreference::OpenCL => Ok(Backend::OpenCL(OpenCLBackend::new()?)),
    }
}

/// Get the default backend (CPU with auto-detected features).
pub fn default_backend() -> CpuBackend {
    CpuBackend::new()
}

// =============================================================================
// Backend Discovery and Device Enumeration
// =============================================================================

/// Information about an available backend.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend type
    pub backend_type: BackendType,
    /// Human-readable name
    pub name: String,
    /// Whether the backend is currently available
    pub available: bool,
    /// Device information (if available)
    pub devices: Vec<DeviceInfo>,
    /// Error message if not available
    pub error: Option<String>,
}

/// Type of backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    Cpu,
    Cuda,
    Rocm,
    Metal,
    OpenCL,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Cpu => write!(f, "CPU"),
            BackendType::Cuda => write!(f, "CUDA"),
            BackendType::Rocm => write!(f, "ROCm"),
            BackendType::Metal => write!(f, "Metal"),
            BackendType::OpenCL => write!(f, "OpenCL"),
        }
    }
}

/// Information about a specific device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device index
    pub index: usize,
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Free memory in bytes (if available)
    pub free_memory: Option<usize>,
    /// Compute capability or equivalent
    pub compute_capability: Option<String>,
    /// Whether unified memory is used
    pub unified_memory: bool,
}

impl DeviceInfo {
    /// Get total memory in GB.
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory as f64 / 1024.0 / 1024.0 / 1024.0
    }

    /// Get free memory in GB (if available).
    pub fn free_memory_gb(&self) -> Option<f64> {
        self.free_memory
            .map(|m| m as f64 / 1024.0 / 1024.0 / 1024.0)
    }

    /// Check if device can fit a model of given size.
    pub fn can_fit(&self, required_bytes: usize) -> bool {
        if let Some(free) = self.free_memory {
            free >= required_bytes
        } else {
            // If we don't know free memory, assume 80% of total is usable
            (self.total_memory as f64 * 0.8) as usize >= required_bytes
        }
    }
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {}: {:.2} GB",
            self.index,
            self.name,
            self.total_memory_gb()
        )?;
        if let Some(free_gb) = self.free_memory_gb() {
            write!(f, " ({:.2} GB free)", free_gb)?;
        }
        if let Some(ref cc) = self.compute_capability {
            write!(f, ", compute {}", cc)?;
        }
        if self.unified_memory {
            write!(f, " (unified memory)")?;
        }
        Ok(())
    }
}

/// Discover all available backends and their devices.
pub fn discover_backends() -> Vec<BackendInfo> {
    let mut backends = Vec::new();

    // CPU is always available
    backends.push(BackendInfo {
        backend_type: BackendType::Cpu,
        name: "CPU".to_string(),
        available: true,
        devices: vec![DeviceInfo {
            index: 0,
            name: get_cpu_name(),
            total_memory: get_system_memory(),
            free_memory: get_available_memory(),
            compute_capability: None,
            unified_memory: false,
        }],
        error: None,
    });

    // Check CUDA
    #[cfg(feature = "cuda")]
    {
        backends.push(discover_cuda());
    }
    #[cfg(not(feature = "cuda"))]
    {
        backends.push(BackendInfo {
            backend_type: BackendType::Cuda,
            name: "CUDA".to_string(),
            available: false,
            devices: vec![],
            error: Some("CUDA feature not enabled at compile time".to_string()),
        });
    }

    // Check ROCm
    #[cfg(feature = "rocm")]
    {
        backends.push(discover_rocm());
    }
    #[cfg(not(feature = "rocm"))]
    {
        backends.push(BackendInfo {
            backend_type: BackendType::Rocm,
            name: "ROCm".to_string(),
            available: false,
            devices: vec![],
            error: Some("ROCm feature not enabled at compile time".to_string()),
        });
    }

    // Check Metal
    #[cfg(feature = "metal-gpu")]
    {
        backends.push(discover_metal());
    }
    #[cfg(not(feature = "metal-gpu"))]
    {
        backends.push(BackendInfo {
            backend_type: BackendType::Metal,
            name: "Metal".to_string(),
            available: false,
            devices: vec![],
            error: Some("Metal feature not enabled at compile time".to_string()),
        });
    }

    // Check OpenCL
    #[cfg(feature = "opencl")]
    {
        backends.push(discover_opencl());
    }
    #[cfg(not(feature = "opencl"))]
    {
        backends.push(BackendInfo {
            backend_type: BackendType::OpenCL,
            name: "OpenCL".to_string(),
            available: false,
            devices: vec![],
            error: Some("OpenCL feature not enabled at compile time".to_string()),
        });
    }

    backends
}

/// Get the best available backend based on performance heuristics.
pub fn best_available_backend() -> BackendType {
    let backends = discover_backends();

    // Priority: CUDA > ROCm > Metal > OpenCL > CPU
    for backend_type in [
        BackendType::Cuda,
        BackendType::Rocm,
        BackendType::Metal,
        BackendType::OpenCL,
    ] {
        if let Some(info) = backends.iter().find(|b| b.backend_type == backend_type) {
            if info.available && !info.devices.is_empty() {
                return backend_type;
            }
        }
    }

    BackendType::Cpu
}

/// Select a backend that can fit a model of the given size.
pub fn select_backend_for_model(required_bytes: usize) -> anyhow::Result<(BackendType, usize)> {
    let backends = discover_backends();

    // Priority: CUDA > ROCm > Metal > OpenCL > CPU
    for backend_type in [
        BackendType::Cuda,
        BackendType::Rocm,
        BackendType::Metal,
        BackendType::OpenCL,
    ] {
        if let Some(info) = backends.iter().find(|b| b.backend_type == backend_type) {
            if info.available {
                for device in &info.devices {
                    if device.can_fit(required_bytes) {
                        return Ok((backend_type, device.index));
                    }
                }
            }
        }
    }

    // Fall back to CPU
    if let Some(cpu_info) = backends.iter().find(|b| b.backend_type == BackendType::Cpu) {
        if let Some(device) = cpu_info.devices.first() {
            if device.can_fit(required_bytes) {
                return Ok((BackendType::Cpu, 0));
            }
        }
    }

    anyhow::bail!(
        "No backend has enough memory for model ({:.2} GB required)",
        required_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    )
}

/// Print a summary of all available backends.
pub fn print_backend_summary() {
    let backends = discover_backends();

    println!("=== Available Backends ===");
    for info in &backends {
        let status = if info.available { "✓" } else { "✗" };
        println!("{} {} ({})", status, info.name, info.backend_type);

        if info.available {
            for device in &info.devices {
                println!("    {}", device);
            }
        } else if let Some(ref error) = info.error {
            println!("    Error: {}", error);
        }
    }

    let best = best_available_backend();
    println!("\nRecommended backend: {}", best);
}

// =============================================================================
// Backend-specific Discovery Functions
// =============================================================================

#[cfg(feature = "cuda")]
fn discover_cuda() -> BackendInfo {
    use cudarc::driver::CudaDevice;

    let mut devices = Vec::new();
    let mut available = false;

    // Try to enumerate CUDA devices
    match CudaDevice::count() {
        Ok(count) if count > 0 => {
            available = true;
            for i in 0..(count as usize) {
                match CudaDevice::new(i) {
                    Ok(device) => {
                        // Get device properties
                        let name = device
                            .name()
                            .unwrap_or_else(|_| format!("CUDA Device {}", i));

                        // Note: cudarc doesn't expose memory info directly
                        // We'd need to call CUDA runtime APIs for that
                        // For now, use a reasonable estimate
                        devices.push(DeviceInfo {
                            index: i,
                            name,
                            total_memory: 8 * 1024 * 1024 * 1024, // 8 GB estimate
                            free_memory: None,
                            compute_capability: None,
                            unified_memory: false,
                        });
                    }
                    Err(_) => continue,
                }
            }
        }
        Ok(_) => {
            return BackendInfo {
                backend_type: BackendType::Cuda,
                name: "CUDA".to_string(),
                available: false,
                devices: vec![],
                error: Some("No CUDA devices found".to_string()),
            };
        }
        Err(e) => {
            return BackendInfo {
                backend_type: BackendType::Cuda,
                name: "CUDA".to_string(),
                available: false,
                devices: vec![],
                error: Some(format!("CUDA initialization failed: {:?}", e)),
            };
        }
    }

    BackendInfo {
        backend_type: BackendType::Cuda,
        name: "CUDA".to_string(),
        available,
        devices,
        error: None,
    }
}

#[cfg(feature = "rocm")]
fn discover_rocm() -> BackendInfo {
    // ROCm discovery - similar to CUDA but using HIP
    // For now, return basic info since we don't have HIP bindings
    BackendInfo {
        backend_type: BackendType::Rocm,
        name: "ROCm".to_string(),
        available: RocmBackend::is_available(),
        devices: if RocmBackend::is_available() {
            vec![DeviceInfo {
                index: 0,
                name: "AMD GPU".to_string(),
                total_memory: 8 * 1024 * 1024 * 1024, // 8 GB estimate
                free_memory: None,
                compute_capability: None,
                unified_memory: false,
            }]
        } else {
            vec![]
        },
        error: if !RocmBackend::is_available() {
            Some("ROCm runtime not available".to_string())
        } else {
            None
        },
    }
}

#[cfg(feature = "metal-gpu")]
fn discover_metal() -> BackendInfo {
    use metal::Device;

    match Device::system_default() {
        Some(device) => {
            let name = device.name().to_string();
            let total_memory = device.recommended_max_working_set_size() as usize;

            BackendInfo {
                backend_type: BackendType::Metal,
                name: "Metal".to_string(),
                available: true,
                devices: vec![DeviceInfo {
                    index: 0,
                    name,
                    total_memory,
                    free_memory: None, // Metal doesn't expose this directly
                    compute_capability: None,
                    unified_memory: true, // Apple Silicon always uses unified memory
                }],
                error: None,
            }
        }
        None => BackendInfo {
            backend_type: BackendType::Metal,
            name: "Metal".to_string(),
            available: false,
            devices: vec![],
            error: Some("No Metal device found".to_string()),
        },
    }
}

#[cfg(feature = "opencl")]
fn discover_opencl() -> BackendInfo {
    use ocl::{Device as OclDevice, Platform};

    let mut devices = Vec::new();
    let mut available = false;

    // Enumerate all platforms and devices
    let platforms = Platform::list();
    let mut device_index = 0;
    for platform in platforms {
        if let Ok(platform_devices) = OclDevice::list_all(&platform) {
            for device in platform_devices {
                available = true;

                let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
                let total_memory = device
                    .info(ocl::enums::DeviceInfo::GlobalMemSize)
                    .ok()
                    .and_then(|info| match info {
                        ocl::enums::DeviceInfoResult::GlobalMemSize(size) => Some(size as usize),
                        _ => None,
                    })
                    .unwrap_or(0);

                devices.push(DeviceInfo {
                    index: device_index,
                    name,
                    total_memory,
                    free_memory: None,
                    compute_capability: None,
                    unified_memory: false,
                });
                device_index += 1;
            }
        }
    }

    BackendInfo {
        backend_type: BackendType::OpenCL,
        name: "OpenCL".to_string(),
        available,
        devices,
        error: if !available {
            Some("No OpenCL devices found".to_string())
        } else {
            None
        },
    }
}

// =============================================================================
// System Information Helpers
// =============================================================================

/// Get the CPU name/model.
fn get_cpu_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in contents.lines() {
                if line.starts_with("model name") {
                    if let Some(name) = line.split(':').nth(1) {
                        return name.trim().to_string();
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if output.status.success() {
                if let Ok(name) = String::from_utf8(output.stdout) {
                    return name.trim().to_string();
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("wmic").args(["cpu", "get", "name"]).output() {
            if output.status.success() {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    let lines: Vec<&str> = text.lines().collect();
                    if lines.len() > 1 {
                        return lines[1].trim().to_string();
                    }
                }
            }
        }
    }

    "Unknown CPU".to_string()
}

/// Get total system memory in bytes.
fn get_system_memory() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if output.status.success() {
                if let Ok(bytes_str) = String::from_utf8(output.stdout) {
                    if let Ok(bytes) = bytes_str.trim().parse::<usize>() {
                        return bytes;
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows MEMORYSTATUS would require windows-sys crate
        // Return reasonable default
        return 16 * 1024 * 1024 * 1024; // 16 GB default
    }

    16 * 1024 * 1024 * 1024 // 16 GB default
}

/// Get available (free) system memory in bytes.
fn get_available_memory() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<usize>() {
                            return Some(kb * 1024);
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        // On macOS, get page size and free pages
        if let (Ok(pagesize_output), Ok(vmstat_output)) = (
            Command::new("sysctl").args(["-n", "hw.pagesize"]).output(),
            Command::new("vm_stat").output(),
        ) {
            if pagesize_output.status.success() && vmstat_output.status.success() {
                if let Ok(pagesize_str) = String::from_utf8(pagesize_output.stdout) {
                    if let Ok(pagesize) = pagesize_str.trim().parse::<usize>() {
                        if let Ok(vmstat) = String::from_utf8(vmstat_output.stdout) {
                            // Parse "Pages free" line
                            for line in vmstat.lines() {
                                if line.contains("Pages free:") {
                                    let parts: Vec<&str> = line.split_whitespace().collect();
                                    if let Some(pages_str) = parts.last() {
                                        let pages_str = pages_str.trim_end_matches('.');
                                        if let Ok(pages) = pages_str.parse::<usize>() {
                                            return Some(pages * pagesize);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

// =============================================================================
// Memory-Aware Backend Initialization
// =============================================================================

/// Initialize a backend with memory checks.
///
/// This function checks if there's enough memory available before initializing
/// the backend, and provides helpful error messages if not.
pub fn init_backend_with_memory_check(
    preference: BackendPreference,
    required_bytes: usize,
) -> anyhow::Result<Backend> {
    #[allow(unused_variables)]
    let backends = discover_backends();

    // If Auto preference, try to find best backend with enough memory
    if matches!(preference, BackendPreference::Auto) {
        match select_backend_for_model(required_bytes) {
            Ok((BackendType::Cpu, _)) => return Ok(Backend::Cpu(CpuBackend::new())),
            #[cfg(feature = "cuda")]
            Ok((BackendType::Cuda, _)) => {
                if let Ok(backend) = CudaBackend::new() {
                    return Ok(Backend::Cuda(backend));
                }
            }
            #[cfg(feature = "rocm")]
            Ok((BackendType::Rocm, _)) => {
                if let Ok(backend) = RocmBackend::new() {
                    return Ok(Backend::Rocm(backend));
                }
            }
            #[cfg(feature = "metal-gpu")]
            Ok((BackendType::Metal, _)) => {
                if let Ok(backend) = MetalBackend::new() {
                    return Ok(Backend::Metal(backend));
                }
            }
            #[cfg(feature = "opencl")]
            Ok((BackendType::OpenCL, _)) => {
                if let Ok(backend) = OpenCLBackend::new() {
                    return Ok(Backend::OpenCL(backend));
                }
            }
            _ => {}
        }

        // Fall back to regular init
        return init_backend(BackendPreference::Auto);
    }

    // For specific preferences, check memory first
    match preference {
        BackendPreference::Cpu => {
            // CPU always available
            Ok(Backend::Cpu(CpuBackend::new()))
        }
        #[cfg(feature = "cuda")]
        BackendPreference::Cuda => {
            let cuda_info = backends
                .iter()
                .find(|b| b.backend_type == BackendType::Cuda);
            if let Some(info) = cuda_info {
                if !info.available {
                    anyhow::bail!(
                        "CUDA not available: {}",
                        info.error.as_deref().unwrap_or("Unknown error")
                    );
                }
                if let Some(device) = info.devices.first() {
                    if !device.can_fit(required_bytes) {
                        anyhow::bail!(
                            "CUDA device {} has insufficient memory: {:.2} GB available, {:.2} GB required",
                            device.name,
                            device.free_memory.unwrap_or(device.total_memory) as f64 / 1024.0 / 1024.0 / 1024.0,
                            required_bytes as f64 / 1024.0 / 1024.0 / 1024.0
                        );
                    }
                }
            }
            Ok(Backend::Cuda(CudaBackend::new()?))
        }
        #[cfg(feature = "rocm")]
        BackendPreference::Rocm => {
            let rocm_info = backends
                .iter()
                .find(|b| b.backend_type == BackendType::Rocm);
            if let Some(info) = rocm_info {
                if !info.available {
                    anyhow::bail!(
                        "ROCm not available: {}",
                        info.error.as_deref().unwrap_or("Unknown error")
                    );
                }
            }
            Ok(Backend::Rocm(RocmBackend::new()?))
        }
        #[cfg(feature = "metal-gpu")]
        BackendPreference::Metal => {
            let metal_info = backends
                .iter()
                .find(|b| b.backend_type == BackendType::Metal);
            if let Some(info) = metal_info {
                if !info.available {
                    anyhow::bail!(
                        "Metal not available: {}",
                        info.error.as_deref().unwrap_or("Unknown error")
                    );
                }
                if let Some(device) = info.devices.first() {
                    if !device.can_fit(required_bytes) {
                        anyhow::bail!(
                            "Metal device {} has insufficient memory: {:.2} GB available, {:.2} GB required",
                            device.name,
                            device.total_memory as f64 / 1024.0 / 1024.0 / 1024.0,
                            required_bytes as f64 / 1024.0 / 1024.0 / 1024.0
                        );
                    }
                }
            }
            Ok(Backend::Metal(MetalBackend::new()?))
        }
        #[cfg(feature = "opencl")]
        BackendPreference::OpenCL => {
            let opencl_info = backends
                .iter()
                .find(|b| b.backend_type == BackendType::OpenCL);
            if let Some(info) = opencl_info {
                if !info.available {
                    anyhow::bail!(
                        "OpenCL not available: {}",
                        info.error.as_deref().unwrap_or("Unknown error")
                    );
                }
            }
            Ok(Backend::OpenCL(OpenCLBackend::new()?))
        }
        #[allow(unreachable_patterns)]
        _ => init_backend(preference),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
        assert!(CpuBackend::is_available());
    }

    #[test]
    fn test_init_backend_auto() {
        let backend = init_backend(BackendPreference::Auto).unwrap();
        // Should at least get CPU backend
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn test_init_backend_cpu() {
        let backend = init_backend(BackendPreference::Cpu).unwrap();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn test_cpu_backend_matmul_vec() {
        let backend = CpuBackend::new();

        let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let result = backend.matmul_vec(&w, &x);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-6); // 1*1 + 2*2 + 3*3 = 14
        assert!((result[1] - 32.0).abs() < 1e-6); // 4*1 + 5*2 + 6*3 = 32
    }

    #[test]
    fn test_cpu_backend_softmax() {
        let backend = CpuBackend::new();

        let mut x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        backend.softmax(&mut x);

        // Check that softmax sums to 1
        let sum: f32 = x.sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that larger inputs have larger probabilities
        assert!(x[2] > x[1]);
        assert!(x[1] > x[0]);
    }

    #[test]
    fn test_cpu_backend_silu() {
        let backend = CpuBackend::new();

        let x = Array1::from_vec(vec![0.0, 1.0, -1.0]);
        let result = backend.silu(&x);

        assert_eq!(result.len(), 3);
        // SiLU(0) = 0
        assert!(result[0].abs() < 1e-6);
        // SiLU(1) = 1 / (1 + exp(-1)) ≈ 0.731
        assert!((result[1] - 0.731).abs() < 0.01);
        // SiLU(-1) = -1 / (1 + exp(1)) ≈ -0.269
        assert!((result[2] - (-0.269)).abs() < 0.01);
    }
}
