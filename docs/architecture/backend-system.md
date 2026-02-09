# Backend System

The backend trait system provides an abstraction layer for multiple compute backends (CPU, GPU).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KernelBackend Trait                       │
└─────────────────────────────────────────────────────────────┘
                             │
       ┌──────────┬──────────┬──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│CpuBackend│ │CudaBackend│ │MetalBackend│ │OpenCLBackend│ │WebGPUBackend│
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## KernelBackend Trait

Defines the interface for all compute backends:

```rust
pub trait KernelBackend: Send + Sync + Debug {
    type Tensor1: Clone;
    type Tensor2: Clone;

    fn name(&self) -> &'static str;
    fn is_available() -> bool where Self: Sized;

    // Matrix operations
    fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32>;
    fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>);
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;

    // Normalization
    fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32);
    fn softmax(&self, x: &mut Array1<f32>);
    fn softmax_view(&self, x: &mut ArrayViewMut1<f32>);

    // Activation
    fn silu(&self, x: &Array1<f32>) -> Array1<f32>;

    // Positional encoding
    fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>);

    // Attention
    fn compute_attention_scores(&self, query: ArrayView1<f32>, keys: ArrayView2<f32>, 
                                 scores: &mut ArrayViewMut1<f32>, scale: f32);
    fn weighted_sum_rows(&self, weights: ArrayView1<f32>, matrix: ArrayView2<f32>, 
                          out: &mut ArrayViewMut1<f32>);
}
```

## CpuBackend Implementation

Fully implemented CPU backend with feature-gated optimizations:

```rust
pub struct CpuBackend {
    pub use_simd: bool,     // Uses SIMD kernels when `simd` feature enabled
    pub use_parallel: bool, // Uses Rayon parallelism when `parallel` feature enabled
}
```

## GPU Backend Implementations

| Backend | Feature | Platforms | Library |
|---------|---------|-----------|---------|
| CUDA | `cuda` | Linux, Windows | cudarc, cuBLAS |
| ROCm | `rocm` | Linux | HIP, rocBLAS |
| Metal | `metal-gpu` | macOS | metal-rs, MPS |
| OpenCL | `opencl` | All | ocl |
| WebGPU | `webgpu` | All | wgpu |

## Backend Selection

Runtime backend selection with automatic fallback:

```rust
pub enum BackendPreference {
    Auto,    // Select best available
    Cpu,     // Force CPU
    Cuda,    // Prefer CUDA
    Metal,   // Prefer Metal
    Rocm,    // Prefer ROCm
    OpenCL,  // Prefer OpenCL
    WebGPU,  // Prefer WebGPU
}

// Usage
let backend = init_backend(BackendPreference::Auto)?;
println!("Using backend: {}", backend.name());
```

### Selection Priority (Auto mode)

1. CUDA (NVIDIA GPUs)
2. ROCm (AMD GPUs)
3. Metal (Apple Silicon)
4. OpenCL (fallback GPU)
5. WebGPU (cross-platform via wgpu)
6. CPU (always available)

## Backend Discovery

```rust
use torchless::{discover_backends, print_backend_summary, best_available_backend};

// Print all available backends
print_backend_summary();
// Output:
// === Available Backends ===
// ✓ CPU (CPU)
//     [0] Apple M2 Pro: 32.00 GB (28.50 GB free)
// ✓ Metal (Metal)
//     [0] Apple M2 Pro: 21.33 GB (unified memory)
// ✗ CUDA (CUDA)
//     Error: CUDA feature not enabled at compile time
// 
// Recommended backend: Metal

// Get the best backend programmatically
let best = best_available_backend();
```

## Memory-Aware Initialization

```rust
use torchless::{init_backend_with_memory_check, estimate_inference_memory};

// Estimate memory for Mistral 7B
let estimate = estimate_inference_memory(
    4096, 14336, 32, 32000, 32, 8, 4096, 4
);

println!("Model requires: {:.2} GB", estimate.total_gb());

// Initialize with memory check
match init_backend_with_memory_check(BackendPreference::Auto, estimate.total_bytes) {
    Ok(backend) => println!("Initialized {} backend", backend.name()),
    Err(e) => eprintln!("Failed: {}", e),
}
```

## Device Information

```rust
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub total_memory: usize,
    pub free_memory: Option<usize>,
    pub compute_capability: Option<String>,
    pub unified_memory: bool,
}
```

## Adding a New Backend

1. Create a new module `src/kernels/mybackend/mod.rs`
2. Implement the `KernelBackend` trait
3. Add feature flag to `Cargo.toml`
4. Update `Backend` enum and `init_backend()` in `src/kernels/backend.rs`
5. Add exports to `src/lib.rs`

Example structure:

```rust
#[cfg(feature = "mybackend")]
pub struct MyBackend {
    device: MyDevice,
    // ...
}

#[cfg(feature = "mybackend")]
impl KernelBackend for MyBackend {
    type Tensor1 = MyTensor1;
    type Tensor2 = MyTensor2;
    
    fn name(&self) -> &'static str { "mybackend" }
    fn is_available() -> bool { /* check availability */ }
    
    // ... implement all trait methods
}
```
