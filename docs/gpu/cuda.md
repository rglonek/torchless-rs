# CUDA Backend (NVIDIA)

GPU-accelerated inference for NVIDIA GPUs using CUDA.

**Impact:** SPEED+++ (10-100x vs CPU)  
**Platform:** Linux, Windows

> **Note:** The CUDA backend has full kernel implementations. However, `UnifiedTensor::to_device()` is not yet implemented—use `CudaBackend::to_device_1d/2d()` for explicit data transfer.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CudaBackend                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ CudaDevice   │  │ CudaBlas     │  │ CudaKernels  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│          │                │                 │              │
│          └────────────────┼─────────────────┘              │
│                           ▼                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                Custom CUDA Kernels                    │ │
│  │  - RMSNorm      - Softmax       - SiLU               │ │
│  │  - RoPE         - Attention     - Weighted Sum       │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Building

```bash
# Requires CUDA Toolkit installed
cargo build --release --features cuda
```

## Usage

```rust
use torchless::{CudaBackend, KernelBackend, init_backend, BackendPreference};

// Option 1: Direct usage
if CudaBackend::is_available() {
    let backend = CudaBackend::new()?;
    let result = backend.matmul_vec(&weights, &input);
}

// Option 2: Via Backend enum (auto-selects best available)
let backend = init_backend(BackendPreference::Auto)?;
println!("Using: {}", backend.name()); // "cuda" if available

// Option 3: Explicitly request CUDA
let backend = init_backend(BackendPreference::Cuda)?;
```

## CudaBackend

```rust
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    cublas: Arc<CudaBlas>,
    kernels: Arc<CudaKernels>,
    memory_pool: Arc<Mutex<CudaMemoryPool>>,
    device_index: usize,
}
```

**Key Methods**:
- `CudaBackend::new()` - Create backend on device 0
- `CudaBackend::with_device(idx)` - Create on specific device
- `CudaBackend::is_available()` - Check CUDA availability
- `to_device_1d/2d()` - Transfer CPU arrays to GPU
- `to_host_1d/2d()` - Transfer GPU tensors to CPU

## Implemented Operations

**Via KernelBackend trait:**
- `matmul_vec()` - Matrix-vector multiplication via cuBLAS GEMV
- `matmul()` - Matrix-matrix multiplication via cuBLAS GEMM
- `rmsnorm()` - RMS normalization (custom kernel)
- `softmax()` - Softmax with numerical stability (custom kernel)
- `silu()` - SiLU activation (custom kernel)
- `apply_rope()` - Rotary position embeddings (custom kernel)
- `compute_attention_scores()` - Q·K^T computation (custom kernel)
- `weighted_sum_rows()` - Attention output (custom kernel)

## Custom CUDA Kernels

All custom kernels are compiled at runtime via NVRTC:

### RMSNorm Kernel
```cuda
extern "C" __global__ void rmsnorm_kernel(
    float* x, const float* weight, int n, float eps
);
```
- Uses shared memory for parallel reduction
- Computes: `x = x * weight / sqrt(mean(x²) + eps)`

### Softmax Kernel
```cuda
extern "C" __global__ void softmax_kernel(float* x, int n);
```
- Numerically stable (subtracts max)
- Two-pass reduction: max finding, then normalization

### SiLU Kernel
```cuda
extern "C" __global__ void silu_kernel(const float* x, float* y, int n);
```
- Computes: `y = x / (1 + exp(-x))`
- Fully parallel, no shared memory needed

### RoPE Kernel
```cuda
extern "C" __global__ void rope_kernel(
    float* x, const float* cos, const float* sin,
    int n_heads, int head_dim, int half
);
```
- Half-split layout for Mistral compatibility
- Applies rotation to query/key tensors

## Memory Estimation

```rust
use torchless::estimate_model_memory_mb;

let model_mb = estimate_model_memory_mb(
    4096,   // hidden_size
    14336,  // intermediate_size
    32,     // n_layers
    32000,  // vocab_size
    32,     // n_heads
    8,      // n_kv_heads
);
println!("Model VRAM: {:.1} GB", model_mb / 1024.0);
```

## Performance

| Operation | CPU (7B model) | CUDA (7B model) | Speedup |
|-----------|----------------|-----------------|---------|
| Token generation | ~1 tok/s | 20-50 tok/s | 20-50x |
| Prompt processing | ~10 tok/s | 500-1000 tok/s | 50-100x |
| Matrix multiply | baseline | cuBLAS optimized | 10-100x |

## Requirements

- **NVIDIA GPU**: Compute capability 5.0+ (Maxwell or newer)
- **CUDA Toolkit**: Version 12.0+ recommended
- **Driver**: Compatible with CUDA toolkit version
- **OS**: Linux or Windows

## Testing

```bash
# Run CUDA tests (requires GPU)
cargo test --features cuda -- --ignored
```
