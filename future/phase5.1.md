# Phase 5.1: CUDA Backend Implementation

**Status**: ✅ Completed  
**Impact**: SPEED+++ (10-100x vs CPU)  
**Platform**: Linux, Windows (requires NVIDIA GPU + CUDA Toolkit)

---

## Overview

This phase implements GPU-accelerated inference for NVIDIA GPUs using CUDA via the `cudarc` crate. The CUDA backend provides significant speedups over CPU inference by leveraging:

- **cuBLAS**: Optimized matrix multiplication routines
- **Custom CUDA kernels**: Transformer-specific operations (RMSNorm, softmax, SiLU, RoPE)
- **GPU memory management**: Buffer pooling for efficient memory reuse

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

## Files Created

### `src/kernels/cuda/mod.rs`

Main CUDA backend implementation:

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

**Implemented `KernelBackend` trait**:
- `matmul_vec()` - Matrix-vector multiplication via cuBLAS GEMV
- `matmul()` - Matrix-matrix multiplication via cuBLAS GEMM
- `rmsnorm()` - RMS normalization (custom kernel)
- `softmax()` - Softmax with numerical stability (custom kernel)
- `silu()` - SiLU activation (custom kernel)
- `apply_rope()` - Rotary position embeddings (custom kernel)
- `compute_attention_scores()` - Q·K^T computation (custom kernel)
- `weighted_sum_rows()` - Attention output (custom kernel)

### `src/kernels/cuda/tensor.rs`

GPU tensor types:

```rust
pub struct CudaTensor {
    data: CudaSlice<f32>,
    shape: Vec<usize>,
    len: usize,
}

pub struct CudaTensorBatch { ... }
pub struct CudaKVCache { ... }
```

### `src/kernels/cuda/memory.rs`

GPU memory management:

```rust
pub struct CudaMemoryPool {
    free_buffers: BTreeMap<usize, Vec<CudaSlice<f32>>>,
    // ... statistics
}
```

**Features**:
- Buffer pooling with power-of-2 size buckets
- Hit rate tracking for optimization insights
- Memory estimation utilities

**Utility Functions**:
- `estimate_model_memory_mb()` - Estimate model VRAM usage
- `estimate_kv_cache_memory_mb()` - Estimate KV cache VRAM usage

### `src/kernels/cuda/kernels.rs`

Kernel launch utilities and additional kernel sources:

- `optimal_block_size()` - Calculate optimal CUDA block size
- `num_blocks()` - Calculate grid dimensions
- `launch_config_1d/2d()` - Create launch configurations
- `FP16_KERNELS_SOURCE` - Half-precision kernel source (for future use)
- `QUANTIZED_KERNELS_SOURCE` - INT4/INT8 kernel source (for future use)
- `FLASH_ATTENTION_KERNELS_SOURCE` - Flash attention kernel (for future use)

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
extern "C" __global__ void silu_kernel(
    const float* x, float* y, int n
);
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

### Attention Scores Kernel
```cuda
extern "C" __global__ void attention_scores_kernel(
    const float* query, const float* keys, float* scores,
    int seq_len, int head_dim, float scale
);
```
- Computes: `scores[i] = (query · keys[i]) * scale`

### Weighted Sum Kernel
```cuda
extern "C" __global__ void weighted_sum_kernel(
    const float* weights, const float* matrix, float* out,
    int n, int d
);
```
- Computes: `out[j] = Σᵢ weights[i] * matrix[i,j]`

## Usage

### Building with CUDA Support

```bash
# Requires CUDA Toolkit installed
cargo build --release --features cuda
```

### Using the CUDA Backend

```rust
use torchless::{CudaBackend, KernelBackend, Backend, init_backend, BackendPreference};

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

### Memory Estimation

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

## Dependencies

Added to `Cargo.toml`:

```toml
[dependencies.cudarc]
version = "0.11"
optional = true
default-features = false
features = ["cuda-12050", "cublas", "driver", "nvrtc"]

[features]
cuda = ["cudarc"]
```

## Files Modified

| File | Changes |
|------|---------|
| `src/kernels/mod.rs` | Added `cuda` module, re-exported types |
| `src/kernels/backend.rs` | Updated `Backend` enum dispatch for CUDA |
| `src/lib.rs` | Added CUDA exports |
| `Cargo.toml` | Added cudarc dependency with features |

## Performance Expectations

| Operation | CPU (7B model) | CUDA (7B model) | Speedup |
|-----------|----------------|-----------------|---------|
| Token generation | ~1 tok/s | 20-50 tok/s | 20-50x |
| Prompt processing | ~10 tok/s | 500-1000 tok/s | 50-100x |
| Matrix multiply | baseline | cuBLAS optimized | 10-100x |

*Actual performance depends on GPU model and batch size.*

## Requirements

- **NVIDIA GPU**: Compute capability 5.0+ (Maxwell or newer)
- **CUDA Toolkit**: Version 12.0+ recommended
- **Driver**: Compatible with CUDA toolkit version
- **OS**: Linux or Windows

## Future Enhancements

The implementation includes kernel source code for future features:

1. **FP16 Operations** (`FP16_KERNELS_SOURCE`)
   - Half-precision kernels for 2x memory reduction
   - FP16↔FP32 conversion kernels

2. **Quantized Operations** (`QUANTIZED_KERNELS_SOURCE`)
   - Fused INT8 dequantize + matmul
   - Fused INT4 dequantize + matmul
   - Q4_0 block dequantization (GGUF compatible)

3. **Flash Attention** (`FLASH_ATTENTION_KERNELS_SOURCE`)
   - O(N) memory complexity
   - Tiled computation with online softmax

## Testing

CUDA tests are marked with `#[ignore]` and only run when CUDA is available:

```bash
# Run CUDA tests (requires GPU)
cargo test --features cuda -- --ignored
```

Test coverage:
- `test_cuda_availability` - Checks availability detection
- `test_cuda_backend_creation` - Backend initialization
- `test_cuda_matmul_vec` - Matrix-vector multiplication
- `test_cuda_softmax` - Softmax correctness
- `test_cuda_silu` - SiLU activation correctness
