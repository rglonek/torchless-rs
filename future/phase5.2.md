# Phase 5.2: ROCm Backend (AMD GPUs)

**Status:** ✅ Implemented  
**Impact:** SPEED+++ (10-100x vs CPU on AMD hardware)  
**Platform:** Linux only

## Overview

The ROCm backend provides GPU-accelerated inference for AMD GPUs using the HIP (Heterogeneous-compute Interface for Portability) runtime and rocBLAS library. HIP is AMD's CUDA-like API that enables portable GPU code.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RocmBackend                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ HipDevice    │  │ HipStream    │  │ RocblasHandle│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│          │                │                 │              │
│          └────────────────┼─────────────────┘              │
│                           ▼                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                  HipKernels                           │ │
│  │  - RMSNorm      - Softmax       - SiLU               │ │
│  │  - RoPE         - Attention     - Weighted Sum       │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Files Created

### `src/kernels/rocm/mod.rs`
Main backend implementation containing:
- `RocmBackend` struct implementing `KernelBackend` trait
- HIP FFI bindings for device management and memory operations
- rocBLAS FFI bindings for BLAS operations
- `HipSlice<T>` wrapper for device memory pointers
- CPU-GPU data transfer methods

### `src/kernels/rocm/tensor.rs`
GPU tensor types:
- `RocmTensor` - Single tensor on GPU memory with shape tracking
- `RocmTensorBatch` - Batched tensors for efficient batch operations
- `RocmKVCache` - Key-value cache for attention layers

### `src/kernels/rocm/memory.rs`
Memory management utilities:
- `RocmMemoryPool` - Buffer pooling to reduce allocation overhead
- `PinnedBuffer` - Page-locked host memory for faster transfers
- Memory estimation functions for model and KV cache sizing

### `src/kernels/rocm/kernels.rs`
Kernel utilities and source code:
- Launch configuration helpers optimized for AMD wavefront size (64)
- HIP kernel source code (CUDA-compatible syntax)
- FP16 kernel templates
- Quantized kernel templates (INT4/INT8)

## Files Modified

| File | Changes |
|------|---------|
| `Cargo.toml` | Added `rocm = []` feature flag |
| `src/kernels/mod.rs` | Added rocm module and re-exports |
| `src/kernels/backend.rs` | Integrated ROCm into Backend enum and init_backend |
| `src/lib.rs` | Added public ROCm type exports |

## Implementation Details

### HIP FFI Bindings

```rust
#[link(name = "amdhip64")]
extern "C" {
    pub fn hipGetDeviceCount(count: *mut c_int) -> c_int;
    pub fn hipSetDevice(device: c_int) -> c_int;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    pub fn hipFree(ptr: *mut c_void) -> c_int;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: HipMemcpyKind) -> c_int;
    pub fn hipDeviceSynchronize() -> c_int;
    // ... more functions
}
```

### rocBLAS Integration

Matrix operations use rocBLAS for hardware acceleration:

```rust
#[link(name = "rocblas")]
extern "C" {
    pub fn rocblas_sgemv(...) -> c_int;  // Matrix-vector multiply
    pub fn rocblas_sgemm(...) -> c_int;  // Matrix-matrix multiply
}
```

### Backend Selection

ROCm is automatically detected in the backend selection hierarchy:

```rust
pub fn init_backend(preference: BackendPreference) -> Result<Backend> {
    match preference {
        BackendPreference::Auto => {
            // 1. Try CUDA first (NVIDIA)
            // 2. Try ROCm (AMD)
            // 3. Try Metal (Apple)
            // 4. Try OpenCL
            // 5. Fall back to CPU
        }
        BackendPreference::Rocm => Ok(Backend::Rocm(RocmBackend::new()?)),
        // ...
    }
}
```

## HIP Kernel Source

The implementation includes ready-to-compile HIP kernels:

```cpp
// RMSNorm kernel
extern "C" __global__ void rmsnorm_kernel(float* x, const float* weight, int n, float eps) {
    extern __shared__ float shared[];
    // ... parallel sum of squares
    // ... reduction for RMS
    // ... apply normalization
}

// SiLU activation
extern "C" __global__ void silu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + expf(-val));
    }
}

// ... more kernels for softmax, RoPE, attention, etc.
```

## Usage

### Enabling ROCm Support

```bash
# Build with ROCm feature
cargo build --release --features rocm
```

### Using the Backend

```rust
use torchless::kernels::backend::{init_backend, BackendPreference};

// Auto-detect best backend (prefers GPU)
let backend = init_backend(BackendPreference::Auto)?;

// Explicitly request ROCm
#[cfg(feature = "rocm")]
{
    use torchless::BackendPreference;
    let backend = init_backend(BackendPreference::Rocm)?;
    println!("Using device: {}", backend.device_name());
}
```

### Direct Backend Usage

```rust
#[cfg(feature = "rocm")]
{
    use torchless::RocmBackend;
    use torchless::kernels::KernelBackend;
    
    if RocmBackend::is_available() {
        let backend = RocmBackend::new()?;
        
        // Matrix-vector multiply on GPU
        let result = backend.matmul_vec(&weights, &input);
        
        // Other operations
        backend.softmax(&mut logits);
        let activated = backend.silu(&hidden);
    }
}
```

## Requirements

### Hardware
- AMD GPU with ROCm support:
  - RDNA architecture (RX 5000/6000/7000 series)
  - CDNA architecture (MI100, MI200, MI300 series)
  - GCN architecture (older Vega, Polaris with limited support)

### Software
- Linux operating system (Ubuntu 20.04/22.04, RHEL 8/9, SLES 15)
- ROCm runtime 5.x or 6.x
- rocBLAS library
- hipcc compiler (for custom kernel compilation)

### Installation (Ubuntu)

```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Verify installation
rocminfo
hipcc --version
```

## Performance Characteristics

| Operation | CPU (baseline) | ROCm (estimated) |
|-----------|----------------|------------------|
| matmul_vec (4096x4096) | 1x | 20-50x |
| matmul (batched) | 1x | 50-100x |
| softmax | 1x | 10-30x |
| RMSNorm | 1x | 10-30x |

*Actual performance varies by GPU model and batch size.*

## Current Limitations

1. **Kernel compilation**: Custom HIP kernels require runtime compilation with hiprtc (not yet implemented). Currently, non-BLAS operations fall back to CPU.

2. **Memory transfer overhead**: Each operation transfers data CPU↔GPU. For optimal performance, keep tensors on GPU (future improvement).

3. **FP16 support**: FP16 kernels are defined but not yet integrated into the main code path.

4. **Quantization**: INT4/INT8 kernels are defined but not yet connected to the quantization system.

## Future Improvements

1. **hiprtc integration**: Runtime kernel compilation similar to CUDA's nvrtc
2. **Persistent GPU tensors**: Keep model weights on GPU to avoid transfer overhead
3. **Multi-GPU support**: Tensor parallelism across multiple AMD GPUs
4. **Async transfers**: Overlap computation with data transfer using HIP streams
5. **FP16 inference**: Enable half-precision for memory savings and potential speedup
6. **Quantized inference**: Fused dequantize+matmul for INT4/INT8 models

## Testing

```bash
# Run ROCm-specific tests (requires ROCm hardware)
cargo test --features rocm -- --ignored

# Tests that run without hardware
cargo test rocm
```

## Comparison with Other Backends

| Feature | CUDA | ROCm | Metal | OpenCL |
|---------|------|------|-------|--------|
| Vendor | NVIDIA | AMD | Apple | Cross-platform |
| BLAS Library | cuBLAS | rocBLAS | MPS | clBLAS |
| Runtime Compilation | nvrtc | hiprtc | MSL | OpenCL C |
| Unified Memory | CUDA Unified | HSA | Metal Shared | SVM |
| Platform | Linux/Windows | Linux | macOS | All |

## References

- [ROCm Documentation](https://rocm.docs.amd.com/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [rocBLAS Documentation](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/)
- [AMD GPU Architecture](https://gpuopen.com/learn/amd-gcn-isa-documentation/)
