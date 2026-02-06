# GPU Acceleration Overview

GPU support provides 10-100x speedup over CPU inference.

## Current Status

All GPU backends have **full kernel implementations** for core transformer operations (matmul, rmsnorm, softmax, silu, rope, attention). However, the `DeviceTransfer` trait for `UnifiedTensor` is **not yet complete**, meaning:

- GPU operations require explicit CPU-GPU data copies
- Models cannot be loaded directly to GPU via `UnifiedTensor::to_device()`
- Use the backend-specific `to_device_1d/2d()` methods for data transfer

See [FUTURE.md](../../FUTURE.md) for the roadmap to complete unified tensor transfer.

## Supported Backends

| Backend | Vendor | Feature Flag | Platforms |
|---------|--------|--------------|-----------|
| [CUDA](cuda.md) | NVIDIA | `cuda` | Linux, Windows |
| [ROCm](rocm.md) | AMD | `rocm` | Linux |
| [Metal](metal.md) | Apple | `metal-gpu` | macOS |
| [OpenCL](opencl.md) | Cross-platform | `opencl` | All |

## Backend Selection

### Automatic Selection

```rust
use torchless::{init_backend, BackendPreference};

// Auto-select best available backend
let backend = init_backend(BackendPreference::Auto)?;
println!("Using: {}", backend.name());
```

**Selection Priority (Auto mode)**:
1. CUDA (NVIDIA GPUs)
2. ROCm (AMD GPUs) 
3. Metal (Apple Silicon)
4. OpenCL (fallback)
5. CPU (always available)

### Explicit Selection

```rust
// Force specific backend
let backend = init_backend(BackendPreference::Cuda)?;
let backend = init_backend(BackendPreference::Metal)?;
let backend = init_backend(BackendPreference::Rocm)?;
let backend = init_backend(BackendPreference::OpenCL)?;
let backend = init_backend(BackendPreference::Cpu)?;
```

## Backend Discovery

```rust
use torchless::{discover_backends, print_backend_summary};

// Print all available backends
print_backend_summary();
```

Example output:
```
=== Available Backends ===
✓ CPU (CPU)
    [0] Apple M2 Pro: 32.00 GB (28.50 GB free)
✓ Metal (Metal)
    [0] Apple M2 Pro: 21.33 GB (unified memory)
✗ CUDA (CUDA)
    Error: CUDA feature not enabled at compile time

Recommended backend: Metal
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

## Building with GPU Support

```bash
# NVIDIA CUDA
cargo build --release --features "cuda,simd,parallel"

# AMD ROCm
cargo build --release --features "rocm,simd,parallel"

# Apple Metal
cargo build --release --features "metal-gpu,simd,parallel"

# OpenCL (cross-platform)
cargo build --release --features "opencl,simd,parallel"
```

## Performance Expectations

| Operation | CPU (7B model) | GPU (7B model) | Speedup |
|-----------|----------------|----------------|---------|
| Token generation | ~1 tok/s | 20-50 tok/s | 20-50x |
| Prompt processing | ~10 tok/s | 500-1000 tok/s | 50-100x |
| Matrix multiply | baseline | optimized BLAS | 10-100x |

*Actual performance depends on GPU model and batch size.*

## Backend Comparison

| Feature | CUDA | ROCm | Metal | OpenCL |
|---------|------|------|-------|--------|
| Vendor | NVIDIA | AMD | Apple | Cross-platform |
| BLAS Library | cuBLAS | rocBLAS | MPS | clBLAS |
| Runtime Compilation | nvrtc | hiprtc | MSL | OpenCL C |
| Performance | Best for NVIDIA | Best for AMD | Best for Apple | Good fallback |
| Platform | Linux/Windows | Linux | macOS | All |

## GPU Requirements

### CUDA (NVIDIA)
- NVIDIA GPU: Compute capability 5.0+ (Maxwell or newer)
- CUDA Toolkit: Version 12.0+ recommended
- Compatible driver

### ROCm (AMD)
- AMD GPU with ROCm support (RDNA, CDNA, GCN)
- ROCm runtime 5.x or 6.x
- Linux only

### Metal (Apple)
- Apple Silicon (M1/M2/M3) or Intel Mac with discrete GPU
- macOS only

### OpenCL
- OpenCL 1.2+ runtime
- Any GPU with OpenCL driver
