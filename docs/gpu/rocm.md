# ROCm Backend (AMD)

GPU-accelerated inference for AMD GPUs using ROCm/HIP.

**Impact:** SPEED+++ (10-100x vs CPU)  
**Platform:** Linux only

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

## Building

```bash
# Requires ROCm installed
cargo build --release --features rocm
```

## Usage

```rust
use torchless::{init_backend, BackendPreference};

// Auto-detect best backend (prefers GPU)
let backend = init_backend(BackendPreference::Auto)?;

// Explicitly request ROCm
#[cfg(feature = "rocm")]
{
    use torchless::RocmBackend;
    
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

## Backend Selection

ROCm is automatically detected in the hierarchy:

1. Try CUDA (NVIDIA)
2. **Try ROCm (AMD)**
3. Try Metal (Apple)
4. Try OpenCL
5. Fall back to CPU

## HIP Kernel Source

HIP kernels use CUDA-compatible syntax:

```cpp
// RMSNorm kernel
extern "C" __global__ void rmsnorm_kernel(
    float* x, const float* weight, int n, float eps
) {
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
```

## Performance

| Operation | CPU (baseline) | ROCm (estimated) |
|-----------|----------------|------------------|
| matmul_vec (4096x4096) | 1x | 20-50x |
| matmul (batched) | 1x | 50-100x |
| softmax | 1x | 10-30x |
| RMSNorm | 1x | 10-30x |

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

## Testing

```bash
# Run ROCm-specific tests (requires ROCm hardware)
cargo test --features rocm -- --ignored

# Tests that run without hardware
cargo test rocm
```
