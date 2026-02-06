# OpenCL Backend (Cross-Platform)

Cross-platform GPU acceleration via OpenCL.

**Impact:** SPEED++ (10-50x vs CPU)  
**Platform:** Linux, macOS, Windows (any OpenCL-capable GPU)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenCLBackend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Context      │  │ Queue        │  │ Program      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│          │                │                 │              │
│          └────────────────┼─────────────────┘              │
│                           ▼                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                  OpenCL Compute Kernels              │ │
│  │  - RMSNorm      - Softmax       - SiLU               │ │
│  │  - RoPE         - Attention     - Weighted Sum       │ │
│  │  - MatMul       - Fused Ops     - Element-wise       │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Building

```bash
cargo build --release --features opencl
```

## Usage

```rust
use torchless::{init_backend, BackendPreference};

// Auto-select best available backend
let backend = init_backend(BackendPreference::Auto)?;
println!("Using backend: {}", backend.name());

// Or explicitly request OpenCL
let backend = init_backend(BackendPreference::OpenCL)?;

// Use backend for operations
let result = backend.matmul_vec(&weights, &input);
```

### Direct OpenCLBackend Usage

```rust
use torchless::kernels::opencl::OpenCLBackend;

// List available OpenCL devices
let devices = OpenCLBackend::list_devices();
for (platform, device, idx) in &devices {
    println!("[{}] {} - {}", idx, platform, device);
}

// Create backend with specific device
if OpenCLBackend::is_available() {
    let backend = OpenCLBackend::new()?;  // First device
    // Or: let backend = OpenCLBackend::with_device_index(1)?;  // Specific device
    
    println!("OpenCL device: {} ({})", backend.device_name(), backend.platform_name());
    
    // Create GPU buffers
    let tensor = backend.to_device_1d(&array)?;
    
    // Run operations
    backend.softmax(&mut x);
    backend.rmsnorm(&mut x, &weight, 1e-5);
}
```

## OpenCL Kernels

| Kernel | Description |
|--------|-------------|
| `rmsnorm_kernel` | RMSNorm with parallel reduction |
| `softmax_kernel` | Numerically stable softmax |
| `silu_kernel` | SiLU (Swish) activation |
| `rope_kernel` | Rotary Position Embedding |
| `attention_scores_kernel` | Q @ K^T computation |
| `weighted_sum_kernel` | Attention-weighted values |
| `matmul_vec_kernel` | Matrix-vector multiplication |
| `matmul_vec_tiled_kernel` | Tiled matmul with local memory |
| `fused_silu_mul_kernel` | Fused SiLU + multiply |
| `matmul_kernel` | Matrix-matrix multiplication |
| `elementwise_mul_kernel` | Element-wise multiplication |
| `elementwise_add_kernel` | Element-wise addition |

## Backend Selection Priority

OpenCL serves as a fallback in the backend hierarchy:

1. CUDA (NVIDIA) - best for NVIDIA GPUs
2. ROCm (AMD) - best for AMD GPUs
3. Metal (Apple) - best for Apple Silicon
4. **OpenCL - cross-platform fallback**
5. CPU (always available)

## Performance

### Pros
- **Cross-platform**: Works on virtually any GPU (NVIDIA, AMD, Intel)
- **Portable**: Same code runs on different vendors' hardware
- **Wide support**: Even older GPUs typically have OpenCL drivers

### Cons
- **Performance overhead**: Generally slower than vendor-specific APIs
- **Driver quality**: OpenCL driver quality varies by vendor

### Expectations

| Operation | CPU (baseline) | OpenCL (estimated) |
|-----------|----------------|-------------------|
| matmul_vec (4096x4096) | 1x | 10-30x |
| matmul (batched) | 1x | 20-50x |
| softmax | 1x | 5-20x |
| RMSNorm | 1x | 5-20x |

## Platform Support

| Platform | OpenCL Support |
|----------|---------------|
| NVIDIA GPUs | Via NVIDIA drivers |
| AMD GPUs | Via AMD drivers |
| Intel GPUs | Via Intel OpenCL runtime |
| Intel CPUs | Via Intel OpenCL runtime |
| AMD CPUs | Limited (via AMD APP SDK) |
| Apple (M1/M2/M3) | Deprecated (prefer Metal) |

## Requirements

### Software
- OpenCL 1.2+ runtime
- Vendor-specific OpenCL driver

### Installation

**Ubuntu/Debian (NVIDIA):**
```bash
sudo apt install nvidia-opencl-dev
```

**Ubuntu/Debian (Intel):**
```bash
sudo apt install intel-opencl-icd
```

**macOS:**
OpenCL is included with macOS (but deprecated, prefer Metal)

**Windows:**
OpenCL is typically included with GPU drivers

## Testing

```bash
cargo test --features opencl --lib kernels::opencl

# Include GPU hardware tests:
cargo test --features opencl --lib kernels::opencl -- --ignored
```
