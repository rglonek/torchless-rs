# WebGPU Backend (Cross-Platform)

GPU-accelerated inference using WebGPU via the wgpu library. Runs on Vulkan (Linux/Windows), DirectX 12 (Windows), or Metal (macOS).

**Impact:** SPEED+++ (10-50x vs CPU)  
**Platform:** Linux, Windows, macOS (via wgpu)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     WebGPUBackend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ wgpu::Device │  │ wgpu::Queue  │  │ WGSL Shaders │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│          │                │                 │              │
│          └────────────────┼─────────────────┘              │
│                           ▼                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                    wgpu Runtime                       │ │
│  │         Vulkan │ DirectX 12 │ Metal                   │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Building

```bash
cargo build --release --features webgpu
```

## Usage

```rust
use torchless::{init_backend, BackendPreference};

// Auto-select best available backend (WebGPU used after other GPU backends)
let backend = init_backend(BackendPreference::Auto)?;
println!("Using backend: {}", backend.name());

// Or explicitly request WebGPU
let backend = init_backend(BackendPreference::WebGPU)?;

// Use backend for operations
let result = backend.matmul_vec(&weights, &input);
```

### Direct WebGPUBackend Usage

```rust
use torchless::kernels::webgpu::WebGPUBackend;

if WebGPUBackend::is_available() {
    let backend = WebGPUBackend::new()?;
    println!("WebGPU device: {}", backend.device_name());

    // Create GPU buffers (staging buffers for CPU-GPU transfer)
    let buffer = backend.create_buffer(&array);

    // Run operations via KernelBackend trait
    let result = backend.matmul_vec(&weights, &input);
}
```

## WGSL Compute Shaders

| Kernel | Description |
|--------|-------------|
| `rmsnorm_kernel` | RMSNorm with parallel reduction |
| `softmax_kernel` | Numerically stable softmax |
| `silu_kernel` | SiLU (Swish) activation |
| `rope_kernel` | Rotary Position Embedding |
| `attention_scores_kernel` | Q @ K^T computation |
| `weighted_sum_kernel` | Attention-weighted values |
| `matmul_vec_kernel` | Matrix-vector multiplication |
| `elementwise_mul_kernel` | Element-wise multiplication |
| `elementwise_add_kernel` | Element-wise addition |

## Performance Considerations

### Staging Buffers
- Uses staging buffers for CPU-GPU data transfer (unlike Metal's unified memory)
- Explicit copy steps add latency for small operations
- Best for batch workloads where transfer overhead is amortized

### API Translation
- wgpu translates to native GPU APIs: Vulkan on Linux/Windows, Metal on macOS, DirectX 12 on Windows
- Some overhead compared to vendor-specific backends

### When to Choose WebGPU
- **Use WebGPU when:** No CUDA, ROCm, or Metal available — it works everywhere
- **Prefer CUDA/ROCm/Metal when:** You have vendor hardware — they offer better optimizations
- **Best choice:** Any GPU with Vulkan, DX12, or Metal support when no vendor-specific backend is available

## Cross-Platform Compatibility

| Platform | Backend | Requirements |
|----------|---------|--------------|
| Linux | Vulkan | Vulkan 1.1+ driver |
| Windows | DirectX 12 or Vulkan | DX12 or Vulkan driver |
| macOS | Metal | Metal-capable GPU (built-in) |

## Selection Priority

WebGPU sits at lowest priority in auto-selection (position 5), after CUDA, ROCm, Metal, and OpenCL, but before CPU fallback.

## Testing

```bash
cargo test --features webgpu --lib kernels::webgpu

# Include GPU hardware tests:
cargo test --features webgpu --lib kernels::webgpu -- --ignored
```
