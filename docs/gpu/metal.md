# Metal Backend (Apple Silicon)

GPU-accelerated inference on Apple Silicon using Metal compute shaders.

**Impact:** SPEED+++ (10-50x vs CPU)  
**Platform:** macOS (Apple Silicon M1/M2/M3, Intel Macs with Metal support)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MetalBackend                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Device       │  │ CommandQueue │  │ Library      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│          │                │                 │              │
│          └────────────────┼─────────────────┘              │
│                           ▼                                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │                  Metal Compute Shaders                │ │
│  │  - RMSNorm      - Softmax       - SiLU               │ │
│  │  - RoPE         - Attention     - Weighted Sum       │ │
│  │  - MatMul       - Fused Ops                          │ │
│  └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Building

```bash
cargo build --release --features metal-gpu
```

## Usage

```rust
use torchless::{init_backend, BackendPreference};

// Auto-select best available backend (Metal on macOS)
let backend = init_backend(BackendPreference::Auto)?;
println!("Using backend: {}", backend.name());

// Or explicitly request Metal
let backend = init_backend(BackendPreference::Metal)?;

// Use backend for operations
let result = backend.matmul_vec(&weights, &input);
```

### Direct MetalBackend Usage

```rust
use torchless::kernels::metal::MetalBackend;

if MetalBackend::is_available() {
    let backend = MetalBackend::new()?;
    println!("Metal device: {}", backend.device_name());
    
    // Create GPU buffers (unified memory - no copy needed)
    let tensor = backend.to_device_1d(&array)?;
    
    // Run operations
    backend.softmax(&mut x);
    backend.rmsnorm(&mut x, &weight, 1e-5);
}
```

## Metal Compute Shaders

| Kernel | Description |
|--------|-------------|
| `rmsnorm_kernel` | RMSNorm with parallel reduction |
| `softmax_kernel` | Numerically stable softmax |
| `silu_kernel` | SiLU (Swish) activation |
| `rope_kernel` | Rotary Position Embedding |
| `attention_scores_kernel` | Q @ K^T computation |
| `weighted_sum_kernel` | Attention-weighted values |
| `matmul_vec_kernel` | Matrix-vector multiplication |
| `matmul_vec_tiled_kernel` | Tiled matmul with shared memory |
| `fused_silu_mul_kernel` | Fused SiLU + multiply for SwiGLU |
| `matmul_kernel` | Matrix-matrix multiplication |

## Unified Memory Architecture

Apple Silicon uses unified memory, allowing both CPU and GPU to access the same memory without explicit copies:

```rust
// Create buffer from CPU data - no copy!
let buffer = device.new_buffer_with_data(
    data.as_ptr() as *const _,
    size as u64,
    MTLResourceOptions::StorageModeShared,
);

// Read directly from GPU buffer
let slice = unsafe {
    std::slice::from_raw_parts(buffer.contents() as *const f32, len)
};
```

## Performance Considerations

### Memory Bandwidth
- Unified memory eliminates CPU-GPU transfer overhead
- M1/M2/M3 chips have high memory bandwidth (200-400 GB/s)
- Buffer pooling reduces allocation overhead

### Compute Efficiency
- Custom shaders optimized for transformer operations
- Tiled matrix multiplication for better cache utilization
- Fused operations reduce memory round-trips

### Recommendations
1. Use larger batch sizes when possible
2. Reuse buffers via memory pool
3. Consider MPS for large matrix operations
4. Profile with Instruments for optimization

## Compatibility

| Platform | Support |
|----------|---------|
| macOS (Apple Silicon) | Full support |
| macOS (Intel with discrete GPU) | Supported |
| macOS (Intel integrated) | Limited (older Metal) |
| iOS/iPadOS | Potential future support |
| Linux/Windows | Not supported |

## Testing

```bash
cargo test --features metal-gpu --lib kernels::metal

# Include GPU hardware tests:
cargo test --features metal-gpu --lib kernels::metal -- --ignored
```
