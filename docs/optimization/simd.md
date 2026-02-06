# SIMD Optimizations

Architecture-specific SIMD optimizations for significant CPU performance improvements.

## Performance Hierarchy

| Priority | Backend | SIMD Width | Platform | Expected Speedup |
|----------|---------|------------|----------|------------------|
| 1 | AVX-512F | 16 floats | x86_64 (Skylake-X+, Ice Lake+, Zen 4+) | ~2x vs AVX2 |
| 2 | NEON | 4 floats (16 unrolled) | aarch64 (Apple M1/M2/M3, Graviton) | ~1.5x vs scalar |
| 3 | wide crate | 8 floats | Portable (with `simd` feature) | ~1.3x vs scalar |
| 4 | Rayon | N/A | Any (with `parallel` feature) | Scales with cores |
| 5 | Scalar | 1 float | Any | Baseline |

## AVX-512 Kernels

Hand-tuned AVX-512 implementations using `std::arch::x86_64` intrinsics for 16-wide SIMD operations.

### Implemented Operations

```rust
// Runtime detection
pub fn is_avx512_available() -> bool;

// Matrix operations
pub unsafe fn matmul_vec_avx512(weights: &[f32], input: &[f32], output: &mut [f32], rows: usize, cols: usize);
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32;

// Normalization
pub unsafe fn rmsnorm_avx512(x: &mut [f32], weight: &[f32], eps: f32);
pub unsafe fn softmax_avx512(x: &mut [f32]);

// Activations
pub unsafe fn silu_avx512(input: &[f32], output: &mut [f32]);

// Positional encoding
pub unsafe fn apply_rope_avx512(x: &mut [f32], cos: &[f32], sin: &[f32], n_heads: usize, head_dim: usize);

// Attention
pub unsafe fn compute_attention_scores_avx512(query: &[f32], keys: &[f32], scores: &mut [f32], n_keys: usize, key_dim: usize, scale: f32);
pub unsafe fn weighted_sum_rows_avx512(weights: &[f32], matrix: &[f32], output: &mut [f32], n_rows: usize, n_cols: usize);
```

### Key Optimizations

- Uses `_mm512_fmadd_ps` for fused multiply-add operations
- `_mm512_reduce_add_ps` for efficient horizontal reductions
- Loop unrolling with 16-element stride
- Scalar fallback for remainder elements

### Supported CPUs

- Intel Skylake-X and later server CPUs
- Intel Ice Lake and later client CPUs
- AMD Zen 4 and later

## ARM NEON Kernels

Optimized NEON implementations using `std::arch::aarch64` intrinsics, particularly tuned for Apple Silicon cache hierarchies.

### Implemented Operations

```rust
// Runtime detection (always true on aarch64)
pub fn is_neon_available() -> bool;

// Same operations as AVX-512 with NEON intrinsics
pub unsafe fn matmul_vec_neon(...);
pub unsafe fn rmsnorm_neon(...);
pub unsafe fn softmax_neon(...);
pub unsafe fn silu_neon(...);
pub unsafe fn apply_rope_neon(...);
pub unsafe fn dot_product_neon(...);
pub unsafe fn compute_attention_scores_neon(...);
pub unsafe fn weighted_sum_rows_neon(...);
```

### Key Optimizations

- 4-wide base operations with 16-element loop unrolling (4 NEON registers)
- Uses `vfmaq_f32` for fused multiply-add
- `vaddvq_f32` for efficient horizontal sum
- Optimized for M1/M2/M3 unified memory architecture

### Supported CPUs

- Apple M1, M2, M3 series
- ARM Cortex-A series with NEON
- AWS Graviton processors

## Runtime CPU Dispatch

Automatic detection and routing to the optimal kernel implementation.

### CPU Feature Detection

```rust
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub avx512f: bool,  // AVX-512F (x86_64)
    pub avx2: bool,     // AVX2 (x86_64)
    pub fma: bool,      // FMA (x86_64)
    pub neon: bool,     // NEON (aarch64)
}

impl CpuFeatures {
    pub fn detect() -> Self;
    pub fn best_simd_width(&self) -> usize;  // Returns 16, 8, 4, or 1
    pub fn describe(&self) -> String;        // Human-readable description
}
```

### Dispatcher

```rust
pub struct Dispatcher {
    features: CpuFeatures,
}

impl Dispatcher {
    pub fn new() -> Self;                    // Auto-detect features
    pub fn with_features(features: CpuFeatures) -> Self;  // Manual override
    
    // Auto-dispatched operations
    pub fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32>;
    pub fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>);
    pub fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32);
    pub fn softmax(&self, x: &mut Array1<f32>);
    pub fn silu(&self, x: &Array1<f32>) -> Array1<f32>;
    pub fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>);
}
```

### Usage

```rust
use torchless::kernels::{cpu_features, dispatch_matmul_vec, print_cpu_features};

fn main() {
    // Print detected CPU features
    print_cpu_features();
    
    // Check specific features
    let features = cpu_features();
    println!("AVX-512 available: {}", features.avx512f);
    println!("NEON available: {}", features.neon);
    
    // Use auto-dispatched operations
    let weights = Array2::from_shape_fn((128, 64), |(i, j)| (i * 64 + j) as f32 * 0.01);
    let input = Array1::from_vec((0..64).map(|i| i as f32 * 0.1).collect());
    
    let output = dispatch_matmul_vec(&weights, &input);
}
```

## Fused Kernels

Memory-efficient fused operations that combine multiple steps into single passes.

### Fused RMSNorm + Linear

```rust
/// Combines: y = Linear(RMSNorm(x, weight_norm, eps), weight_proj)
pub fn fused_rmsnorm_linear(
    x: &Array1<f32>,
    weight_norm: &Array1<f32>,
    weight_proj: &Array2<f32>,
    eps: f32,
) -> Array1<f32>;
```

### Fused SwiGLU

```rust
/// Combines: output = SiLU(gate_proj(x)) * up_proj(x)
pub fn fused_swiglu(
    x: &Array1<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
) -> Array1<f32>;
```

### Fused MLP

```rust
/// Complete MLP forward: down_proj(SiLU(gate_proj(x)) * up_proj(x))
pub fn fused_mlp(
    x: &Array1<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
    down_proj: &Array2<f32>,
) -> Array1<f32>;
```

### Fused Attention

```rust
/// Computes Q*K^T + softmax + V weighted sum
pub fn fused_attention(
    query: ArrayView1<f32>,
    keys: ArrayView2<f32>,
    values: ArrayView2<f32>,
    scale: f32,
    seq_len: usize,
) -> Array1<f32>;
```

## Platform Support Matrix

| Backend | Linux | macOS | Windows |
|---------|-------|-------|---------|
| CPU (scalar) | Yes | Yes | Yes |
| AVX-512F | Yes | No* | Yes |
| NEON | Yes (ARM) | Yes | No |

*macOS on Intel does not support AVX-512

## Performance Notes

1. **Memory Bandwidth**: Fused kernels reduce memory bandwidth by 2-3x by avoiding intermediate allocations.

2. **Cache Efficiency**: NEON kernels are tuned for Apple Silicon's unified memory architecture.

3. **Remainder Handling**: All SIMD kernels gracefully handle non-aligned sizes with scalar fallback for remainder elements.

4. **Thread Safety**: All dispatchers and feature detection are thread-safe using `OnceLock`.

5. **Zero-Cost Abstraction**: When features are known at compile time, the dispatch overhead is eliminated.
