# Implementation Details

Technical documentation for torchless-rs internals.

## Binary Model Format

The model uses a custom binary format:

```
┌─────────────────────────────────────┐
│ Header Size (8 bytes, little-endian u64)
├─────────────────────────────────────┤
│ JSON Header (variable length)       │
│   - metadata (config values)        │
│   - tensors (name → offset/shape)   │
│   - tokenizer (vocab, merges)       │
├─────────────────────────────────────┤
│ Padding (to 64-byte alignment)      │
├─────────────────────────────────────┤
│ Tensor Payload (binary data)        │
│   - f32: raw little-endian floats   │
│   - int8: quantized + scales        │
└─────────────────────────────────────┘
```

### Quantization

INT8 per-group symmetric quantization:

- **Group size:** 64 weights per scale
- **Storage:** `int8` weights + `f32` scales
- **Dequantization:** `weight_f32 = weight_int8 / scale`

## Transformer Architecture

Mistral 7B configuration:

| Parameter | Value |
|-----------|-------|
| Layers | 32 |
| Hidden size | 4096 |
| Intermediate size | 14336 |
| Attention heads | 32 |
| KV heads | 8 (GQA, 4x reuse) |
| Head dimension | 128 |
| Vocab size | 32000 |
| Max position | 32768 |
| RoPE theta | 10000.0 |

### Layer Structure

```
Input
  │
  ├─► RMSNorm ─► Attention ─► + (residual)
  │                           │
  └───────────────────────────┘
  │
  ├─► RMSNorm ─► MLP ─► + (residual)
  │                     │
  └─────────────────────┘
  │
Output
```

### Grouped-Query Attention (GQA)

- 32 query heads share 8 KV heads
- Each KV head serves 4 query heads
- Reduces KV cache memory by 4x

### Rotary Position Embeddings (RoPE)

Half-split layout: rotate dimension `i` with dimension `i + head_dim/2`.

```rust
// For each head
x'[i] = x[i] * cos - x[i + half] * sin
x'[i + half] = x[i] * sin + x[i + half] * cos
```

### SwiGLU MLP

```rust
output = down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
```

## Tokenizer

BPE tokenizer with:

- **Metaspace pre-tokenization:** Spaces → `▁` (U+2581)
- **UTF-8 character handling**
- **Byte fallback:** Unknown chars → `<0xXX>` tokens
- **BOS token:** Automatically inserted

## Memory Layout

### InferenceState

Pre-allocated buffers for zero-allocation inference:

```rust
pub struct InferenceState {
    // Hidden state: [hidden_size]
    pub hidden_state: Array1<f32>,
    pub residual: Array1<f32>,

    // Attention: [n_heads, head_dim], [n_kv_heads, head_dim]
    pub q_state: Array2<f32>,
    pub k_state: Array2<f32>,
    pub v_state: Array2<f32>,

    // KV cache: [n_layers, n_kv_heads, max_seq_len, head_dim]
    pub k_cache: Array4<f32>,
    pub v_cache: Array4<f32>,

    // Outputs: [vocab_size]
    pub logits: Array1<f32>,
}
```

### Memory Estimates (Mistral 7B)

| Component | Eager | Lazy |
|-----------|-------|------|
| Model weights | ~25GB | 0 (mmap) |
| KV cache (seq=500) | 125MB | 125MB |
| Inference buffers | ~1MB | ~1MB |
| **Total** | **~25GB** | **<2GB** |

## TensorView API

Lazy tensor access for memory-efficient loading:

```rust
use torchless::{Parameters, TensorView, TensorDtype};

let params = Parameters::load("model.bin")?;

// Get lazy view (no copy)
let view: TensorView = params.get_tensor_view("layer.0.q_proj.weight")?;

// Properties
view.dtype    // TensorDtype::F32, F16, Int8, Int4, etc.
view.shape    // [out_features, in_features]
view.nrows()  // Output dimension
view.ncols()  // Input dimension

// Lazy operations
let row: Vec<f32> = view.get_row(0);           // Single row, dequantizes
let out: Vec<f32> = view.matmul_vec(&input);   // Fused dequant + matmul
view.matmul_vec_into(&input, &mut output);     // Pre-allocated output
```

## SIMD Kernels

When built with `--features simd`, these operations use 8-wide vectorization:

| Kernel | Scalar | SIMD |
|--------|--------|------|
| RMSNorm | `rmsnorm()` | `rmsnorm_simd()` |
| Softmax | `softmax()` | `softmax_simd()` |
| SiLU | `silu()` | `silu_simd()` |
| RoPE | `apply_rope()` | `apply_rope_simd()` |

Auto-selecting wrappers (`fast_*`) choose the best available implementation.

With AVX-512 support, operations use 16-wide vectorization for ~2x speedup on supported CPUs.

## Parallel Processing

When built with `--features parallel`, these operations use Rayon:

| Operation | Serial | Parallel |
|-----------|--------|----------|
| Matrix-vector multiply | `matmul_vec()` | `matmul_vec_parallel()` |
| Attention scores | per-position loop | `compute_attention_scores_parallel()` |
| Weighted sum | per-column loop | `weighted_sum_rows_parallel()` |
| Attention heads | sequential | parallel head iteration |

## Build Optimizations

For maximum performance:

```toml
# Cargo.toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

```bash
# Target-specific optimizations
RUSTFLAGS="-C target-cpu=native" cargo build --release --features "simd,parallel,openblas"
```
