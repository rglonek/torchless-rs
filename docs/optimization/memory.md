# Memory Optimization

Memory-level optimizations including custom allocators, cache alignment, prefetching, and bounds check elimination.

## Overview

Memory optimization is critical for LLM inference performance because:
- Modern CPUs are often memory-bound, not compute-bound
- Allocation overhead accumulates across thousands of token generations
- Cache efficiency directly impacts throughput
- Bounds checks add branch misprediction overhead in tight loops

## Custom Memory Allocator

### Arena Allocator for Inference

```rust
use torchless::memory::InferenceArena;

// Create arena sized for typical forward pass
let arena = InferenceArena::for_inference(
    4096,   // hidden_size
    14336,  // intermediate_size  
    32,     // n_heads
    500,    // max_seq_len
);

// Fast O(1) allocations during forward pass
let hidden = arena.alloc_aligned::<f32>(4096);
let scores = arena.alloc_slice::<f32>(512);

// Reset between tokens - instant, no per-allocation overhead
arena.reset();
```

**Benefits**:
- O(1) allocation (just bump a pointer)
- Zero deallocation overhead
- Better cache locality (contiguous allocations)
- 10-50x faster than standard allocation in forward passes

### Arena-Based Inference State

```rust
use torchless::ArenaInferenceState;

let mut state = ArenaInferenceState::new(config);

for token in tokens {
    model.forward(state.as_inference_state(), token, false);
    state.reset_arena(); // Instant memory reclamation
}
```

## Cache-Aligned Buffers

### AlignedBuffer

```rust
use torchless::memory::{AlignedBuffer, SIMD_ALIGNMENT};

// 64-byte aligned buffer for AVX-512 operations
let buffer: AlignedBuffer<f32> = AlignedBuffer::zeros(4096);
assert!(buffer.is_aligned());  // 64-byte alignment guaranteed

// Use as slice
let slice = buffer.as_slice();
```

**Constants**:
- `CACHE_LINE_SIZE = 64` - Modern CPU cache line size
- `SIMD_ALIGNMENT = 64` - AVX-512 alignment requirement  
- `CACHE_LINE_F32S = 16` - f32s per cache line

## Memory Prefetching

### Prefetch Functions

```rust
use torchless::memory::{prefetch_read, prefetch_write, prefetch_sequential};

unsafe {
    // Prefetch for reading into L1 cache
    prefetch_read(data_ptr);
    
    // Prefetch for writing
    prefetch_write(output_ptr);
    
    // Prefetch multiple cache lines ahead (for sequential access)
    prefetch_sequential(ptr, 4); // 4 cache lines ahead
}
```

**Platform Support**:
- x86_64: Uses `_mm_prefetch` intrinsic
- ARM/AArch64: Uses `_prefetch` intrinsic
- Other: No-op fallback

## Bounds Check Elimination

### Unchecked Operations Module

```rust
use torchless::memory::unchecked;

// All functions require unsafe and verified bounds
unsafe {
    // Single element access
    let val = unchecked::load(slice, index);
    unchecked::store(slice, index, value);
    
    // Dot product (4-way unrolled for ILP)
    let dot = unchecked::dot_product_unrolled(&a, &b, len);
    
    // Matrix-vector multiply with prefetching
    unchecked::matmul_vec_prefetch(&weights, &input, &mut output, rows, cols);
    
    // Element-wise operations
    unchecked::mul_acc(&a, &b, &mut out, len);  // out += a * b
    unchecked::add_inplace(&mut a, &b, len);    // a += b
    unchecked::scale_inplace(&mut x, scale, len); // x *= scale
}
```

### Optimized Kernel Building Blocks

```rust
use torchless::memory::{
    sum_squares_unchecked,
    rmsnorm_unchecked,
    max_unchecked,
    softmax_unchecked,
    silu_unchecked,
};

unsafe {
    // These use 4-way unrolling and pointer arithmetic
    let ss = sum_squares_unchecked(&x, len);
    let max = max_unchecked(&x, len);
    rmsnorm_unchecked(&mut x, &weight, eps, len);
    softmax_unchecked(&mut x, len);
    silu_unchecked(&x, &mut out, len);
}
```

## Optimized Kernels

High-level optimized kernel functions that use the memory optimizations:

```rust
use torchless::kernels::{
    matmul_vec_optimized,
    rmsnorm_optimized,
    softmax_optimized,
    silu_optimized,
    compute_attention_scores_optimized,
    fused_attention_optimized,
    fused_swiglu_optimized,
    apply_rope_optimized,
};

// Matrix-vector multiply with prefetching (4 rows ahead)
matmul_vec_optimized(&weights, &input, &mut output);

// RMSNorm with bounds check elimination
rmsnorm_optimized(&mut x, &weight, eps);

// Softmax with unchecked access
softmax_optimized(&mut x);

// SiLU activation
let result = silu_optimized(&x);
```

## Performance Characteristics

### Allocation Overhead Reduction

| Operation | Standard | Arena | Speedup |
|-----------|----------|-------|---------|
| Single alloc | ~50ns | ~5ns | 10x |
| 100 allocs | ~5μs | ~0.5μs | 10x |
| Reset/free | O(n) | O(1) | 50x+ |

### Bounds Check Elimination

| Kernel | With Checks | Unchecked | Improvement |
|--------|-------------|-----------|-------------|
| Dot product | 1.00x | 0.95x | 5% |
| Matmul | 1.00x | 0.92x | 8% |
| RMSNorm | 1.00x | 0.93x | 7% |

### Prefetching Impact

| Access Pattern | Without | With Prefetch | Improvement |
|----------------|---------|---------------|-------------|
| Sequential matmul | 1.00x | 0.90x | 10% |
| Attention scores | 1.00x | 0.88x | 12% |

## Usage Guidelines

### When to Use Arena Allocator

**Do use** for:
- Token generation loops (many small allocations)
- Temporary buffers in forward pass
- Batch processing multiple sequences

**Don't use** for:
- Long-lived allocations (model weights)
- KV cache (needs to persist across tokens)
- Single large allocations

### When to Use Unchecked Operations

**Do use** when:
- Bounds are verified externally (e.g., by model architecture)
- In tight inner loops (matmul, attention)
- Profiling shows bounds checks as hotspot

**Don't use** when:
- Bounds are user-controlled
- Code is not performance-critical
- During development/debugging (use safe versions first)

### Debug Assertions

All unchecked functions include `debug_assert!` for bounds checking:
- Enabled in debug builds (catches bugs during development)
- Disabled in release builds (maximum performance)

```rust
// This will panic in debug mode if index >= len
unsafe {
    unchecked::load(slice, index)
}
```
