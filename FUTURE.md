# Future Optimizations Roadmap

This document outlines potential optimizations to match or exceed C/C++ inference performance while providing flexible memory/speed tradeoffs.

## Priority Legend

- 🔴 **High Impact** - Significant performance or usability improvement
- 🟡 **Medium Impact** - Moderate improvement, good ROI
- 🟢 **Low Impact** - Nice to have, incremental improvement

---

## 1. Quantization Options (Memory/Speed Tradeoffs)

| Quantization | Memory (7B) | Speed | Quality | Status |
|--------------|-------------|-------|---------|--------|
| FP32 | ~28GB | Baseline | Best | ✅ Done |
| FP16 | ~14GB | Faster | Excellent | ❌ Not implemented |
| INT8 | ~7GB | Fast | Very Good | ✅ Done |
| **INT4** | ~4GB | Fast | Good | ❌ Not implemented |
| INT4 + FP16 (mixed) | ~5GB | Fast | Very Good | ❌ Not implemented |

### 🔴 INT4 Quantization

Add 4-bit quantization for the "sweet spot" between memory and quality.

**Formats to implement:**

- [ ] **Q4_0** - Simple 4-bit with FP16 scale per block (32 weights)
- [ ] **Q4_K_M** - K-quants with super-blocks, better quality
- [ ] **Q4_K_S** - Smaller variant for memory-constrained systems

**Implementation approach:**
```rust
// Block structure for Q4_0
struct Q4Block {
    scale: f16,           // 2 bytes
    weights: [u8; 16],    // 16 bytes = 32 x 4-bit weights
}

// Dequantization: weight = (nibble - 8) * scale
fn dequantize_q4(block: &Q4Block, index: usize) -> f32 {
    let nibble = if index % 2 == 0 {
        block.weights[index / 2] & 0x0F
    } else {
        block.weights[index / 2] >> 4
    };
    (nibble as f32 - 8.0) * block.scale.to_f32()
}
```

**Estimated effort:** 2-3 weeks

### 🟡 FP16 Support

Add half-precision floating point for 2x memory reduction with minimal quality loss.

- [ ] Use `half` crate for FP16 type
- [ ] Add FP16 tensor storage and operations
- [ ] SIMD FP16 kernels (AVX-512 FP16, ARM FP16)

**Estimated effort:** 1-2 weeks

### 🟢 Mixed Precision

Combine INT4 weights with FP16 activations for better quality.

- [ ] INT4 weights, FP16 KV cache
- [ ] FP16 attention scores, INT4 MLP weights

---

## 2. SIMD Optimizations

### 🔴 AVX-512 Support

Current SIMD uses `wide` crate (AVX2 max). AVX-512 can double throughput.

- [ ] Add `avx512` feature flag
- [ ] Implement AVX-512 variants of all kernels
- [ ] Runtime CPU detection for automatic selection

**Target kernels:**
```rust
#[cfg(target_feature = "avx512f")]
pub fn matmul_vec_avx512(weights: &[f32], input: &[f32], output: &mut [f32]) {
    // Process 16 floats per iteration instead of 8
    use std::arch::x86_64::*;
    // ...
}
```

**Estimated speedup:** 1.5-2x on supported CPUs

### 🟡 ARM NEON Optimization

Improve ARM performance beyond what `wide` provides.

- [ ] Hand-tuned NEON intrinsics for Apple Silicon
- [ ] Use `std::arch::aarch64` for direct control
- [ ] Optimize for M1/M2/M3 cache hierarchies

### 🟡 Architecture-Specific Dispatch

Runtime detection to use best available instruction set.

```rust
pub fn fast_matmul_vec(weights: &[f32], input: &[f32]) -> Vec<f32> {
    if is_x86_feature_detected!("avx512f") {
        matmul_vec_avx512(weights, input)
    } else if is_x86_feature_detected!("avx2") {
        matmul_vec_avx2(weights, input)
    } else if is_x86_feature_detected!("avx") {
        matmul_vec_avx(weights, input)
    } else {
        matmul_vec_scalar(weights, input)
    }
}
```

---

## 3. Memory Optimizations

### 🔴 Custom Allocator

Replace default allocator with arena/bump allocator for inference buffers.

- [ ] Use `bumpalo` for per-token allocations
- [ ] Pre-allocate all inference buffers at model load
- [ ] Zero-copy tensor views where possible

**Benefits:**
- Eliminate allocation overhead during inference
- Better cache locality
- Reduced memory fragmentation

### 🟡 Cache-Aligned Memory Layout

Align all weight tensors to cache line boundaries (64 bytes).

```rust
#[repr(C, align(64))]
struct AlignedWeights {
    data: Vec<f32>,
}
```

### 🟡 Memory Prefetching

Add explicit prefetch hints for sequential access patterns.

```rust
use std::arch::x86_64::_mm_prefetch;

unsafe {
    _mm_prefetch(weights.as_ptr().add(64) as *const i8, _MM_HINT_T0);
}
```

---

## 4. Algorithmic Optimizations

### 🔴 Flash Attention

Implement memory-efficient attention that reduces memory bandwidth.

**Benefits:**
- O(N) memory instead of O(N²) for attention scores
- Better GPU utilization (future GPU support)
- Faster for long sequences

**Implementation:**
- [ ] Tiled softmax computation
- [ ] Online softmax normalization
- [ ] Fused attention kernel

**Reference:** [FlashAttention paper](https://arxiv.org/abs/2205.14135)

### 🔴 Fused Kernels

Combine multiple operations into single passes to reduce memory bandwidth.

**Candidates:**
- [ ] Fused RMSNorm + Linear
- [ ] Fused Linear + SiLU + Linear (SwiGLU)
- [ ] Fused Attention (Q·K + softmax + ·V)
- [ ] Fused dequantize + matmul (partially done for INT8)

### 🟡 Speculative Decoding

Use smaller draft model to propose tokens, verify with main model.

- [ ] Support for draft model loading
- [ ] Batch verification of proposed tokens
- [ ] Tree-based speculation

**Estimated speedup:** 2-3x for autoregressive generation

### 🟡 Continuous Batching

Process multiple sequences with different lengths efficiently.

- [ ] Sequence-level batching
- [ ] Dynamic batch size based on memory
- [ ] Preemption for long sequences

---

## 5. Parallelization Improvements

### 🔴 Better Work Distribution

Current parallel implementation may have load imbalance.

- [ ] Profile and optimize Rayon chunk sizes
- [ ] Implement work-stealing for uneven workloads
- [ ] NUMA-aware thread placement

### 🟡 Pipeline Parallelism

Overlap computation of different layers.

```
Time ->
Layer 0: [Token 1] [Token 2] [Token 3]
Layer 1:          [Token 1] [Token 2] [Token 3]
Layer 2:                    [Token 1] [Token 2] [Token 3]
```

### 🟢 Tensor Parallelism

Split large matrices across threads for single-token latency.

- [ ] Column-parallel linear layers
- [ ] Row-parallel linear layers
- [ ] All-reduce for combining results

---

## 6. GPU Support

### 🔴 CUDA Backend

Add NVIDIA GPU support for massive speedup.

**Options:**
1. **cuBLAS** - Easy, good for matmul
2. **Custom CUDA kernels** - Best performance
3. **Rust CUDA bindings** (`cuda-rs`, `cudarc`)

**Implementation path:**
- [ ] Abstract backend trait (`CpuBackend`, `CudaBackend`)
- [ ] cuBLAS integration for matmul
- [ ] Custom attention kernel
- [ ] Memory management (pinned memory, async transfers)

### 🟡 Metal Backend (Apple Silicon)

Native GPU acceleration for Mac.

- [ ] Use `metal-rs` crate
- [ ] Metal Performance Shaders for matmul
- [ ] Custom compute shaders for attention

### 🟢 WebGPU Backend

Cross-platform GPU via `wgpu` crate.

- [ ] WGSL compute shaders
- [ ] Works on all platforms
- [ ] Good for browser deployment (future)

---

## 7. Unsafe Optimizations

### 🟡 Bounds Check Elimination

Remove bounds checks in hot loops.

```rust
// Current (safe)
for i in 0..n {
    output[i] = input[i] * weight[i];
}

// Optimized (unsafe)
unsafe {
    let out_ptr = output.as_mut_ptr();
    let in_ptr = input.as_ptr();
    let w_ptr = weight.as_ptr();
    for i in 0..n {
        *out_ptr.add(i) = *in_ptr.add(i) * *w_ptr.add(i);
    }
}
```

### 🟡 Raw Pointer SIMD

Direct SIMD intrinsics without `wide` crate abstraction.

```rust
use std::arch::x86_64::*;

unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    for i in (0..a.len()).step_by(8) {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    // Horizontal sum...
}
```

---

## 8. Build Optimizations

### 🟡 Profile-Guided Optimization (PGO)

Build with runtime profile data for better code generation.

```bash
# Step 1: Build with profiling
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release

# Step 2: Run representative workload
./target/release/torchless model.bin "test prompt" --max-tokens 100

# Step 3: Build with profile data
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release
```

**Estimated speedup:** 5-15%

### 🟡 Link-Time Optimization (LTO)

Enable cross-crate optimization.

```toml
# Cargo.toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

**Estimated speedup:** 5-10%

### 🟢 Target-Specific Builds

Optimize for specific CPU.

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

---

## 9. Model Format Improvements

### 🟡 GGUF Compatibility

Support llama.cpp's GGUF format for broader model ecosystem.

- [ ] GGUF parser
- [ ] Multiple quantization format support
- [ ] Metadata handling

**Benefits:**
- Access to pre-quantized models
- Community model compatibility
- No need for separate export step

### 🟢 Safetensors Support

Load models directly from HuggingFace format.

- [ ] Parse safetensors format
- [ ] On-the-fly quantization option
- [ ] Streaming model loading

---

## 10. Additional Model Support

### 🟡 LLaMA Architecture

Extend beyond Mistral to support LLaMA models.

- [ ] LLaMA 2 (7B, 13B, 70B)
- [ ] LLaMA 3 / 3.1
- [ ] Code Llama

### 🟢 Other Architectures

- [ ] Phi-3
- [ ] Gemma
- [ ] Qwen

---

## Implementation Priority

### Phase 1: Quick Wins (Match llama.cpp for simple use cases)
1. INT4 quantization (Q4_0)
2. LTO + PGO build optimizations
3. Bounds check elimination in hot paths
4. Better work distribution for parallel

### Phase 2: Performance Parity
1. AVX-512 SIMD kernels
2. Fused kernels (RMSNorm+Linear, SwiGLU)
3. Flash Attention
4. Custom allocator

### Phase 3: Exceed C/C++ (with hardware acceleration)
1. CUDA backend
2. Metal backend
3. Speculative decoding
4. Continuous batching

### Phase 4: Ecosystem
1. GGUF format support
2. Additional model architectures
3. Safetensors support

---

## Benchmarking Plan

To validate optimizations, implement comprehensive benchmarks:

```bash
# Token generation throughput (tokens/second)
cargo bench --bench generation_bench

# Time-to-first-token (TTFT) latency
cargo bench --bench latency_bench

# Memory usage profiling
cargo bench --bench memory_bench

# Compare against llama.cpp
./scripts/benchmark_comparison.sh
```

**Metrics to track:**
- Tokens per second (various batch sizes)
- Time to first token
- Peak memory usage
- Memory bandwidth utilization
- CPU utilization

---

## Contributing

If you're interested in implementing any of these optimizations:

1. Open an issue to discuss the approach
2. Start with a benchmark showing current performance
3. Implement the optimization
4. Add tests to verify correctness
5. Submit PR with before/after benchmarks

See existing SIMD and parallel implementations as reference.
