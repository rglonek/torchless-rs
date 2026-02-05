use crate::tensor::Tensor1;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

pub mod backend;

// Architecture-specific SIMD implementations
pub mod avx512;
pub mod neon;

// Fused kernel implementations
pub mod fused;

// Runtime CPU dispatch
pub mod dispatch;

// Optimized kernels with bounds check elimination and prefetching
pub mod optimized;

// GPU backends
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "rocm")]
pub mod rocm;

#[cfg(feature = "metal-gpu")]
pub mod metal;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(test)]
mod tests;

// Re-export backend types for convenient access
pub use backend::{Backend, BackendPreference, CpuBackend, KernelBackend, init_backend, default_backend};

// Re-export CUDA backend types when the feature is enabled
#[cfg(feature = "cuda")]
pub use cuda::{CudaBackend, CudaTensor, CudaMemoryPool};

// Re-export ROCm backend types when the feature is enabled
#[cfg(feature = "rocm")]
pub use rocm::{RocmBackend, RocmTensor, RocmMemoryPool};

// Re-export Metal backend types when the feature is enabled
#[cfg(feature = "metal-gpu")]
pub use metal::{MetalBackend, MetalTensor, MetalMemoryPool};

// Re-export OpenCL backend types when the feature is enabled
#[cfg(feature = "opencl")]
pub use opencl::{OpenCLBackend, OpenCLTensor, OpenCLMemoryPool};

// Re-export dispatch types for runtime CPU feature detection
pub use dispatch::{
    CpuFeatures, Dispatcher, cpu_features, global_dispatcher,
    dispatch_matmul_vec, dispatch_matmul_vec_into, dispatch_rmsnorm,
    dispatch_softmax, dispatch_silu, dispatch_apply_rope,
    dispatch_compute_attention_scores, dispatch_weighted_sum_rows,
};

// Re-export fused kernel functions
pub use fused::{
    fused_rmsnorm_linear, fused_rmsnorm_linear_into,
    fused_swiglu, fused_swiglu_into,
    fused_attention, fused_attention_into,
    fused_mlp, fused_mlp_into,
};

#[cfg(feature = "parallel")]
pub use fused::{fused_swiglu_parallel, fused_mlp_parallel};

// Re-export architecture-specific availability checks
pub use avx512::is_avx512_available;
pub use neon::is_neon_available;

// =============================================================================
// SIMD-optimized kernel implementations
// =============================================================================

#[cfg(feature = "simd")]
mod simd_kernels {
    use ndarray::{Array1, Array2, ArrayViewMut1};
    use wide::f32x8;

    /// SIMD-optimized RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    /// Uses 8-wide SIMD lanes for sum of squares and normalization.
    pub fn rmsnorm_simd(x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
        let x_slice = x.as_slice_mut().expect("x must be contiguous");
        let weight_slice = weight.as_slice().expect("weight must be contiguous");

        // Vectorized sum of squares
        let mut sum_sq = f32x8::ZERO;
        let chunks = x_slice.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let v = f32x8::from(chunk);
            sum_sq += v * v;
        }

        // Reduce the 8-lane sum to scalar
        let sum_sq_arr: [f32; 8] = sum_sq.into();
        let mut total_sq: f32 = sum_sq_arr.iter().sum();

        // Handle remainder (scalar)
        for &val in remainder {
            total_sq += val * val;
        }

        // Compute RMS
        let rms = (total_sq / x_slice.len() as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let inv_rms_vec = f32x8::splat(inv_rms);

        // Vectorized normalization and weight multiplication
        let x_chunks = x_slice.chunks_exact_mut(8);
        let x_remainder_start = x_chunks.len() * 8;
        let weight_chunks = weight_slice.chunks_exact(8);

        for (x_chunk, w_chunk) in x_chunks.zip(weight_chunks) {
            let x_vec = f32x8::from(&*x_chunk);
            let w_vec = f32x8::from(w_chunk);
            let result = x_vec * inv_rms_vec * w_vec;
            let result_arr: [f32; 8] = result.into();
            x_chunk.copy_from_slice(&result_arr);
        }

        // Handle remainder (scalar)
        for i in x_remainder_start..x_slice.len() {
            x_slice[i] = x_slice[i] * inv_rms * weight_slice[i];
        }
    }

    /// SIMD-optimized softmax with numerical stability (subtract max)
    /// Uses 8-wide SIMD lanes for max finding, exp, and division.
    pub fn softmax_simd(x: &mut Array1<f32>) {
        let x_slice = x.as_slice_mut().expect("x must be contiguous");
        softmax_slice_simd(x_slice);
    }

    /// SIMD-optimized softmax for mutable array views
    pub fn softmax_view_simd(x: &mut ArrayViewMut1<f32>) {
        // ArrayViewMut1 may not be contiguous, so we need to handle this case
        if let Some(x_slice) = x.as_slice_mut() {
            softmax_slice_simd(x_slice);
        } else {
            // Fallback to scalar implementation for non-contiguous views
            super::softmax_view(x);
        }
    }

    /// Internal SIMD softmax implementation on a slice
    fn softmax_slice_simd(x_slice: &mut [f32]) {
        // Find max using SIMD
        let mut max_vec = f32x8::splat(f32::NEG_INFINITY);
        let chunks = x_slice.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let v = f32x8::from(chunk);
            max_vec = max_vec.max(v);
        }

        // Reduce to scalar max
        let max_arr: [f32; 8] = max_vec.into();
        let mut max_val = max_arr.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Check remainder
        for &val in remainder {
            max_val = max_val.max(val);
        }

        // Compute exp(x - max) and sum using SIMD
        let max_splat = f32x8::splat(max_val);
        let mut sum_vec = f32x8::ZERO;

        // We need to compute exp and store, then sum
        // Process in place
        let chunks = x_slice.chunks_exact_mut(8);
        let remainder_start = chunks.len() * 8;

        for chunk in chunks {
            let v = f32x8::from(&*chunk);
            // Note: wide crate doesn't have exp, so we use the scalar fallback per-element
            // This is a known limitation - for production, consider using `packed_simd` or `sleef`
            let shifted: [f32; 8] = (v - max_splat).into();
            let exp_vals: [f32; 8] = shifted.map(|s| s.exp());
            chunk.copy_from_slice(&exp_vals);
            sum_vec += f32x8::from(exp_vals);
        }

        // Handle remainder (scalar)
        let sum_arr: [f32; 8] = sum_vec.into();
        let mut sum: f32 = sum_arr.iter().sum();

        for i in remainder_start..x_slice.len() {
            let exp_val = (x_slice[i] - max_val).exp();
            x_slice[i] = exp_val;
            sum += exp_val;
        }

        // Divide by sum using SIMD
        let inv_sum = 1.0 / sum;
        let inv_sum_vec = f32x8::splat(inv_sum);

        let chunks = x_slice.chunks_exact_mut(8);
        let remainder_start = chunks.len() * 8;

        for chunk in chunks {
            let v = f32x8::from(&*chunk);
            let result = v * inv_sum_vec;
            let result_arr: [f32; 8] = result.into();
            chunk.copy_from_slice(&result_arr);
        }

        // Handle remainder (scalar)
        for i in remainder_start..x_slice.len() {
            x_slice[i] *= inv_sum;
        }
    }

    /// SIMD-optimized SiLU activation: x / (1 + exp(-x))
    /// Returns a new array with the activation applied.
    pub fn silu_simd(x: &Array1<f32>) -> Array1<f32> {
        let x_slice = x.as_slice().expect("x must be contiguous");
        let mut result = vec![0.0f32; x_slice.len()];

        // Process 8 elements at a time
        let x_chunks = x_slice.chunks_exact(8);
        let remainder_start = x_chunks.len() * 8;
        let result_chunks = result.chunks_exact_mut(8);

        for (x_chunk, result_chunk) in x_chunks.zip(result_chunks) {
            let v = f32x8::from(x_chunk);
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            // Note: wide crate doesn't have exp, use scalar per-element
            let v_arr: [f32; 8] = v.into();
            let silu_vals: [f32; 8] = v_arr.map(|val| val / (1.0 + (-val).exp()));
            result_chunk.copy_from_slice(&silu_vals);
        }

        // Handle remainder (scalar)
        for i in remainder_start..x_slice.len() {
            let val = x_slice[i];
            result[i] = val / (1.0 + (-val).exp());
        }

        Array1::from_vec(result)
    }

    /// SIMD-optimized RoPE application using half-split layout
    /// x: [n_heads, head_dim], cos/sin: [head_dim/2]
    /// Mistral uses half-split: rotate dimension i with dimension i + head_dim/2
    pub fn apply_rope_simd(x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
        let (n_heads, head_dim) = x.dim();
        let half = head_dim / 2;

        let cos_slice = cos.as_slice().expect("cos must be contiguous");
        let sin_slice = sin.as_slice().expect("sin must be contiguous");

        // Process each head
        for h in 0..n_heads {
            // Get mutable slice for this head's row
            let head_row = x.row_mut(h);
            let head_slice = head_row.into_slice().expect("head row must be contiguous");

            // Split into first half and second half
            let (first_half, second_half) = head_slice.split_at_mut(half);

            // Process 8 dimensions at a time using SIMD
            let first_chunks = first_half.chunks_exact_mut(8);
            let first_remainder_start = first_chunks.len() * 8;
            let second_chunks = second_half.chunks_exact_mut(8);
            let cos_chunks = cos_slice.chunks_exact(8);
            let sin_chunks = sin_slice.chunks_exact(8);

            for (((first_chunk, second_chunk), cos_chunk), sin_chunk) in
                first_chunks.zip(second_chunks).zip(cos_chunks).zip(sin_chunks)
            {
                let xi = f32x8::from(&*first_chunk);
                let yi = f32x8::from(&*second_chunk);
                let c = f32x8::from(cos_chunk);
                let s = f32x8::from(sin_chunk);

                // Rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
                let new_x = xi * c - yi * s;
                let new_y = xi * s + yi * c;

                let new_x_arr: [f32; 8] = new_x.into();
                let new_y_arr: [f32; 8] = new_y.into();

                first_chunk.copy_from_slice(&new_x_arr);
                second_chunk.copy_from_slice(&new_y_arr);
            }

            // Handle remainder (scalar)
            for i in first_remainder_start..half {
                let xi = first_half[i];
                let yi = second_half[i];
                let c = cos_slice[i];
                let s = sin_slice[i];

                first_half[i] = xi * c - yi * s;
                second_half[i] = xi * s + yi * c;
            }
        }
    }
}

/// Matrix multiplication: out = W @ x
/// W: (n, d), x: (d,) -> out: (n,)
pub fn matmul_vec(w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
    w.dot(x)
}

/// Matrix multiplication with pre-allocated output: out = W @ x
/// W: (n, d), x: (d,) -> out: (n,)
/// Writes result directly into `out` buffer to avoid allocation.
pub fn matmul_vec_into(w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
    let result = w.dot(x);
    out.assign(&result);
}

/// Matrix multiplication: C = A @ B
pub fn matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    a.dot(b)
}

/// Row-major matrix-vector multiplication: out = x @ W
/// x: (n,), W: (n, d) -> out: (d,)
pub fn row_matmul(x: &Array1<f32>, w: &Array2<f32>) -> Array1<f32> {
    x.dot(w)
}

// =============================================================================
// Parallel kernel implementations (Rayon)
// =============================================================================

#[cfg(feature = "parallel")]
mod parallel_kernels {
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
    use rayon::prelude::*;

    /// Parallel matrix-vector multiplication: out = W @ x
    /// W: (n, d), x: (d,) -> out: (n,)
    /// Parallelizes across output dimensions (rows of W).
    pub fn matmul_vec_parallel(w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        let x_slice = x.as_slice().expect("x must be contiguous");

        let result: Vec<f32> = w
            .outer_iter()
            .into_par_iter()
            .map(|row| {
                row.as_slice()
                    .expect("row must be contiguous")
                    .iter()
                    .zip(x_slice.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect();

        Array1::from_vec(result)
    }

    /// Parallel matrix-vector multiplication with pre-allocated output: out = W @ x
    /// W: (n, d), x: (d,) -> out: (n,)
    /// Writes result directly into `out` buffer to avoid allocation.
    pub fn matmul_vec_into_parallel(w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
        let x_slice = x.as_slice().expect("x must be contiguous");
        let out_slice = out.as_slice_mut().expect("out must be contiguous");

        // Parallel iteration over rows, writing directly to output
        out_slice
            .par_iter_mut()
            .zip(w.outer_iter().into_par_iter())
            .for_each(|(out_elem, row)| {
                *out_elem = row
                    .as_slice()
                    .expect("row must be contiguous")
                    .iter()
                    .zip(x_slice.iter())
                    .map(|(a, b)| a * b)
                    .sum();
            });
    }

    /// Parallel weighted sum of rows: out[j] = sum_i(weights[i] * matrix[i, j])
    /// weights: (n,), matrix: (n, d) -> out: (d,)
    /// Parallelizes across output columns.
    pub fn weighted_sum_rows_parallel(
        weights: ArrayView1<f32>,
        matrix: ArrayView2<f32>,
        out: &mut ArrayViewMut1<f32>,
    ) {
        // Get output slice for parallel writing
        let out_slice = out.as_slice_mut().expect("out must be contiguous");

        // For each output column, compute the weighted sum in parallel
        out_slice.par_iter_mut().enumerate().for_each(|(j, out_j)| {
            let col = matrix.column(j);
            *out_j = weights
                .iter()
                .zip(col.iter())
                .map(|(&w, &v)| w * v)
                .sum();
        });
    }

    /// Parallel attention scores computation
    /// Computes scores[i] = keys[i, :].dot(query) * scale for each row in parallel
    /// query: (d,), keys: (n, d) -> scores: (n,)
    pub fn compute_attention_scores_parallel(
        query: ArrayView1<f32>,
        keys: ArrayView2<f32>,
        scores: &mut ArrayViewMut1<f32>,
        scale: f32,
    ) {
        let query_slice = query.as_slice().expect("query must be contiguous");
        let scores_slice = scores.as_slice_mut().expect("scores must be contiguous");

        scores_slice
            .par_iter_mut()
            .zip(keys.outer_iter().into_par_iter())
            .for_each(|(score, key_row)| {
                let dot: f32 = key_row
                    .as_slice()
                    .expect("key row must be contiguous")
                    .iter()
                    .zip(query_slice.iter())
                    .map(|(k, q)| k * q)
                    .sum();
                *score = dot * scale;
            });
    }
}

#[cfg(feature = "parallel")]
pub use parallel_kernels::{
    compute_attention_scores_parallel, matmul_vec_into_parallel, matmul_vec_parallel,
    weighted_sum_rows_parallel,
};

/// Compute weighted sum of rows: out[j] = sum_i(weights[i] * matrix[i, j])
/// weights: (n,), matrix: (n, d) -> out: (d,)
/// This is equivalent to weights @ matrix but writes directly into `out` buffer.
pub fn weighted_sum_rows(
    weights: ArrayView1<f32>,
    matrix: ArrayView2<f32>,
    out: &mut ArrayViewMut1<f32>,
) {
    out.fill(0.0);
    for (i, w) in weights.iter().enumerate() {
        let row = matrix.row(i);
        for (j, &v) in row.iter().enumerate() {
            out[j] += w * v;
        }
    }
}

/// Compute dot products of a query vector with each row of a key matrix
/// Writes scores[i] = keys[i, :].dot(query) directly into the scores slice.
/// query: (d,), keys: (n, d) -> scores: (n,)
pub fn compute_attention_scores(
    query: ArrayView1<f32>,
    keys: ArrayView2<f32>,
    scores: &mut ArrayViewMut1<f32>,
    scale: f32,
) {
    for i in 0..keys.nrows() {
        let dot: f32 = keys.row(i).iter().zip(query.iter()).map(|(k, q)| k * q).sum();
        scores[i] = dot * scale;
    }
}

/// Softmax with numerical stability (subtract max)
pub fn softmax(x: &mut Tensor1) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    x.mapv_inplace(|v| (v - max).exp());
    let sum: f32 = x.sum();
    x.mapv_inplace(|v| v / sum);
}

/// Softmax with numerical stability for mutable array views
/// This variant accepts mutable slice views (ArrayViewMut1) to avoid allocation.
pub fn softmax_view(x: &mut ArrayViewMut1<f32>) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    x.mapv_inplace(|v| (v - max).exp());
    let sum: f32 = x.sum();
    x.mapv_inplace(|v| v / sum);
}

/// SiLU activation: x / (1 + exp(-x))
pub fn silu(x: &Array1<f32>) -> Array1<f32> {
    x.mapv(|v| v / (1.0 + (-v).exp()))
}

/// RMSNorm: x * weight / sqrt(mean(x^2) + eps)
pub fn rmsnorm(x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
    let squares: f32 = x.iter().map(|v| v * v).sum();
    let rms = (squares / x.len() as f32 + eps).sqrt();

    x.mapv_inplace(|v| v / rms);

    // Element-wise multiply with weight
    for (i, w) in weight.iter().enumerate() {
        x[i] *= w;
    }
}

/// Initialize RoPE inverse frequencies
/// inv_freq[i] = 1 / (theta^(i / head_dim))
pub fn init_rope_freqs(head_dim: usize, rope_theta: f32) -> Array1<f32> {
    let half_dim = head_dim / 2;
    Array1::from_vec(
        (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf(i as f32 / half_dim as f32))
            .collect(),
    )
}

/// Generate cos/sin embeddings for a given position
pub fn rope_embeddings(inv_freq: &Array1<f32>, pos: usize) -> (Array1<f32>, Array1<f32>) {
    let pos_f = pos as f32;
    let cos = inv_freq.mapv(|freq| (freq * pos_f).cos());
    let sin = inv_freq.mapv(|freq| (freq * pos_f).sin());
    (cos, sin)
}

/// Apply RoPE to query or key tensor using half-split layout
/// x: [n_heads, head_dim], cos/sin: [head_dim/2]
/// Mistral uses half-split: rotate dimension i with dimension i + head_dim/2
pub fn apply_rope(x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
    let (n_heads, head_dim) = x.dim();
    let half = head_dim / 2;

    for h in 0..n_heads {
        for i in 0..half {
            let xi = x[[h, i]];
            let yi = x[[h, i + half]];
            let c = cos[i];
            let s = sin[i];

            // Rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
            x[[h, i]] = xi * c - yi * s;
            x[[h, i + half]] = xi * s + yi * c;
        }
    }
}

// =============================================================================
// Feature-gated SIMD re-exports
// When the `simd` feature is enabled, these provide optimized versions of the
// core kernels. Call the `*_simd` variants directly for explicit SIMD usage,
// or use the `fast_*` functions which auto-select the best implementation.
// =============================================================================

#[cfg(feature = "simd")]
pub use simd_kernels::{apply_rope_simd, rmsnorm_simd, silu_simd, softmax_simd, softmax_view_simd};

/// Auto-selecting RMSNorm: uses SIMD when available, falls back to scalar
#[cfg(feature = "simd")]
pub fn fast_rmsnorm(x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
    simd_kernels::rmsnorm_simd(x, weight, eps)
}

#[cfg(not(feature = "simd"))]
pub fn fast_rmsnorm(x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
    rmsnorm(x, weight, eps)
}

/// Auto-selecting softmax: uses SIMD when available, falls back to scalar
#[cfg(feature = "simd")]
pub fn fast_softmax(x: &mut Tensor1) {
    simd_kernels::softmax_simd(x)
}

#[cfg(not(feature = "simd"))]
pub fn fast_softmax(x: &mut Tensor1) {
    softmax(x)
}

/// Auto-selecting softmax view: uses SIMD when available, falls back to scalar
#[cfg(feature = "simd")]
pub fn fast_softmax_view(x: &mut ArrayViewMut1<f32>) {
    simd_kernels::softmax_view_simd(x)
}

#[cfg(not(feature = "simd"))]
pub fn fast_softmax_view(x: &mut ArrayViewMut1<f32>) {
    softmax_view(x)
}

/// Auto-selecting SiLU: uses SIMD when available, falls back to scalar
#[cfg(feature = "simd")]
pub fn fast_silu(x: &Array1<f32>) -> Array1<f32> {
    simd_kernels::silu_simd(x)
}

#[cfg(not(feature = "simd"))]
pub fn fast_silu(x: &Array1<f32>) -> Array1<f32> {
    silu(x)
}

/// Auto-selecting RoPE: uses SIMD when available, falls back to scalar
#[cfg(feature = "simd")]
pub fn fast_apply_rope(x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
    simd_kernels::apply_rope_simd(x, cos, sin)
}

#[cfg(not(feature = "simd"))]
pub fn fast_apply_rope(x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
    apply_rope(x, cos, sin)
}

// =============================================================================
// Feature-gated Parallel re-exports
// When the `parallel` feature is enabled, these provide parallelized versions
// of the core kernels. Use `fast_*` functions for auto-selection.
// =============================================================================

/// Auto-selecting matmul_vec: uses parallel when available, falls back to serial
#[cfg(feature = "parallel")]
pub fn fast_matmul_vec(w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
    parallel_kernels::matmul_vec_parallel(w, x)
}

#[cfg(not(feature = "parallel"))]
pub fn fast_matmul_vec(w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
    matmul_vec(w, x)
}

/// Auto-selecting matmul_vec_into: uses parallel when available, falls back to serial
#[cfg(feature = "parallel")]
pub fn fast_matmul_vec_into(w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
    parallel_kernels::matmul_vec_into_parallel(w, x, out)
}

#[cfg(not(feature = "parallel"))]
pub fn fast_matmul_vec_into(w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
    matmul_vec_into(w, x, out)
}

/// Auto-selecting weighted_sum_rows: uses parallel when available, falls back to serial
#[cfg(feature = "parallel")]
pub fn fast_weighted_sum_rows(
    weights: ArrayView1<f32>,
    matrix: ArrayView2<f32>,
    out: &mut ArrayViewMut1<f32>,
) {
    parallel_kernels::weighted_sum_rows_parallel(weights, matrix, out)
}

#[cfg(not(feature = "parallel"))]
pub fn fast_weighted_sum_rows(
    weights: ArrayView1<f32>,
    matrix: ArrayView2<f32>,
    out: &mut ArrayViewMut1<f32>,
) {
    weighted_sum_rows(weights, matrix, out)
}

/// Auto-selecting attention scores: uses parallel when available, falls back to serial
#[cfg(feature = "parallel")]
pub fn fast_compute_attention_scores(
    query: ArrayView1<f32>,
    keys: ArrayView2<f32>,
    scores: &mut ArrayViewMut1<f32>,
    scale: f32,
) {
    parallel_kernels::compute_attention_scores_parallel(query, keys, scores, scale)
}

#[cfg(not(feature = "parallel"))]
pub fn fast_compute_attention_scores(
    query: ArrayView1<f32>,
    keys: ArrayView2<f32>,
    scores: &mut ArrayViewMut1<f32>,
    scale: f32,
) {
    compute_attention_scores(query, keys, scores, scale)
}

// =============================================================================
// Runtime-Dispatched Kernel Selection
// =============================================================================
// 
// The dispatch module provides runtime CPU feature detection and automatic
// selection of the optimal kernel implementation. Use the `dispatch_*` functions
// for automatic selection, or create a `Dispatcher` for more control.
//
// Feature Priority (highest to lowest):
// 1. AVX-512F (x86_64) - 16-wide SIMD
// 2. NEON (aarch64) - 4-wide SIMD with 16-element loop unrolling
// 3. wide crate (portable) - 8-wide SIMD when compiled with `simd` feature
// 4. Rayon parallel - multi-threaded when compiled with `parallel` feature
// 5. Scalar - fallback implementation
//
// The `fast_*` functions use compile-time feature selection (simd/parallel).
// The `dispatch_*` functions use runtime CPU detection for best performance.
//
// Example:
// ```ignore
// use torchless::kernels::{cpu_features, dispatch_matmul_vec};
// 
// // Check what features are available
// let features = cpu_features();
// println!("CPU features: {}", features.describe());
// 
// // Use runtime dispatch for optimal performance
// let result = dispatch_matmul_vec(&weights, &input);
// ```

/// Print CPU feature information to stdout.
/// Useful for debugging and performance tuning.
pub fn print_cpu_features() {
    let features = cpu_features();
    println!("=== CPU Feature Detection ===");
    println!("  AVX-512F: {}", features.avx512f);
    println!("  AVX2: {}", features.avx2);
    println!("  FMA: {}", features.fma);
    println!("  NEON: {}", features.neon);
    println!("  Best SIMD width: {} floats", features.best_simd_width());
    println!("  Summary: {}", features.describe());
    println!();
    println!("=== Compile-time Features ===");
    #[cfg(feature = "simd")]
    println!("  simd: enabled (wide crate)");
    #[cfg(not(feature = "simd"))]
    println!("  simd: disabled");
    #[cfg(feature = "parallel")]
    println!("  parallel: enabled (rayon)");
    #[cfg(not(feature = "parallel"))]
    println!("  parallel: disabled");
}

// =============================================================================
// Optimized Kernel Re-exports (Phase 3 Memory Optimizations)
// =============================================================================
//
// These functions use bounds check elimination, memory prefetching, and
// cache-aligned operations for maximum performance.
//
// Use these in performance-critical code paths after verifying input sizes.

pub use optimized::{
    // Optimized matrix operations
    matmul_vec_optimized,
    weighted_sum_rows_optimized,
    // Optimized normalization
    rmsnorm_optimized,
    softmax_optimized,
    softmax_view_optimized,
    // Optimized activations
    silu_optimized,
    silu_optimized_into,
    // Optimized attention
    compute_attention_scores_optimized,
    fused_attention_optimized,
    // Optimized fused operations
    fused_swiglu_optimized,
    // Optimized RoPE
    apply_rope_optimized,
};
