//! AVX-512 optimized kernel implementations
//!
//! These kernels use AVX-512 intrinsics for 16-wide SIMD operations on x86_64.
//! They provide significant speedup over the portable wide crate on supported CPUs.
//!
//! # Supported CPUs
//! - Intel Skylake-X and later server CPUs
//! - Intel Ice Lake and later client CPUs
//! - AMD Zen 4 and later
//!
//! # Safety
//! All functions in this module use unsafe intrinsics and require AVX-512F support.
//! Callers must verify CPU support before calling these functions.

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::*;

/// Check if AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
pub fn is_avx512_available() -> bool {
    is_x86_feature_detected!("avx512f")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn is_avx512_available() -> bool {
    false
}

/// AVX-512 optimized matrix-vector multiplication: out = W @ x
/// W: (n, d), x: (d,) -> out: (n,)
///
/// # Safety
/// Requires AVX-512F support. Caller must verify with `is_avx512_available()`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn matmul_vec_avx512(
    weights: &[f32],
    input: &[f32],
    output: &mut [f32],
    rows: usize,
    cols: usize,
) {
    debug_assert_eq!(weights.len(), rows * cols);
    debug_assert_eq!(input.len(), cols);
    debug_assert_eq!(output.len(), rows);

    let w_ptr = weights.as_ptr();
    let x_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for row in 0..rows {
        let row_offset = row * cols;
        let mut sum = _mm512_setzero_ps();

        // Process 16 elements at a time
        let mut col = 0;
        while col + 16 <= cols {
            let w = _mm512_loadu_ps(w_ptr.add(row_offset + col));
            let x = _mm512_loadu_ps(x_ptr.add(col));
            sum = _mm512_fmadd_ps(w, x, sum);
            col += 16;
        }

        // Horizontal sum of 16 elements
        let mut result = _mm512_reduce_add_ps(sum);

        // Handle remainder with scalar operations
        while col < cols {
            result += *w_ptr.add(row_offset + col) * *x_ptr.add(col);
            col += 1;
        }

        *out_ptr.add(row) = result;
    }
}

/// AVX-512 optimized RMSNorm: x = x * weight / sqrt(mean(x^2) + eps)
///
/// # Safety
/// Requires AVX-512F support. Caller must verify with `is_avx512_available()`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn rmsnorm_avx512(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();

    // Pass 1: Compute sum of squares
    let mut sum_sq = _mm512_setzero_ps();
    let x_ptr = x.as_ptr();

    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(x_ptr.add(i));
        sum_sq = _mm512_fmadd_ps(v, v, sum_sq);
        i += 16;
    }

    // Horizontal sum
    let mut total_sq = _mm512_reduce_add_ps(sum_sq);

    // Handle remainder
    while i < n {
        let v = *x_ptr.add(i);
        total_sq += v * v;
        i += 1;
    }

    // Compute inverse RMS
    let rms = (total_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_vec = _mm512_set1_ps(inv_rms);

    // Pass 2: Normalize and apply weight
    let x_ptr_mut = x.as_mut_ptr();
    let w_ptr = weight.as_ptr();

    let mut i = 0;
    while i + 16 <= n {
        let x_vec = _mm512_loadu_ps(x_ptr_mut.add(i));
        let w_vec = _mm512_loadu_ps(w_ptr.add(i));
        let result = _mm512_mul_ps(_mm512_mul_ps(x_vec, inv_rms_vec), w_vec);
        _mm512_storeu_ps(x_ptr_mut.add(i), result);
        i += 16;
    }

    // Handle remainder
    while i < n {
        *x_ptr_mut.add(i) = *x_ptr_mut.add(i) * inv_rms * *w_ptr.add(i);
        i += 1;
    }
}

/// AVX-512 optimized softmax with numerical stability
///
/// # Safety
/// Requires AVX-512F support. Caller must verify with `is_avx512_available()`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_avx512(x: &mut [f32]) {
    let n = x.len();
    let x_ptr = x.as_mut_ptr();

    // Pass 1: Find max
    let mut max_vec = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(x_ptr.add(i));
        max_vec = _mm512_max_ps(max_vec, v);
        i += 16;
    }

    let mut max_val = _mm512_reduce_max_ps(max_vec);
    while i < n {
        max_val = max_val.max(*x_ptr.add(i));
        i += 1;
    }

    let max_splat = _mm512_set1_ps(max_val);

    // Pass 2: Compute exp(x - max) and sum
    // Note: AVX-512 doesn't have native exp, so we use a polynomial approximation
    // For production, consider using SVML or computing exp in scalar
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(x_ptr.add(i));
        let shifted = _mm512_sub_ps(v, max_splat);

        // Store shifted, compute exp in scalar (AVX-512 lacks native exp)
        let mut temp = [0.0f32; 16];
        _mm512_storeu_ps(temp.as_mut_ptr(), shifted);
        for j in 0..16 {
            temp[j] = temp[j].exp();
            sum += temp[j];
        }
        let exp_vec = _mm512_loadu_ps(temp.as_ptr());
        _mm512_storeu_ps(x_ptr.add(i), exp_vec);

        i += 16;
    }

    // Handle remainder
    while i < n {
        let exp_val = (*x_ptr.add(i) - max_val).exp();
        *x_ptr.add(i) = exp_val;
        sum += exp_val;
        i += 1;
    }

    // Pass 3: Divide by sum
    let inv_sum = 1.0 / sum;
    let inv_sum_vec = _mm512_set1_ps(inv_sum);

    let mut i = 0;
    while i + 16 <= n {
        let v = _mm512_loadu_ps(x_ptr.add(i));
        let result = _mm512_mul_ps(v, inv_sum_vec);
        _mm512_storeu_ps(x_ptr.add(i), result);
        i += 16;
    }

    // Handle remainder
    while i < n {
        *x_ptr.add(i) *= inv_sum;
        i += 1;
    }
}

/// AVX-512 optimized SiLU activation: x / (1 + exp(-x))
///
/// # Safety
/// Requires AVX-512F support. Caller must verify with `is_avx512_available()`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn silu_avx512(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let n = input.len();

    let x_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let one = _mm512_set1_ps(1.0);

    let mut i = 0;
    while i + 16 <= n {
        let x = _mm512_loadu_ps(x_ptr.add(i));

        // SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        // Compute in scalar due to lack of native exp
        let mut temp = [0.0f32; 16];
        _mm512_storeu_ps(temp.as_mut_ptr(), x);
        for j in 0..16 {
            temp[j] = temp[j] / (1.0 + (-temp[j]).exp());
        }
        let result = _mm512_loadu_ps(temp.as_ptr());
        _mm512_storeu_ps(out_ptr.add(i), result);

        i += 16;
    }

    // Handle remainder
    while i < n {
        let val = *x_ptr.add(i);
        *out_ptr.add(i) = val / (1.0 + (-val).exp());
        i += 1;
    }
}

/// AVX-512 optimized RoPE application
/// x: [n_heads * head_dim] (flattened), cos/sin: [head_dim/2]
///
/// # Safety
/// Requires AVX-512F support. Caller must verify with `is_avx512_available()`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn apply_rope_avx512(
    x: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    n_heads: usize,
    head_dim: usize,
) {
    debug_assert_eq!(x.len(), n_heads * head_dim);
    debug_assert_eq!(cos.len(), head_dim / 2);
    debug_assert_eq!(sin.len(), head_dim / 2);

    let half = head_dim / 2;
    let x_ptr = x.as_mut_ptr();
    let cos_ptr = cos.as_ptr();
    let sin_ptr = sin.as_ptr();

    for h in 0..n_heads {
        let head_offset = h * head_dim;

        let mut i = 0;
        while i + 16 <= half {
            // Load first half and second half elements
            let xi = _mm512_loadu_ps(x_ptr.add(head_offset + i));
            let yi = _mm512_loadu_ps(x_ptr.add(head_offset + half + i));
            let c = _mm512_loadu_ps(cos_ptr.add(i));
            let s = _mm512_loadu_ps(sin_ptr.add(i));

            // Rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
            let new_x = _mm512_sub_ps(_mm512_mul_ps(xi, c), _mm512_mul_ps(yi, s));
            let new_y = _mm512_add_ps(_mm512_mul_ps(xi, s), _mm512_mul_ps(yi, c));

            _mm512_storeu_ps(x_ptr.add(head_offset + i), new_x);
            _mm512_storeu_ps(x_ptr.add(head_offset + half + i), new_y);

            i += 16;
        }

        // Handle remainder
        while i < half {
            let xi = *x_ptr.add(head_offset + i);
            let yi = *x_ptr.add(head_offset + half + i);
            let c = *cos_ptr.add(i);
            let s = *sin_ptr.add(i);

            *x_ptr.add(head_offset + i) = xi * c - yi * s;
            *x_ptr.add(head_offset + half + i) = xi * s + yi * c;

            i += 1;
        }
    }
}

/// AVX-512 optimized dot product
///
/// # Safety
/// Requires AVX-512F support. Caller must verify with `is_avx512_available()`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum = _mm512_setzero_ps();
    let mut i = 0;

    while i + 16 <= n {
        let va = _mm512_loadu_ps(a_ptr.add(i));
        let vb = _mm512_loadu_ps(b_ptr.add(i));
        sum = _mm512_fmadd_ps(va, vb, sum);
        i += 16;
    }

    let mut result = _mm512_reduce_add_ps(sum);

    // Handle remainder
    while i < n {
        result += *a_ptr.add(i) * *b_ptr.add(i);
        i += 1;
    }

    result
}

/// AVX-512 optimized attention scores computation
/// Computes scores[i] = keys[i, :].dot(query) * scale
///
/// # Safety
/// Requires AVX-512F support. Caller must verify with `is_avx512_available()`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_attention_scores_avx512(
    query: &[f32],
    keys: &[f32],
    scores: &mut [f32],
    n_keys: usize,
    key_dim: usize,
    scale: f32,
) {
    debug_assert_eq!(query.len(), key_dim);
    debug_assert_eq!(keys.len(), n_keys * key_dim);
    debug_assert_eq!(scores.len(), n_keys);

    let q_ptr = query.as_ptr();
    let k_ptr = keys.as_ptr();
    let s_ptr = scores.as_mut_ptr();
    let scale_vec = _mm512_set1_ps(scale);

    for i in 0..n_keys {
        let key_offset = i * key_dim;
        let mut sum = _mm512_setzero_ps();

        let mut j = 0;
        while j + 16 <= key_dim {
            let q = _mm512_loadu_ps(q_ptr.add(j));
            let k = _mm512_loadu_ps(k_ptr.add(key_offset + j));
            sum = _mm512_fmadd_ps(q, k, sum);
            j += 16;
        }

        let mut result = _mm512_reduce_add_ps(sum);

        // Handle remainder
        while j < key_dim {
            result += *q_ptr.add(j) * *k_ptr.add(key_offset + j);
            j += 1;
        }

        *s_ptr.add(i) = result * scale;
    }
}

/// AVX-512 optimized weighted sum of rows
/// out[j] = sum_i(weights[i] * matrix[i, j])
///
/// # Safety
/// Requires AVX-512F support. Caller must verify with `is_avx512_available()`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f")]
pub unsafe fn weighted_sum_rows_avx512(
    weights: &[f32],
    matrix: &[f32],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert_eq!(weights.len(), n_rows);
    debug_assert_eq!(matrix.len(), n_rows * n_cols);
    debug_assert_eq!(output.len(), n_cols);

    let w_ptr = weights.as_ptr();
    let m_ptr = matrix.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // Zero output
    let mut j = 0;
    while j + 16 <= n_cols {
        _mm512_storeu_ps(out_ptr.add(j), _mm512_setzero_ps());
        j += 16;
    }
    while j < n_cols {
        *out_ptr.add(j) = 0.0;
        j += 1;
    }

    // Accumulate weighted rows
    for i in 0..n_rows {
        let weight = *w_ptr.add(i);
        let weight_vec = _mm512_set1_ps(weight);
        let row_offset = i * n_cols;

        let mut j = 0;
        while j + 16 <= n_cols {
            let row = _mm512_loadu_ps(m_ptr.add(row_offset + j));
            let out = _mm512_loadu_ps(out_ptr.add(j));
            let result = _mm512_fmadd_ps(weight_vec, row, out);
            _mm512_storeu_ps(out_ptr.add(j), result);
            j += 16;
        }

        // Handle remainder
        while j < n_cols {
            *out_ptr.add(j) += weight * *m_ptr.add(row_offset + j);
            j += 1;
        }
    }
}

// Stub implementations for non-AVX512 targets
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub unsafe fn matmul_vec_avx512(
    _weights: &[f32],
    _input: &[f32],
    _output: &mut [f32],
    _rows: usize,
    _cols: usize,
) {
    panic!("AVX-512 not available - check is_avx512_available() before calling");
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub unsafe fn rmsnorm_avx512(_x: &mut [f32], _weight: &[f32], _eps: f32) {
    panic!("AVX-512 not available - check is_avx512_available() before calling");
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub unsafe fn softmax_avx512(_x: &mut [f32]) {
    panic!("AVX-512 not available - check is_avx512_available() before calling");
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub unsafe fn silu_avx512(_input: &[f32], _output: &mut [f32]) {
    panic!("AVX-512 not available - check is_avx512_available() before calling");
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub unsafe fn apply_rope_avx512(
    _x: &mut [f32],
    _cos: &[f32],
    _sin: &[f32],
    _n_heads: usize,
    _head_dim: usize,
) {
    panic!("AVX-512 not available - check is_avx512_available() before calling");
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub unsafe fn dot_product_avx512(_a: &[f32], _b: &[f32]) -> f32 {
    panic!("AVX-512 not available - check is_avx512_available() before calling");
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub unsafe fn compute_attention_scores_avx512(
    _query: &[f32],
    _keys: &[f32],
    _scores: &mut [f32],
    _n_keys: usize,
    _key_dim: usize,
    _scale: f32,
) {
    panic!("AVX-512 not available - check is_avx512_available() before calling");
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub unsafe fn weighted_sum_rows_avx512(
    _weights: &[f32],
    _matrix: &[f32],
    _output: &mut [f32],
    _n_rows: usize,
    _n_cols: usize,
) {
    panic!("AVX-512 not available - check is_avx512_available() before calling");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx512_availability() {
        // Just verify the detection function works
        let _available = is_avx512_available();
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    mod avx512_tests {
        use super::*;

        #[test]
        fn test_dot_product_avx512() {
            if !is_avx512_available() {
                return;
            }

            let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..32).map(|i| (i * 2) as f32).collect();

            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let result = unsafe { dot_product_avx512(&a, &b) };

            assert!((result - expected).abs() < 1e-3);
        }

        #[test]
        fn test_matmul_vec_avx512() {
            if !is_avx512_available() {
                return;
            }

            let rows = 4;
            let cols = 32;
            let weights: Vec<f32> = (0..rows * cols).map(|i| (i % 10) as f32 * 0.1).collect();
            let input: Vec<f32> = (0..cols).map(|i| i as f32 * 0.1).collect();
            let mut output = vec![0.0f32; rows];

            // Expected result
            let mut expected = vec![0.0f32; rows];
            for r in 0..rows {
                for c in 0..cols {
                    expected[r] += weights[r * cols + c] * input[c];
                }
            }

            unsafe {
                matmul_vec_avx512(&weights, &input, &mut output, rows, cols);
            }

            for i in 0..rows {
                assert!(
                    (output[i] - expected[i]).abs() < 1e-3,
                    "Row {} mismatch: {} vs {}",
                    i,
                    output[i],
                    expected[i]
                );
            }
        }

        #[test]
        fn test_rmsnorm_avx512() {
            if !is_avx512_available() {
                return;
            }

            let mut x: Vec<f32> = (1..33).map(|i| i as f32).collect();
            let weight = vec![1.0f32; 32];
            let eps = 1e-5;

            let mut x_scalar = x.clone();
            // Scalar implementation
            let squares: f32 = x_scalar.iter().map(|v| v * v).sum();
            let rms = (squares / x_scalar.len() as f32 + eps).sqrt();
            for (i, w) in weight.iter().enumerate() {
                x_scalar[i] = x_scalar[i] / rms * w;
            }

            unsafe {
                rmsnorm_avx512(&mut x, &weight, eps);
            }

            for i in 0..x.len() {
                assert!(
                    (x[i] - x_scalar[i]).abs() < 1e-4,
                    "Index {} mismatch: {} vs {}",
                    i,
                    x[i],
                    x_scalar[i]
                );
            }
        }

        #[test]
        fn test_softmax_avx512() {
            if !is_avx512_available() {
                return;
            }

            let mut x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
            let mut x_scalar = x.clone();

            // Scalar softmax
            let max = x_scalar.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in x_scalar.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in x_scalar.iter_mut() {
                *v /= sum;
            }

            unsafe {
                softmax_avx512(&mut x);
            }

            // Check sum is 1.0
            let sum: f32 = x.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);

            for i in 0..x.len() {
                assert!(
                    (x[i] - x_scalar[i]).abs() < 1e-5,
                    "Index {} mismatch: {} vs {}",
                    i,
                    x[i],
                    x_scalar[i]
                );
            }
        }

        #[test]
        fn test_silu_avx512() {
            if !is_avx512_available() {
                return;
            }

            let input: Vec<f32> = (-16..16).map(|i| i as f32 * 0.25).collect();
            let mut output = vec![0.0f32; input.len()];

            let expected: Vec<f32> = input.iter().map(|&v| v / (1.0 + (-v).exp())).collect();

            unsafe {
                silu_avx512(&input, &mut output);
            }

            for i in 0..output.len() {
                assert!(
                    (output[i] - expected[i]).abs() < 1e-5,
                    "Index {} mismatch: {} vs {}",
                    i,
                    output[i],
                    expected[i]
                );
            }
        }
    }
}
