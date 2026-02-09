//! ARM NEON optimized kernel implementations
//!
//! These kernels use NEON intrinsics for 4-wide SIMD operations on ARM64.
//! They are particularly optimized for Apple Silicon (M1/M2/M3) cache hierarchies.
//!
//! # Supported CPUs
//! - Apple M1, M2, M3 series
//! - ARM Cortex-A series with NEON
//! - AWS Graviton processors
//!
//! # Safety
//! All functions in this module use unsafe intrinsics and require NEON support.
//! On aarch64, NEON is always available, so no runtime check is needed.
#![allow(clippy::missing_safety_doc)]

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Check if NEON is available at runtime.
/// On aarch64, NEON is always available.
#[cfg(target_arch = "aarch64")]
pub fn is_neon_available() -> bool {
    true
}

#[cfg(not(target_arch = "aarch64"))]
pub fn is_neon_available() -> bool {
    false
}

/// NEON optimized matrix-vector multiplication: out = W @ x
/// W: (n, d), x: (d,) -> out: (n,)
///
/// # Safety
/// Requires NEON support. On aarch64, NEON is always available.
#[cfg(target_arch = "aarch64")]
pub unsafe fn matmul_vec_neon(
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
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        // Process 16 elements at a time (4 NEON registers)
        let mut col = 0;
        while col + 16 <= cols {
            let w0 = vld1q_f32(w_ptr.add(row_offset + col));
            let w1 = vld1q_f32(w_ptr.add(row_offset + col + 4));
            let w2 = vld1q_f32(w_ptr.add(row_offset + col + 8));
            let w3 = vld1q_f32(w_ptr.add(row_offset + col + 12));

            let x0 = vld1q_f32(x_ptr.add(col));
            let x1 = vld1q_f32(x_ptr.add(col + 4));
            let x2 = vld1q_f32(x_ptr.add(col + 8));
            let x3 = vld1q_f32(x_ptr.add(col + 12));

            sum0 = vfmaq_f32(sum0, w0, x0);
            sum1 = vfmaq_f32(sum1, w1, x1);
            sum2 = vfmaq_f32(sum2, w2, x2);
            sum3 = vfmaq_f32(sum3, w3, x3);

            col += 16;
        }

        // Process 4 elements at a time
        while col + 4 <= cols {
            let w = vld1q_f32(w_ptr.add(row_offset + col));
            let x = vld1q_f32(x_ptr.add(col));
            sum0 = vfmaq_f32(sum0, w, x);
            col += 4;
        }

        // Combine sums
        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);

        // Horizontal sum
        let mut result = vaddvq_f32(sum0);

        // Handle remainder
        while col < cols {
            result += *w_ptr.add(row_offset + col) * *x_ptr.add(col);
            col += 1;
        }

        *out_ptr.add(row) = result;
    }
}

/// NEON optimized RMSNorm: x = x * weight / sqrt(mean(x^2) + eps)
///
/// # Safety
/// Requires NEON support. On aarch64, NEON is always available.
#[cfg(target_arch = "aarch64")]
pub unsafe fn rmsnorm_neon(x: &mut [f32], weight: &[f32], eps: f32) {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();

    let x_ptr = x.as_ptr();

    // Pass 1: Compute sum of squares
    let mut sum_sq0 = vdupq_n_f32(0.0);
    let mut sum_sq1 = vdupq_n_f32(0.0);
    let mut sum_sq2 = vdupq_n_f32(0.0);
    let mut sum_sq3 = vdupq_n_f32(0.0);

    let mut i = 0;
    while i + 16 <= n {
        let v0 = vld1q_f32(x_ptr.add(i));
        let v1 = vld1q_f32(x_ptr.add(i + 4));
        let v2 = vld1q_f32(x_ptr.add(i + 8));
        let v3 = vld1q_f32(x_ptr.add(i + 12));

        sum_sq0 = vfmaq_f32(sum_sq0, v0, v0);
        sum_sq1 = vfmaq_f32(sum_sq1, v1, v1);
        sum_sq2 = vfmaq_f32(sum_sq2, v2, v2);
        sum_sq3 = vfmaq_f32(sum_sq3, v3, v3);

        i += 16;
    }

    while i + 4 <= n {
        let v = vld1q_f32(x_ptr.add(i));
        sum_sq0 = vfmaq_f32(sum_sq0, v, v);
        i += 4;
    }

    // Combine and horizontal sum
    sum_sq0 = vaddq_f32(sum_sq0, sum_sq1);
    sum_sq2 = vaddq_f32(sum_sq2, sum_sq3);
    sum_sq0 = vaddq_f32(sum_sq0, sum_sq2);
    let mut total_sq = vaddvq_f32(sum_sq0);

    // Handle remainder
    while i < n {
        let v = *x_ptr.add(i);
        total_sq += v * v;
        i += 1;
    }

    // Compute inverse RMS
    let rms = (total_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let inv_rms_vec = vdupq_n_f32(inv_rms);

    // Pass 2: Normalize and apply weight
    let x_ptr_mut = x.as_mut_ptr();
    let w_ptr = weight.as_ptr();

    let mut i = 0;
    while i + 16 <= n {
        let x0 = vld1q_f32(x_ptr_mut.add(i));
        let x1 = vld1q_f32(x_ptr_mut.add(i + 4));
        let x2 = vld1q_f32(x_ptr_mut.add(i + 8));
        let x3 = vld1q_f32(x_ptr_mut.add(i + 12));

        let w0 = vld1q_f32(w_ptr.add(i));
        let w1 = vld1q_f32(w_ptr.add(i + 4));
        let w2 = vld1q_f32(w_ptr.add(i + 8));
        let w3 = vld1q_f32(w_ptr.add(i + 12));

        let r0 = vmulq_f32(vmulq_f32(x0, inv_rms_vec), w0);
        let r1 = vmulq_f32(vmulq_f32(x1, inv_rms_vec), w1);
        let r2 = vmulq_f32(vmulq_f32(x2, inv_rms_vec), w2);
        let r3 = vmulq_f32(vmulq_f32(x3, inv_rms_vec), w3);

        vst1q_f32(x_ptr_mut.add(i), r0);
        vst1q_f32(x_ptr_mut.add(i + 4), r1);
        vst1q_f32(x_ptr_mut.add(i + 8), r2);
        vst1q_f32(x_ptr_mut.add(i + 12), r3);

        i += 16;
    }

    while i + 4 <= n {
        let x_vec = vld1q_f32(x_ptr_mut.add(i));
        let w_vec = vld1q_f32(w_ptr.add(i));
        let result = vmulq_f32(vmulq_f32(x_vec, inv_rms_vec), w_vec);
        vst1q_f32(x_ptr_mut.add(i), result);
        i += 4;
    }

    // Handle remainder
    while i < n {
        *x_ptr_mut.add(i) = *x_ptr_mut.add(i) * inv_rms * *w_ptr.add(i);
        i += 1;
    }
}

/// NEON optimized softmax with numerical stability
///
/// # Safety
/// Requires NEON support. On aarch64, NEON is always available.
#[cfg(target_arch = "aarch64")]
pub unsafe fn softmax_neon(x: &mut [f32]) {
    let n = x.len();
    let x_ptr = x.as_mut_ptr();

    // Pass 1: Find max
    let mut max_vec = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 4 <= n {
        let v = vld1q_f32(x_ptr.add(i));
        max_vec = vmaxq_f32(max_vec, v);
        i += 4;
    }

    let mut max_val = vmaxvq_f32(max_vec);
    while i < n {
        max_val = max_val.max(*x_ptr.add(i));
        i += 1;
    }

    // Pass 2: Compute exp(x - max) and sum
    let max_splat = vdupq_n_f32(max_val);
    let mut sum = 0.0f32;

    let mut i = 0;
    while i + 4 <= n {
        let v = vld1q_f32(x_ptr.add(i));
        let shifted = vsubq_f32(v, max_splat);

        // NEON doesn't have native exp, compute in scalar
        let mut temp = [0.0f32; 4];
        vst1q_f32(temp.as_mut_ptr(), shifted);
        for j in 0..4 {
            temp[j] = temp[j].exp();
            sum += temp[j];
        }
        vst1q_f32(x_ptr.add(i), vld1q_f32(temp.as_ptr()));

        i += 4;
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
    let inv_sum_vec = vdupq_n_f32(inv_sum);

    let mut i = 0;
    while i + 4 <= n {
        let v = vld1q_f32(x_ptr.add(i));
        let result = vmulq_f32(v, inv_sum_vec);
        vst1q_f32(x_ptr.add(i), result);
        i += 4;
    }

    // Handle remainder
    while i < n {
        *x_ptr.add(i) *= inv_sum;
        i += 1;
    }
}

/// NEON optimized SiLU activation: x / (1 + exp(-x))
///
/// # Safety
/// Requires NEON support. On aarch64, NEON is always available.
#[cfg(target_arch = "aarch64")]
pub unsafe fn silu_neon(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let n = input.len();

    let x_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let mut i = 0;
    while i + 4 <= n {
        let x = vld1q_f32(x_ptr.add(i));

        // SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        // Compute exp in scalar due to lack of native exp
        let mut temp = [0.0f32; 4];
        vst1q_f32(temp.as_mut_ptr(), x);
        for j in 0..4 {
            temp[j] = temp[j] / (1.0 + (-temp[j]).exp());
        }
        vst1q_f32(out_ptr.add(i), vld1q_f32(temp.as_ptr()));

        i += 4;
    }

    // Handle remainder
    while i < n {
        let val = *x_ptr.add(i);
        *out_ptr.add(i) = val / (1.0 + (-val).exp());
        i += 1;
    }
}

/// NEON optimized RoPE application
/// x: [n_heads * head_dim] (flattened), cos/sin: [head_dim/2]
///
/// # Safety
/// Requires NEON support. On aarch64, NEON is always available.
#[cfg(target_arch = "aarch64")]
pub unsafe fn apply_rope_neon(
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
        while i + 4 <= half {
            // Load first half and second half elements
            let xi = vld1q_f32(x_ptr.add(head_offset + i));
            let yi = vld1q_f32(x_ptr.add(head_offset + half + i));
            let c = vld1q_f32(cos_ptr.add(i));
            let s = vld1q_f32(sin_ptr.add(i));

            // Rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
            let new_x = vsubq_f32(vmulq_f32(xi, c), vmulq_f32(yi, s));
            let new_y = vaddq_f32(vmulq_f32(xi, s), vmulq_f32(yi, c));

            vst1q_f32(x_ptr.add(head_offset + i), new_x);
            vst1q_f32(x_ptr.add(head_offset + half + i), new_y);

            i += 4;
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

/// NEON optimized dot product
///
/// # Safety
/// Requires NEON support. On aarch64, NEON is always available.
#[cfg(target_arch = "aarch64")]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);

    let mut i = 0;
    while i + 16 <= n {
        let a0 = vld1q_f32(a_ptr.add(i));
        let a1 = vld1q_f32(a_ptr.add(i + 4));
        let a2 = vld1q_f32(a_ptr.add(i + 8));
        let a3 = vld1q_f32(a_ptr.add(i + 12));

        let b0 = vld1q_f32(b_ptr.add(i));
        let b1 = vld1q_f32(b_ptr.add(i + 4));
        let b2 = vld1q_f32(b_ptr.add(i + 8));
        let b3 = vld1q_f32(b_ptr.add(i + 12));

        sum0 = vfmaq_f32(sum0, a0, b0);
        sum1 = vfmaq_f32(sum1, a1, b1);
        sum2 = vfmaq_f32(sum2, a2, b2);
        sum3 = vfmaq_f32(sum3, a3, b3);

        i += 16;
    }

    while i + 4 <= n {
        let a_vec = vld1q_f32(a_ptr.add(i));
        let b_vec = vld1q_f32(b_ptr.add(i));
        sum0 = vfmaq_f32(sum0, a_vec, b_vec);
        i += 4;
    }

    // Combine and horizontal sum
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    sum0 = vaddq_f32(sum0, sum2);
    let mut result = vaddvq_f32(sum0);

    // Handle remainder
    while i < n {
        result += *a_ptr.add(i) * *b_ptr.add(i);
        i += 1;
    }

    result
}

/// NEON optimized attention scores computation
/// Computes scores[i] = keys[i, :].dot(query) * scale
///
/// # Safety
/// Requires NEON support. On aarch64, NEON is always available.
#[cfg(target_arch = "aarch64")]
pub unsafe fn compute_attention_scores_neon(
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

    for i in 0..n_keys {
        let key_offset = i * key_dim;
        let mut sum0 = vdupq_n_f32(0.0);
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        let mut sum3 = vdupq_n_f32(0.0);

        let mut j = 0;
        while j + 16 <= key_dim {
            let q0 = vld1q_f32(q_ptr.add(j));
            let q1 = vld1q_f32(q_ptr.add(j + 4));
            let q2 = vld1q_f32(q_ptr.add(j + 8));
            let q3 = vld1q_f32(q_ptr.add(j + 12));

            let k0 = vld1q_f32(k_ptr.add(key_offset + j));
            let k1 = vld1q_f32(k_ptr.add(key_offset + j + 4));
            let k2 = vld1q_f32(k_ptr.add(key_offset + j + 8));
            let k3 = vld1q_f32(k_ptr.add(key_offset + j + 12));

            sum0 = vfmaq_f32(sum0, q0, k0);
            sum1 = vfmaq_f32(sum1, q1, k1);
            sum2 = vfmaq_f32(sum2, q2, k2);
            sum3 = vfmaq_f32(sum3, q3, k3);

            j += 16;
        }

        while j + 4 <= key_dim {
            let q = vld1q_f32(q_ptr.add(j));
            let k = vld1q_f32(k_ptr.add(key_offset + j));
            sum0 = vfmaq_f32(sum0, q, k);
            j += 4;
        }

        // Combine and horizontal sum
        sum0 = vaddq_f32(sum0, sum1);
        sum2 = vaddq_f32(sum2, sum3);
        sum0 = vaddq_f32(sum0, sum2);
        let mut result = vaddvq_f32(sum0);

        // Handle remainder
        while j < key_dim {
            result += *q_ptr.add(j) * *k_ptr.add(key_offset + j);
            j += 1;
        }

        *s_ptr.add(i) = result * scale;
    }
}

/// NEON optimized weighted sum of rows
/// out[j] = sum_i(weights[i] * matrix[i, j])
///
/// # Safety
/// Requires NEON support. On aarch64, NEON is always available.
#[cfg(target_arch = "aarch64")]
pub unsafe fn weighted_sum_rows_neon(
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
    let zero = vdupq_n_f32(0.0);
    let mut j = 0;
    while j + 4 <= n_cols {
        vst1q_f32(out_ptr.add(j), zero);
        j += 4;
    }
    while j < n_cols {
        *out_ptr.add(j) = 0.0;
        j += 1;
    }

    // Accumulate weighted rows
    for i in 0..n_rows {
        let weight = *w_ptr.add(i);
        let weight_vec = vdupq_n_f32(weight);
        let row_offset = i * n_cols;

        let mut j = 0;
        while j + 16 <= n_cols {
            let row0 = vld1q_f32(m_ptr.add(row_offset + j));
            let row1 = vld1q_f32(m_ptr.add(row_offset + j + 4));
            let row2 = vld1q_f32(m_ptr.add(row_offset + j + 8));
            let row3 = vld1q_f32(m_ptr.add(row_offset + j + 12));

            let out0 = vld1q_f32(out_ptr.add(j));
            let out1 = vld1q_f32(out_ptr.add(j + 4));
            let out2 = vld1q_f32(out_ptr.add(j + 8));
            let out3 = vld1q_f32(out_ptr.add(j + 12));

            let r0 = vfmaq_f32(out0, weight_vec, row0);
            let r1 = vfmaq_f32(out1, weight_vec, row1);
            let r2 = vfmaq_f32(out2, weight_vec, row2);
            let r3 = vfmaq_f32(out3, weight_vec, row3);

            vst1q_f32(out_ptr.add(j), r0);
            vst1q_f32(out_ptr.add(j + 4), r1);
            vst1q_f32(out_ptr.add(j + 8), r2);
            vst1q_f32(out_ptr.add(j + 12), r3);

            j += 16;
        }

        while j + 4 <= n_cols {
            let row = vld1q_f32(m_ptr.add(row_offset + j));
            let out = vld1q_f32(out_ptr.add(j));
            let result = vfmaq_f32(out, weight_vec, row);
            vst1q_f32(out_ptr.add(j), result);
            j += 4;
        }

        // Handle remainder
        while j < n_cols {
            *out_ptr.add(j) += weight * *m_ptr.add(row_offset + j);
            j += 1;
        }
    }
}

// Stub implementations for non-aarch64 targets
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn matmul_vec_neon(
    _weights: &[f32],
    _input: &[f32],
    _output: &mut [f32],
    _rows: usize,
    _cols: usize,
) {
    panic!("NEON not available - check is_neon_available() before calling");
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn rmsnorm_neon(_x: &mut [f32], _weight: &[f32], _eps: f32) {
    panic!("NEON not available - check is_neon_available() before calling");
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn softmax_neon(_x: &mut [f32]) {
    panic!("NEON not available - check is_neon_available() before calling");
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn silu_neon(_input: &[f32], _output: &mut [f32]) {
    panic!("NEON not available - check is_neon_available() before calling");
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn apply_rope_neon(
    _x: &mut [f32],
    _cos: &[f32],
    _sin: &[f32],
    _n_heads: usize,
    _head_dim: usize,
) {
    panic!("NEON not available - check is_neon_available() before calling");
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn dot_product_neon(_a: &[f32], _b: &[f32]) -> f32 {
    panic!("NEON not available - check is_neon_available() before calling");
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn compute_attention_scores_neon(
    _query: &[f32],
    _keys: &[f32],
    _scores: &mut [f32],
    _n_keys: usize,
    _key_dim: usize,
    _scale: f32,
) {
    panic!("NEON not available - check is_neon_available() before calling");
}

#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn weighted_sum_rows_neon(
    _weights: &[f32],
    _matrix: &[f32],
    _output: &mut [f32],
    _n_rows: usize,
    _n_cols: usize,
) {
    panic!("NEON not available - check is_neon_available() before calling");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_availability() {
        let available = is_neon_available();
        #[cfg(target_arch = "aarch64")]
        assert!(available, "NEON should always be available on aarch64");
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!available, "NEON should not be available on non-aarch64");
    }

    #[cfg(target_arch = "aarch64")]
    mod neon_tests {
        use super::*;

        #[test]
        fn test_dot_product_neon() {
            let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..32).map(|i| (i * 2) as f32).collect();

            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let result = unsafe { dot_product_neon(&a, &b) };

            assert!((result - expected).abs() < 1e-3);
        }

        #[test]
        fn test_matmul_vec_neon() {
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
                matmul_vec_neon(&weights, &input, &mut output, rows, cols);
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
        fn test_rmsnorm_neon() {
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
                rmsnorm_neon(&mut x, &weight, eps);
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
        fn test_softmax_neon() {
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
                softmax_neon(&mut x);
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
        fn test_silu_neon() {
            let input: Vec<f32> = (-16..16).map(|i| i as f32 * 0.25).collect();
            let mut output = vec![0.0f32; input.len()];

            let expected: Vec<f32> = input.iter().map(|&v| v / (1.0 + (-v).exp())).collect();

            unsafe {
                silu_neon(&input, &mut output);
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

        #[test]
        fn test_apply_rope_neon() {
            let n_heads = 4;
            let head_dim = 16;
            let half = head_dim / 2;

            let mut x: Vec<f32> = (0..n_heads * head_dim).map(|i| i as f32).collect();
            let cos: Vec<f32> = (0..half).map(|i| (i as f32 * 0.1).cos()).collect();
            let sin: Vec<f32> = (0..half).map(|i| (i as f32 * 0.1).sin()).collect();

            // Scalar reference
            let mut x_scalar = x.clone();
            for h in 0..n_heads {
                for i in 0..half {
                    let idx_first = h * head_dim + i;
                    let idx_second = h * head_dim + half + i;
                    let xi = x_scalar[idx_first];
                    let yi = x_scalar[idx_second];
                    x_scalar[idx_first] = xi * cos[i] - yi * sin[i];
                    x_scalar[idx_second] = xi * sin[i] + yi * cos[i];
                }
            }

            unsafe {
                apply_rope_neon(&mut x, &cos, &sin, n_heads, head_dim);
            }

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
    }
}
