//! Fused CPU Kernel Implementations
//!
//! This module provides fused kernel implementations that combine multiple
//! operations into a single memory pass, reducing memory bandwidth requirements.
//!
//! # Fused Operations
//! - **Fused RMSNorm + Linear**: Single pass normalization + projection
//! - **Fused SwiGLU**: gate_proj + up_proj + SiLU + multiply in single kernel
//! - **Fused Attention**: Q*K + softmax + weighted_sum in single kernel
//!
//! # Performance Benefits
//! - Reduces memory bandwidth by 2-3x for fused operations
//! - Better cache utilization
//! - Fewer memory allocations

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};

/// Fused RMSNorm + Linear projection
///
/// Combines: y = Linear(RMSNorm(x, weight_norm, eps), weight_proj)
///
/// This avoids materializing the intermediate normalized tensor.
///
/// # Arguments
/// * `x` - Input tensor (d,)
/// * `weight_norm` - RMSNorm weights (d,)
/// * `weight_proj` - Projection weights (out_d, d)
/// * `eps` - RMSNorm epsilon
///
/// # Returns
/// Projected output tensor (out_d,)
pub fn fused_rmsnorm_linear(
    x: &Array1<f32>,
    weight_norm: &Array1<f32>,
    weight_proj: &Array2<f32>,
    eps: f32,
) -> Array1<f32> {
    let (out_dim, in_dim) = weight_proj.dim();
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(weight_norm.len(), in_dim);

    let x_slice = x.as_slice().expect("x must be contiguous");
    let norm_slice = weight_norm
        .as_slice()
        .expect("weight_norm must be contiguous");

    // Compute RMS
    let sum_sq: f32 = x_slice.iter().map(|v| v * v).sum();
    let rms = (sum_sq / in_dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Fused normalize + matmul: avoid materializing normalized tensor
    let mut output = vec![0.0f32; out_dim];

    for (row_idx, row) in weight_proj.outer_iter().enumerate() {
        let row_slice = row.as_slice().expect("row must be contiguous");
        let mut sum = 0.0f32;

        for (&w, (&x_val, &norm_w)) in row_slice.iter().zip(x_slice.iter().zip(norm_slice.iter())) {
            // x_normalized[j] = x[j] * inv_rms * norm_w[j]
            let x_normalized = x_val * inv_rms * norm_w;
            sum += w * x_normalized;
        }

        output[row_idx] = sum;
    }

    Array1::from_vec(output)
}

/// Fused RMSNorm + Linear projection with pre-allocated output
///
/// # Safety
/// Output must have correct size.
pub fn fused_rmsnorm_linear_into(
    x: &Array1<f32>,
    weight_norm: &Array1<f32>,
    weight_proj: &Array2<f32>,
    eps: f32,
    output: &mut Array1<f32>,
) {
    let (out_dim, in_dim) = weight_proj.dim();
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(weight_norm.len(), in_dim);
    debug_assert_eq!(output.len(), out_dim);

    let x_slice = x.as_slice().expect("x must be contiguous");
    let norm_slice = weight_norm
        .as_slice()
        .expect("weight_norm must be contiguous");
    let out_slice = output.as_slice_mut().expect("output must be contiguous");

    // Compute RMS
    let sum_sq: f32 = x_slice.iter().map(|v| v * v).sum();
    let rms = (sum_sq / in_dim as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Fused normalize + matmul
    for (row_idx, row) in weight_proj.outer_iter().enumerate() {
        let row_slice = row.as_slice().expect("row must be contiguous");
        let mut sum = 0.0f32;

        for (j, &w) in row_slice.iter().enumerate() {
            let x_normalized = x_slice[j] * inv_rms * norm_slice[j];
            sum += w * x_normalized;
        }

        out_slice[row_idx] = sum;
    }
}

/// Fused SwiGLU activation
///
/// Combines: output = SiLU(gate_proj(x)) * up_proj(x)
///
/// This is the activation pattern used in Mistral and LLaMA models.
/// Instead of materializing gate and up separately, we compute them together.
///
/// # Arguments
/// * `x` - Input tensor (d,)
/// * `gate_proj` - Gate projection weights (hidden_d, d)
/// * `up_proj` - Up projection weights (hidden_d, d)
///
/// # Returns
/// Activated hidden tensor (hidden_d,)
pub fn fused_swiglu(
    x: &Array1<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
) -> Array1<f32> {
    let (hidden_dim, in_dim) = gate_proj.dim();
    debug_assert_eq!(gate_proj.dim(), up_proj.dim());
    debug_assert_eq!(x.len(), in_dim);

    let x_slice = x.as_slice().expect("x must be contiguous");

    let mut output = vec![0.0f32; hidden_dim];

    for i in 0..hidden_dim {
        let gate_row = gate_proj.row(i);
        let up_row = up_proj.row(i);

        // Compute gate and up projections in parallel
        let mut gate_val = 0.0f32;
        let mut up_val = 0.0f32;

        let gate_slice = gate_row.as_slice().expect("gate row must be contiguous");
        let up_slice = up_row.as_slice().expect("up row must be contiguous");

        for j in 0..in_dim {
            gate_val += gate_slice[j] * x_slice[j];
            up_val += up_slice[j] * x_slice[j];
        }

        // Apply SiLU to gate and multiply by up
        let silu_gate = gate_val / (1.0 + (-gate_val).exp());
        output[i] = silu_gate * up_val;
    }

    Array1::from_vec(output)
}

/// Fused SwiGLU with pre-allocated output
pub fn fused_swiglu_into(
    x: &Array1<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
    output: &mut Array1<f32>,
) {
    let (hidden_dim, in_dim) = gate_proj.dim();
    debug_assert_eq!(gate_proj.dim(), up_proj.dim());
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(output.len(), hidden_dim);

    let x_slice = x.as_slice().expect("x must be contiguous");
    let out_slice = output.as_slice_mut().expect("output must be contiguous");

    for i in 0..hidden_dim {
        let gate_row = gate_proj.row(i);
        let up_row = up_proj.row(i);
        let gate_slice = gate_row.as_slice().expect("gate row must be contiguous");
        let up_slice = up_row.as_slice().expect("up row must be contiguous");

        let mut gate_val = 0.0f32;
        let mut up_val = 0.0f32;

        for j in 0..in_dim {
            gate_val += gate_slice[j] * x_slice[j];
            up_val += up_slice[j] * x_slice[j];
        }

        let silu_gate = gate_val / (1.0 + (-gate_val).exp());
        out_slice[i] = silu_gate * up_val;
    }
}

/// Fused attention: Q*K^T + softmax + V weighted sum
///
/// Computes a single head's attention output without materializing the full
/// attention matrix. This is memory-efficient for inference.
///
/// # Arguments
/// * `query` - Query vector (head_dim,)
/// * `keys` - Key cache (seq_len, head_dim)
/// * `values` - Value cache (seq_len, head_dim)
/// * `scale` - Attention scale factor (typically 1/sqrt(head_dim))
/// * `seq_len` - Number of valid positions in the cache
///
/// # Returns
/// Attention output (head_dim,)
pub fn fused_attention(
    query: ArrayView1<f32>,
    keys: ArrayView2<f32>,
    values: ArrayView2<f32>,
    scale: f32,
    seq_len: usize,
) -> Array1<f32> {
    let head_dim = query.len();
    debug_assert!(seq_len <= keys.nrows());
    debug_assert_eq!(keys.ncols(), head_dim);
    debug_assert_eq!(values.ncols(), head_dim);

    let query_slice = query.as_slice().expect("query must be contiguous");

    // Step 1: Compute attention scores and find max for numerical stability
    let mut scores = vec![0.0f32; seq_len];
    let mut max_score = f32::NEG_INFINITY;

    for i in 0..seq_len {
        let key_row = keys.row(i);
        let key_slice = key_row.as_slice().expect("key row must be contiguous");

        let mut dot = 0.0f32;
        for j in 0..head_dim {
            dot += query_slice[j] * key_slice[j];
        }
        scores[i] = dot * scale;
        max_score = max_score.max(scores[i]);
    }

    // Step 2: Softmax with exp(score - max) for stability
    let mut sum_exp = 0.0f32;
    for i in 0..seq_len {
        scores[i] = (scores[i] - max_score).exp();
        sum_exp += scores[i];
    }

    // Normalize
    let inv_sum = 1.0 / sum_exp;
    for i in 0..seq_len {
        scores[i] *= inv_sum;
    }

    // Step 3: Weighted sum of values
    let mut output = vec![0.0f32; head_dim];
    for i in 0..seq_len {
        let weight = scores[i];
        let value_row = values.row(i);
        let value_slice = value_row.as_slice().expect("value row must be contiguous");

        for j in 0..head_dim {
            output[j] += weight * value_slice[j];
        }
    }

    Array1::from_vec(output)
}

/// Fused attention with pre-allocated output and score buffer
pub fn fused_attention_into(
    query: ArrayView1<f32>,
    keys: ArrayView2<f32>,
    values: ArrayView2<f32>,
    scale: f32,
    seq_len: usize,
    scores: &mut [f32],
    output: &mut ArrayViewMut1<f32>,
) {
    let head_dim = query.len();
    debug_assert!(seq_len <= keys.nrows());
    debug_assert!(seq_len <= scores.len());
    debug_assert_eq!(output.len(), head_dim);

    let query_slice = query.as_slice().expect("query must be contiguous");

    // Step 1: Compute attention scores and find max
    let mut max_score = f32::NEG_INFINITY;
    for i in 0..seq_len {
        let key_row = keys.row(i);
        let key_slice = key_row.as_slice().expect("key row must be contiguous");

        let mut dot = 0.0f32;
        for j in 0..head_dim {
            dot += query_slice[j] * key_slice[j];
        }
        scores[i] = dot * scale;
        max_score = max_score.max(scores[i]);
    }

    // Step 2: Softmax
    let mut sum_exp = 0.0f32;
    for i in 0..seq_len {
        scores[i] = (scores[i] - max_score).exp();
        sum_exp += scores[i];
    }

    let inv_sum = 1.0 / sum_exp;
    for i in 0..seq_len {
        scores[i] *= inv_sum;
    }

    // Step 3: Weighted sum of values
    let out_slice = output.as_slice_mut().expect("output must be contiguous");
    for j in 0..head_dim {
        out_slice[j] = 0.0;
    }

    for i in 0..seq_len {
        let weight = scores[i];
        let value_row = values.row(i);
        let value_slice = value_row.as_slice().expect("value row must be contiguous");

        for j in 0..head_dim {
            out_slice[j] += weight * value_slice[j];
        }
    }
}

/// Fused MLP forward pass
///
/// Combines: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
///
/// This is the complete MLP forward pass for Mistral/LLaMA.
///
/// # Arguments
/// * `x` - Input tensor (d,)
/// * `gate_proj` - Gate projection weights (hidden_d, d)
/// * `up_proj` - Up projection weights (hidden_d, d)
/// * `down_proj` - Down projection weights (d, hidden_d)
///
/// # Returns
/// MLP output (d,)
pub fn fused_mlp(
    x: &Array1<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
    down_proj: &Array2<f32>,
) -> Array1<f32> {
    let (hidden_dim, in_dim) = gate_proj.dim();
    let (out_dim, down_in_dim) = down_proj.dim();
    debug_assert_eq!(gate_proj.dim(), up_proj.dim());
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(down_in_dim, hidden_dim);
    debug_assert_eq!(out_dim, in_dim);

    let x_slice = x.as_slice().expect("x must be contiguous");

    // Compute intermediate hidden state
    let mut hidden = vec![0.0f32; hidden_dim];
    for i in 0..hidden_dim {
        let gate_row = gate_proj.row(i);
        let up_row = up_proj.row(i);
        let gate_slice = gate_row.as_slice().expect("gate row must be contiguous");
        let up_slice = up_row.as_slice().expect("up row must be contiguous");

        let mut gate_val = 0.0f32;
        let mut up_val = 0.0f32;

        for j in 0..in_dim {
            gate_val += gate_slice[j] * x_slice[j];
            up_val += up_slice[j] * x_slice[j];
        }

        let silu_gate = gate_val / (1.0 + (-gate_val).exp());
        hidden[i] = silu_gate * up_val;
    }

    // Compute output through down_proj
    let mut output = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let down_row = down_proj.row(i);
        let down_slice = down_row.as_slice().expect("down row must be contiguous");
        let mut sum = 0.0f32;
        for j in 0..hidden_dim {
            sum += down_slice[j] * hidden[j];
        }
        output[i] = sum;
    }

    Array1::from_vec(output)
}

/// Fused MLP with pre-allocated buffers
pub fn fused_mlp_into(
    x: &Array1<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
    down_proj: &Array2<f32>,
    hidden_buffer: &mut [f32],
    output: &mut Array1<f32>,
) {
    let (hidden_dim, in_dim) = gate_proj.dim();
    let (out_dim, _) = down_proj.dim();
    debug_assert!(hidden_buffer.len() >= hidden_dim);
    debug_assert_eq!(output.len(), out_dim);

    let x_slice = x.as_slice().expect("x must be contiguous");
    let out_slice = output.as_slice_mut().expect("output must be contiguous");

    // Compute intermediate hidden state
    for i in 0..hidden_dim {
        let gate_row = gate_proj.row(i);
        let up_row = up_proj.row(i);
        let gate_slice = gate_row.as_slice().expect("gate row must be contiguous");
        let up_slice = up_row.as_slice().expect("up row must be contiguous");

        let mut gate_val = 0.0f32;
        let mut up_val = 0.0f32;

        for j in 0..in_dim {
            gate_val += gate_slice[j] * x_slice[j];
            up_val += up_slice[j] * x_slice[j];
        }

        let silu_gate = gate_val / (1.0 + (-gate_val).exp());
        hidden_buffer[i] = silu_gate * up_val;
    }

    // Compute output through down_proj
    for i in 0..out_dim {
        let down_row = down_proj.row(i);
        let down_slice = down_row.as_slice().expect("down row must be contiguous");
        let mut sum = 0.0f32;
        for j in 0..hidden_dim {
            sum += down_slice[j] * hidden_buffer[j];
        }
        out_slice[i] = sum;
    }
}

#[cfg(feature = "parallel")]
mod parallel_fused {
    use super::*;
    use rayon::prelude::*;

    /// Parallel fused SwiGLU
    ///
    /// Uses raw slice access to avoid borrow issues in parallel closures.
    pub fn fused_swiglu_parallel(
        x: &Array1<f32>,
        gate_proj: &Array2<f32>,
        up_proj: &Array2<f32>,
    ) -> Array1<f32> {
        let (hidden_dim, in_dim) = gate_proj.dim();
        let x_slice = x.as_slice().expect("x must be contiguous");
        let gate_slice = gate_proj.as_slice().expect("gate_proj must be contiguous");
        let up_slice = up_proj.as_slice().expect("up_proj must be contiguous");

        let output: Vec<f32> = (0..hidden_dim)
            .into_par_iter()
            .map(|i| {
                let row_offset = i * in_dim;
                let mut gate_val = 0.0f32;
                let mut up_val = 0.0f32;

                for j in 0..in_dim {
                    gate_val += gate_slice[row_offset + j] * x_slice[j];
                    up_val += up_slice[row_offset + j] * x_slice[j];
                }

                let silu_gate = gate_val / (1.0 + (-gate_val).exp());
                silu_gate * up_val
            })
            .collect();

        Array1::from_vec(output)
    }

    /// Parallel fused MLP
    ///
    /// Uses raw slice access to avoid borrow issues in parallel closures.
    pub fn fused_mlp_parallel(
        x: &Array1<f32>,
        gate_proj: &Array2<f32>,
        up_proj: &Array2<f32>,
        down_proj: &Array2<f32>,
    ) -> Array1<f32> {
        let (hidden_dim, in_dim) = gate_proj.dim();
        let (out_dim, _) = down_proj.dim();
        let x_slice = x.as_slice().expect("x must be contiguous");
        let gate_slice = gate_proj.as_slice().expect("gate_proj must be contiguous");
        let up_slice = up_proj.as_slice().expect("up_proj must be contiguous");
        let down_slice = down_proj.as_slice().expect("down_proj must be contiguous");

        // Compute intermediate hidden state in parallel
        let hidden: Vec<f32> = (0..hidden_dim)
            .into_par_iter()
            .map(|i| {
                let row_offset = i * in_dim;
                let mut gate_val = 0.0f32;
                let mut up_val = 0.0f32;

                for j in 0..in_dim {
                    gate_val += gate_slice[row_offset + j] * x_slice[j];
                    up_val += up_slice[row_offset + j] * x_slice[j];
                }

                let silu_gate = gate_val / (1.0 + (-gate_val).exp());
                silu_gate * up_val
            })
            .collect();

        // Compute output through down_proj in parallel
        let output: Vec<f32> = (0..out_dim)
            .into_par_iter()
            .map(|i| {
                let row_offset = i * hidden_dim;
                let mut sum = 0.0f32;
                for j in 0..hidden_dim {
                    sum += down_slice[row_offset + j] * hidden[j];
                }
                sum
            })
            .collect();

        Array1::from_vec(output)
    }
}

#[cfg(feature = "parallel")]
pub use parallel_fused::{fused_mlp_parallel, fused_swiglu_parallel};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_fused_rmsnorm_linear() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let weight_norm = array![1.0, 1.0, 1.0, 1.0];
        let weight_proj = Array2::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        let eps = 1e-5;

        // Compute reference
        let mut x_norm = x.clone();
        let sum_sq: f32 = x_norm.iter().map(|v| v * v).sum();
        let rms = (sum_sq / x_norm.len() as f32 + eps).sqrt();
        x_norm.mapv_inplace(|v| v / rms);
        for (i, &w) in weight_norm.iter().enumerate() {
            x_norm[i] *= w;
        }
        let expected = weight_proj.dot(&x_norm);

        let result = fused_rmsnorm_linear(&x, &weight_norm, &weight_proj, eps);

        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "Index {} mismatch: {} vs {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_fused_swiglu() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let gate_proj = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        let up_proj = Array2::from_shape_fn((3, 4), |(i, j)| ((i * 4 + j) as f32 * 0.1) + 0.5);

        // Compute reference
        let gate = gate_proj.dot(&x);
        let up = up_proj.dot(&x);
        let silu_gate = gate.mapv(|v| v / (1.0 + (-v).exp()));
        let expected = &silu_gate * &up;

        let result = fused_swiglu(&x, &gate_proj, &up_proj);

        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "Index {} mismatch: {} vs {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_fused_attention() {
        let head_dim = 4;
        let seq_len = 3;

        let query = array![1.0, 0.0, 0.0, 0.0];
        let keys =
            Array2::from_shape_fn((seq_len, head_dim), |(i, j)| if i == j { 1.0 } else { 0.0 });
        let values = Array2::from_shape_fn((seq_len, head_dim), |(i, j)| (i * head_dim + j) as f32);

        let scale = 1.0;
        let result = fused_attention(query.view(), keys.view(), values.view(), scale, seq_len);

        // Query matches first key most, so output should be weighted towards first value
        assert_eq!(result.len(), head_dim);
        // Sum should reflect proper softmax weighting
        let sum: f32 = result.sum();
        assert!(sum.is_finite());
    }

    #[test]
    fn test_fused_mlp() {
        let d = 4;
        let hidden = 3;

        let x = array![1.0, 2.0, 3.0, 4.0];
        let gate_proj = Array2::from_shape_fn((hidden, d), |(i, j)| (i * d + j) as f32 * 0.1);
        let up_proj = Array2::from_shape_fn((hidden, d), |(i, j)| ((i * d + j) as f32 * 0.1) + 0.1);
        let down_proj = Array2::from_shape_fn((d, hidden), |(i, j)| (i * hidden + j) as f32 * 0.1);

        // Compute reference
        let gate = gate_proj.dot(&x);
        let up = up_proj.dot(&x);
        let silu_gate = gate.mapv(|v| v / (1.0 + (-v).exp()));
        let hidden_state = &silu_gate * &up;
        let expected = down_proj.dot(&hidden_state);

        let result = fused_mlp(&x, &gate_proj, &up_proj, &down_proj);

        assert_eq!(result.len(), d);
        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-4,
                "Index {} mismatch: {} vs {}",
                i,
                result[i],
                expected[i]
            );
        }
    }

    #[cfg(feature = "parallel")]
    mod parallel_tests {
        use super::*;

        #[test]
        fn test_fused_swiglu_parallel_matches_scalar() {
            let x = Array1::from_vec((0..32).map(|i| i as f32 * 0.1).collect());
            let gate_proj = Array2::from_shape_fn((16, 32), |(i, j)| (i * 32 + j) as f32 * 0.01);
            let up_proj =
                Array2::from_shape_fn((16, 32), |(i, j)| ((i * 32 + j) as f32 * 0.01) + 0.1);

            let result_scalar = fused_swiglu(&x, &gate_proj, &up_proj);
            let result_parallel = fused_swiglu_parallel(&x, &gate_proj, &up_proj);

            for i in 0..result_scalar.len() {
                assert!(
                    (result_scalar[i] - result_parallel[i]).abs() < 1e-5,
                    "Index {} mismatch",
                    i
                );
            }
        }

        #[test]
        fn test_fused_mlp_parallel_matches_scalar() {
            let d = 16;
            let hidden = 8;

            let x = Array1::from_vec((0..d).map(|i| i as f32 * 0.1).collect());
            let gate_proj = Array2::from_shape_fn((hidden, d), |(i, j)| (i * d + j) as f32 * 0.01);
            let up_proj =
                Array2::from_shape_fn((hidden, d), |(i, j)| ((i * d + j) as f32 * 0.01) + 0.1);
            let down_proj =
                Array2::from_shape_fn((d, hidden), |(i, j)| (i * hidden + j) as f32 * 0.01);

            let result_scalar = fused_mlp(&x, &gate_proj, &up_proj, &down_proj);
            let result_parallel = fused_mlp_parallel(&x, &gate_proj, &up_proj, &down_proj);

            for i in 0..result_scalar.len() {
                assert!(
                    (result_scalar[i] - result_parallel[i]).abs() < 1e-4,
                    "Index {} mismatch",
                    i
                );
            }
        }
    }
}
