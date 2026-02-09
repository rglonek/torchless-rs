use super::*;
use ndarray::{array, Array2};

#[test]
fn test_matmul_vec() {
    let w = array![[1.0, 2.0], [3.0, 4.0]];
    let x = array![1.0, 2.0];
    let result = matmul_vec(&w, &x);
    assert_eq!(result, array![5.0, 11.0]);
}

#[test]
fn test_softmax() {
    let mut x = array![1.0, 2.0, 3.0];
    softmax(&mut x);

    // Verify sum is 1.0
    let sum: f32 = x.sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // Verify monotonicity (larger inputs -> larger outputs)
    assert!(x[0] < x[1]);
    assert!(x[1] < x[2]);
}

#[test]
fn test_silu() {
    let x = array![0.0, 1.0, -1.0];
    let result = silu(&x);

    // SiLU(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);

    // SiLU(x) > 0 for x > 0
    assert!(result[1] > 0.0);

    // SiLU(x) < 0 for x < 0
    assert!(result[2] < 0.0);
}

#[test]
fn test_rmsnorm() {
    let mut x = array![1.0, 2.0, 3.0];
    let weight = array![1.0, 1.0, 1.0];
    rmsnorm(&mut x, &weight, 1e-5);

    // After normalization, RMS should be approximately 1
    let squares: f32 = x.iter().map(|v| v * v).sum();
    let rms = (squares / x.len() as f32).sqrt();
    assert!((rms - 1.0).abs() < 1e-3);
}

#[test]
fn test_init_rope_freqs() {
    let head_dim = 128;
    let rope_theta = 10000.0;
    let freqs = init_rope_freqs(head_dim, rope_theta);

    // Should have half_dim frequencies
    assert_eq!(freqs.len(), head_dim / 2);

    // First frequency should be 1.0
    assert!((freqs[0] - 1.0).abs() < 1e-6);

    // Frequencies should decrease
    for i in 1..freqs.len() {
        assert!(freqs[i] < freqs[i - 1]);
    }
}

#[test]
fn test_rope_embeddings() {
    let head_dim = 8;
    let inv_freq = init_rope_freqs(head_dim, 10000.0);
    let (cos, sin) = rope_embeddings(&inv_freq, 0);

    // At position 0, cos should be all 1.0, sin should be all 0.0
    for &c in cos.iter() {
        assert!((c - 1.0).abs() < 1e-6);
    }
    for &s in sin.iter() {
        assert!((s - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_apply_rope() {
    let n_heads = 2;
    let head_dim = 4;
    let mut x = Array2::from_shape_fn((n_heads, head_dim), |(i, j)| (i * head_dim + j) as f32);

    let inv_freq = init_rope_freqs(head_dim, 10000.0);
    let (cos, sin) = rope_embeddings(&inv_freq, 1);

    let x_original = x.clone();
    apply_rope(&mut x, &cos, &sin);

    // RoPE should modify the tensor (not leave it unchanged)
    assert_ne!(x, x_original);

    // Shape should remain the same
    assert_eq!(x.shape(), &[n_heads, head_dim]);
}

#[test]
fn test_row_matmul() {
    let x = array![1.0, 2.0, 3.0];
    let w = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let result = row_matmul(&x, &w);

    // x @ w = [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6] = [22, 28]
    assert_eq!(result, array![22.0, 28.0]);
}

// =============================================================================
// SIMD Kernel Tests
// These tests verify that SIMD implementations produce identical results
// to the scalar implementations and handle edge cases correctly.
// =============================================================================

#[cfg(feature = "simd")]
mod simd_tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    /// Helper to compare arrays with tolerance
    fn arrays_approx_equal(a: &Array1<f32>, b: &Array1<f32>, eps: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
    }

    // -------------------------------------------------------------------------
    // RMSNorm SIMD Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rmsnorm_simd_matches_scalar() {
        // Test with array size that's a multiple of 8 (SIMD lane width)
        let mut x_scalar = Array1::from_vec((0..16).map(|i| (i + 1) as f32).collect());
        let mut x_simd = x_scalar.clone();
        let weight = Array1::ones(16);
        let eps = 1e-5;

        rmsnorm(&mut x_scalar, &weight, eps);
        rmsnorm_simd(&mut x_simd, &weight, eps);

        assert!(
            arrays_approx_equal(&x_scalar, &x_simd, 1e-5),
            "SIMD and scalar rmsnorm results differ"
        );
    }

    #[test]
    fn test_rmsnorm_simd_with_remainder() {
        // Test with array size that's NOT a multiple of 8 (has remainder)
        let mut x_scalar = Array1::from_vec((0..13).map(|i| (i + 1) as f32).collect());
        let mut x_simd = x_scalar.clone();
        let weight = Array1::ones(13);
        let eps = 1e-5;

        rmsnorm(&mut x_scalar, &weight, eps);
        rmsnorm_simd(&mut x_simd, &weight, eps);

        assert!(
            arrays_approx_equal(&x_scalar, &x_simd, 1e-5),
            "SIMD and scalar rmsnorm results differ with remainder"
        );
    }

    #[test]
    fn test_rmsnorm_simd_with_weights() {
        // Test with non-uniform weights
        let mut x_scalar = Array1::from_vec((0..16).map(|i| (i + 1) as f32).collect());
        let mut x_simd = x_scalar.clone();
        let weight = Array1::from_vec((0..16).map(|i| 0.5 + 0.1 * i as f32).collect());
        let eps = 1e-5;

        rmsnorm(&mut x_scalar, &weight, eps);
        rmsnorm_simd(&mut x_simd, &weight, eps);

        assert!(
            arrays_approx_equal(&x_scalar, &x_simd, 1e-5),
            "SIMD and scalar rmsnorm results differ with non-uniform weights"
        );
    }

    #[test]
    fn test_rmsnorm_simd_small_array() {
        // Test with array smaller than SIMD lane width
        let mut x_scalar = array![1.0, 2.0, 3.0];
        let mut x_simd = x_scalar.clone();
        let weight = array![1.0, 1.0, 1.0];
        let eps = 1e-5;

        rmsnorm(&mut x_scalar, &weight, eps);
        rmsnorm_simd(&mut x_simd, &weight, eps);

        assert!(
            arrays_approx_equal(&x_scalar, &x_simd, 1e-5),
            "SIMD and scalar rmsnorm results differ for small array"
        );
    }

    // -------------------------------------------------------------------------
    // Softmax SIMD Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_softmax_simd_matches_scalar() {
        // Test with array size that's a multiple of 8
        let mut x_scalar = Array1::from_vec((0..16).map(|i| i as f32).collect());
        let mut x_simd = x_scalar.clone();

        softmax(&mut x_scalar);
        softmax_simd(&mut x_simd);

        assert!(
            arrays_approx_equal(&x_scalar, &x_simd, 1e-5),
            "SIMD and scalar softmax results differ"
        );

        // Both should sum to 1
        assert!((x_scalar.sum() - 1.0).abs() < 1e-5);
        assert!((x_simd.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_simd_with_remainder() {
        // Test with array size that's NOT a multiple of 8
        let mut x_scalar = Array1::from_vec((0..13).map(|i| i as f32).collect());
        let mut x_simd = x_scalar.clone();

        softmax(&mut x_scalar);
        softmax_simd(&mut x_simd);

        assert!(
            arrays_approx_equal(&x_scalar, &x_simd, 1e-5),
            "SIMD and scalar softmax results differ with remainder"
        );
    }

    #[test]
    fn test_softmax_simd_numerical_stability() {
        // Test with large values (numerical stability)
        let mut x_scalar = Array1::from_vec(vec![
            1000.0, 1001.0, 1002.0, 999.0, 998.0, 997.0, 996.0, 995.0,
        ]);
        let mut x_simd = x_scalar.clone();

        softmax(&mut x_scalar);
        softmax_simd(&mut x_simd);

        assert!(
            arrays_approx_equal(&x_scalar, &x_simd, 1e-5),
            "SIMD and scalar softmax results differ with large values"
        );

        // Both should sum to 1
        assert!((x_scalar.sum() - 1.0).abs() < 1e-5);
        assert!((x_simd.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_simd_negative_values() {
        // Test with negative values
        let mut x_scalar = Array1::from_vec(vec![-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0]);
        let mut x_simd = x_scalar.clone();

        softmax(&mut x_scalar);
        softmax_simd(&mut x_simd);

        assert!(
            arrays_approx_equal(&x_scalar, &x_simd, 1e-5),
            "SIMD and scalar softmax results differ with negative values"
        );
    }

    // -------------------------------------------------------------------------
    // SiLU SIMD Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_silu_simd_matches_scalar() {
        // Test with array size that's a multiple of 8
        let x = Array1::from_vec((-8..8).map(|i| i as f32 * 0.5).collect());

        let result_scalar = silu(&x);
        let result_simd = silu_simd(&x);

        assert!(
            arrays_approx_equal(&result_scalar, &result_simd, 1e-5),
            "SIMD and scalar SiLU results differ"
        );
    }

    #[test]
    fn test_silu_simd_with_remainder() {
        // Test with array size that's NOT a multiple of 8
        let x = Array1::from_vec((-6..7).map(|i| i as f32 * 0.5).collect());

        let result_scalar = silu(&x);
        let result_simd = silu_simd(&x);

        assert!(
            arrays_approx_equal(&result_scalar, &result_simd, 1e-5),
            "SIMD and scalar SiLU results differ with remainder"
        );
    }

    #[test]
    fn test_silu_simd_properties() {
        let x = Array1::from_vec(vec![0.0, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0]);
        let result = silu_simd(&x);

        // SiLU(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-6);

        // SiLU(x) > 0 for x > 0
        assert!(result[1] > 0.0);
        assert!(result[3] > 0.0);
        assert!(result[5] > 0.0);
        assert!(result[7] > 0.0);

        // SiLU(x) < 0 for x < 0
        assert!(result[2] < 0.0);
        assert!(result[4] < 0.0);
        assert!(result[6] < 0.0);
    }

    // -------------------------------------------------------------------------
    // RoPE SIMD Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_apply_rope_simd_matches_scalar() {
        let n_heads = 4;
        let head_dim = 16; // Multiple of 8
        let mut x_scalar =
            Array2::from_shape_fn((n_heads, head_dim), |(i, j)| (i * head_dim + j) as f32);
        let mut x_simd = x_scalar.clone();

        let inv_freq = init_rope_freqs(head_dim, 10000.0);
        let (cos, sin) = rope_embeddings(&inv_freq, 5);

        apply_rope(&mut x_scalar, &cos, &sin);
        apply_rope_simd(&mut x_simd, &cos, &sin);

        // Compare element by element
        for h in 0..n_heads {
            for d in 0..head_dim {
                assert!(
                    (x_scalar[[h, d]] - x_simd[[h, d]]).abs() < 1e-5,
                    "SIMD and scalar RoPE results differ at [{}, {}]",
                    h,
                    d
                );
            }
        }
    }

    #[test]
    fn test_apply_rope_simd_with_remainder() {
        let n_heads = 4;
        let head_dim = 12; // NOT a multiple of 8
        let mut x_scalar =
            Array2::from_shape_fn((n_heads, head_dim), |(i, j)| (i * head_dim + j) as f32);
        let mut x_simd = x_scalar.clone();

        let inv_freq = init_rope_freqs(head_dim, 10000.0);
        let (cos, sin) = rope_embeddings(&inv_freq, 5);

        apply_rope(&mut x_scalar, &cos, &sin);
        apply_rope_simd(&mut x_simd, &cos, &sin);

        // Compare element by element
        for h in 0..n_heads {
            for d in 0..head_dim {
                assert!(
                    (x_scalar[[h, d]] - x_simd[[h, d]]).abs() < 1e-5,
                    "SIMD and scalar RoPE results differ at [{}, {}] with remainder",
                    h,
                    d
                );
            }
        }
    }

    #[test]
    fn test_apply_rope_simd_position_zero() {
        let n_heads = 2;
        let head_dim = 16;
        let mut x_scalar =
            Array2::from_shape_fn((n_heads, head_dim), |(i, j)| (i * head_dim + j) as f32);
        let mut x_simd = x_scalar.clone();
        let x_original = x_scalar.clone();

        let inv_freq = init_rope_freqs(head_dim, 10000.0);
        let (cos, sin) = rope_embeddings(&inv_freq, 0);

        apply_rope(&mut x_scalar, &cos, &sin);
        apply_rope_simd(&mut x_simd, &cos, &sin);

        // At position 0, cos=1 and sin=0, so rotation is identity
        // However, due to how rope_embeddings computes, we just verify consistency
        for h in 0..n_heads {
            for d in 0..head_dim {
                assert!(
                    (x_scalar[[h, d]] - x_simd[[h, d]]).abs() < 1e-5,
                    "SIMD and scalar RoPE results differ at position 0"
                );
            }
        }

        // Both should match original since sin=0, cos=1 means identity rotation
        for h in 0..n_heads {
            for d in 0..head_dim {
                assert!(
                    (x_scalar[[h, d]] - x_original[[h, d]]).abs() < 1e-5,
                    "RoPE at position 0 should be identity rotation"
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Fast (auto-selecting) function tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_fast_rmsnorm() {
        let mut x = Array1::from_vec((0..16).map(|i| (i + 1) as f32).collect());
        let weight = Array1::ones(16);
        fast_rmsnorm(&mut x, &weight, 1e-5);

        // After normalization, RMS should be approximately 1
        let squares: f32 = x.iter().map(|v| v * v).sum();
        let rms = (squares / x.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_fast_softmax() {
        let mut x = Array1::from_vec((0..16).map(|i| i as f32).collect());
        fast_softmax(&mut x);

        // Should sum to 1
        assert!((x.sum() - 1.0).abs() < 1e-5);

        // Should be monotonically increasing
        for i in 1..x.len() {
            assert!(x[i] > x[i - 1]);
        }
    }

    #[test]
    fn test_fast_silu() {
        let x = Array1::from_vec(vec![0.0, 1.0, -1.0, 2.0]);
        let result = fast_silu(&x);

        // SiLU(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-6);

        // SiLU(x) > 0 for x > 0
        assert!(result[1] > 0.0);
        assert!(result[3] > 0.0);

        // SiLU(x) < 0 for x < 0
        assert!(result[2] < 0.0);
    }

    #[test]
    fn test_fast_apply_rope() {
        let n_heads = 2;
        let head_dim = 16;
        let mut x = Array2::from_shape_fn((n_heads, head_dim), |(i, j)| (i * head_dim + j) as f32);
        let x_original = x.clone();

        let inv_freq = init_rope_freqs(head_dim, 10000.0);
        let (cos, sin) = rope_embeddings(&inv_freq, 1);

        fast_apply_rope(&mut x, &cos, &sin);

        // Should modify the tensor
        assert_ne!(x, x_original);

        // Shape should remain the same
        assert_eq!(x.shape(), &[n_heads, head_dim]);
    }
}

// =============================================================================
// Tests for fast_* functions without SIMD feature (fallback to scalar)
// =============================================================================

#[cfg(not(feature = "simd"))]
mod fast_fallback_tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_fast_rmsnorm_fallback() {
        let mut x = array![1.0, 2.0, 3.0];
        let weight = array![1.0, 1.0, 1.0];
        fast_rmsnorm(&mut x, &weight, 1e-5);

        // After normalization, RMS should be approximately 1
        let squares: f32 = x.iter().map(|v| v * v).sum();
        let rms = (squares / x.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_fast_softmax_fallback() {
        let mut x = array![1.0, 2.0, 3.0];
        fast_softmax(&mut x);

        // Should sum to 1
        assert!((x.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_fast_silu_fallback() {
        let x = array![0.0, 1.0, -1.0];
        let result = fast_silu(&x);

        // SiLU(0) = 0
        assert!((result[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_fast_apply_rope_fallback() {
        let n_heads = 2;
        let head_dim = 4;
        let mut x = Array2::from_shape_fn((n_heads, head_dim), |(i, j)| (i * head_dim + j) as f32);
        let x_original = x.clone();

        let inv_freq = init_rope_freqs(head_dim, 10000.0);
        let (cos, sin) = rope_embeddings(&inv_freq, 1);

        fast_apply_rope(&mut x, &cos, &sin);

        // Should modify the tensor
        assert_ne!(x, x_original);
    }
}

// =============================================================================
// Parallel Kernel Tests
// These tests verify that parallel implementations produce identical results
// to the scalar implementations.
// =============================================================================

#[cfg(feature = "parallel")]
mod parallel_tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    /// Helper to compare arrays with tolerance
    fn arrays_approx_equal(a: &Array1<f32>, b: &Array1<f32>, eps: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
    }

    // -------------------------------------------------------------------------
    // Parallel matmul Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_matmul_vec_parallel_matches_scalar() {
        let w = Array2::from_shape_fn((64, 32), |(i, j)| (i * 32 + j) as f32 * 0.01);
        let x = Array1::from_vec((0..32).map(|i| i as f32 * 0.1).collect());

        let result_scalar = matmul_vec(&w, &x);
        let result_parallel = matmul_vec_parallel(&w, &x);

        assert!(
            arrays_approx_equal(&result_scalar, &result_parallel, 1e-3),
            "Parallel and scalar matmul_vec results differ"
        );
    }

    #[test]
    fn test_matmul_vec_parallel_small() {
        let w = array![[1.0, 2.0], [3.0, 4.0]];
        let x = array![1.0, 2.0];

        let result_scalar = matmul_vec(&w, &x);
        let result_parallel = matmul_vec_parallel(&w, &x);

        assert_eq!(result_scalar, array![5.0, 11.0]);
        assert!(
            arrays_approx_equal(&result_scalar, &result_parallel, 1e-6),
            "Parallel and scalar matmul_vec results differ for small matrix"
        );
    }

    #[test]
    fn test_matmul_vec_into_parallel_matches_scalar() {
        let w = Array2::from_shape_fn((64, 32), |(i, j)| (i * 32 + j) as f32 * 0.01);
        let x = Array1::from_vec((0..32).map(|i| i as f32 * 0.1).collect());

        let mut out_scalar = Array1::zeros(64);
        let mut out_parallel = Array1::zeros(64);

        matmul_vec_into(&w, &x, &mut out_scalar);
        matmul_vec_into_parallel(&w, &x, &mut out_parallel);

        assert!(
            arrays_approx_equal(&out_scalar, &out_parallel, 1e-3),
            "Parallel and scalar matmul_vec_into results differ"
        );
    }

    // -------------------------------------------------------------------------
    // Parallel weighted_sum_rows Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_weighted_sum_rows_parallel_matches_scalar() {
        let weights = Array1::from_vec(vec![0.2, 0.3, 0.5]);
        let matrix = Array2::from_shape_fn((3, 16), |(i, j)| (i * 16 + j) as f32);

        let mut out_scalar = Array1::zeros(16);
        let mut out_parallel = Array1::zeros(16);

        {
            let mut view_scalar = out_scalar.view_mut();
            weighted_sum_rows(weights.view(), matrix.view(), &mut view_scalar);
        }
        {
            let mut view_parallel = out_parallel.view_mut();
            weighted_sum_rows_parallel(weights.view(), matrix.view(), &mut view_parallel);
        }

        assert!(
            arrays_approx_equal(&out_scalar, &out_parallel, 1e-5),
            "Parallel and scalar weighted_sum_rows results differ"
        );
    }

    // -------------------------------------------------------------------------
    // Parallel attention scores Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_compute_attention_scores_parallel_matches_scalar() {
        let query = Array1::from_vec((0..32).map(|i| i as f32 * 0.1).collect());
        let keys = Array2::from_shape_fn((16, 32), |(i, j)| (i * 32 + j) as f32 * 0.01);
        let scale = 0.125f32;

        let mut scores_scalar = Array1::zeros(16);
        let mut scores_parallel = Array1::zeros(16);

        {
            let mut view_scalar = scores_scalar.view_mut();
            compute_attention_scores(query.view(), keys.view(), &mut view_scalar, scale);
        }
        {
            let mut view_parallel = scores_parallel.view_mut();
            compute_attention_scores_parallel(query.view(), keys.view(), &mut view_parallel, scale);
        }

        assert!(
            arrays_approx_equal(&scores_scalar, &scores_parallel, 1e-3),
            "Parallel and scalar compute_attention_scores results differ"
        );
    }

    // -------------------------------------------------------------------------
    // Fast (auto-selecting) function tests with parallel
    // -------------------------------------------------------------------------

    #[test]
    fn test_fast_matmul_vec() {
        let w = array![[1.0, 2.0], [3.0, 4.0]];
        let x = array![1.0, 2.0];
        let result = fast_matmul_vec(&w, &x);

        assert_eq!(result, array![5.0, 11.0]);
    }

    #[test]
    fn test_fast_matmul_vec_into() {
        let w = array![[1.0, 2.0], [3.0, 4.0]];
        let x = array![1.0, 2.0];
        let mut out = Array1::zeros(2);

        fast_matmul_vec_into(&w, &x, &mut out);

        assert_eq!(out, array![5.0, 11.0]);
    }

    #[test]
    fn test_fast_weighted_sum_rows() {
        let weights = array![0.5, 0.5];
        let matrix = array![[2.0, 4.0], [6.0, 8.0]];
        let mut out = Array1::zeros(2);

        {
            let mut view = out.view_mut();
            fast_weighted_sum_rows(weights.view(), matrix.view(), &mut view);
        }

        // Result should be (0.5 * [2, 4] + 0.5 * [6, 8]) = [4, 6]
        assert!((out[0] - 4.0).abs() < 1e-6);
        assert!((out[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_fast_compute_attention_scores() {
        let query = array![1.0, 0.0];
        let keys = array![[1.0, 0.0], [0.0, 1.0]];
        let scale = 1.0f32;
        let mut scores = Array1::zeros(2);

        {
            let mut view = scores.view_mut();
            fast_compute_attention_scores(query.view(), keys.view(), &mut view, scale);
        }

        // First key matches query (dot product = 1), second doesn't (dot product = 0)
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!((scores[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_parallel_large_matrix() {
        // Test with a larger matrix to actually exercise parallelism
        let w = Array2::from_shape_fn((1024, 512), |(i, j)| ((i * 512 + j) % 1000) as f32 * 0.001);
        let x = Array1::from_vec((0..512).map(|i| (i % 100) as f32 * 0.01).collect());

        let result_scalar = matmul_vec(&w, &x);
        let result_parallel = matmul_vec_parallel(&w, &x);

        assert!(
            arrays_approx_equal(&result_scalar, &result_parallel, 1e-2),
            "Parallel and scalar matmul_vec results differ for large matrix"
        );
    }
}
