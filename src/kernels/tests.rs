#[cfg(test)]
mod tests {
    use crate::kernels::*;
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
}
