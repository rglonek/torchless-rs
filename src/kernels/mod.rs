use crate::tensor::Tensor1;
use ndarray::{Array1, Array2};

mod tests;

/// Matrix multiplication: out = W @ x
/// W: (n, d), x: (d,) -> out: (n,)
pub fn matmul_vec(w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
    w.dot(x)
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

/// Softmax with numerical stability (subtract max)
pub fn softmax(x: &mut Tensor1) {
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
