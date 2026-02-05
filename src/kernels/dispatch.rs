//! Runtime CPU Feature Dispatch
//!
//! This module provides runtime detection of CPU features and automatic
//! dispatch to the optimal kernel implementation.
//!
//! # Supported Features
//! - AVX-512F (x86_64): 16-wide SIMD operations
//! - AVX2/FMA (x86_64): 8-wide SIMD operations (via wide crate)
//! - NEON (aarch64): 4-wide SIMD operations
//!
//! # Usage
//! The `Dispatcher` struct caches feature detection results and provides
//! methods that automatically route to the best available implementation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use std::sync::OnceLock;

// Import architecture-specific modules
// These are conditionally used based on target architecture
#[allow(unused_imports)]
use super::avx512;
#[allow(unused_imports)]
use super::neon;

/// CPU feature flags detected at runtime.
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    /// AVX-512F is available (x86_64 only)
    pub avx512f: bool,
    /// AVX2 is available (x86_64 only)
    pub avx2: bool,
    /// FMA is available (x86_64 only)
    pub fma: bool,
    /// NEON is available (aarch64 only)
    pub neon: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime.
    pub fn detect() -> Self {
        Self {
            avx512f: Self::detect_avx512f(),
            avx2: Self::detect_avx2(),
            fma: Self::detect_fma(),
            neon: Self::detect_neon(),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_avx512f() -> bool {
        is_x86_feature_detected!("avx512f")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect_avx512f() -> bool {
        false
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect_avx2() -> bool {
        false
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_fma() -> bool {
        is_x86_feature_detected!("fma")
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect_fma() -> bool {
        false
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_neon() -> bool {
        true // NEON is always available on aarch64
    }

    #[cfg(not(target_arch = "aarch64"))]
    fn detect_neon() -> bool {
        false
    }

    /// Get the best SIMD width available.
    pub fn best_simd_width(&self) -> usize {
        if self.avx512f {
            16
        } else if self.avx2 || self.neon {
            // wide crate uses 8, NEON uses 4 but we process 16 at a time
            8
        } else {
            1
        }
    }

    /// Get a human-readable description of available features.
    pub fn describe(&self) -> String {
        let mut features = Vec::new();

        if self.avx512f {
            features.push("AVX-512F");
        }
        if self.avx2 {
            features.push("AVX2");
        }
        if self.fma {
            features.push("FMA");
        }
        if self.neon {
            features.push("NEON");
        }

        if features.is_empty() {
            "scalar only".to_string()
        } else {
            features.join(", ")
        }
    }
}

/// Global CPU features, detected once at first use.
static CPU_FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

/// Get the detected CPU features.
pub fn cpu_features() -> &'static CpuFeatures {
    CPU_FEATURES.get_or_init(CpuFeatures::detect)
}

/// Kernel dispatcher that routes to the optimal implementation.
#[derive(Debug, Clone, Copy)]
pub struct Dispatcher {
    features: CpuFeatures,
}

impl Default for Dispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Dispatcher {
    /// Create a new dispatcher with auto-detected features.
    pub fn new() -> Self {
        Self {
            features: *cpu_features(),
        }
    }

    /// Create a dispatcher with specific features (useful for testing).
    pub fn with_features(features: CpuFeatures) -> Self {
        Self { features }
    }

    /// Get the CPU features this dispatcher is using.
    pub fn features(&self) -> CpuFeatures {
        self.features
    }

    /// Matrix-vector multiplication with automatic dispatch.
    /// W: (n, d), x: (d,) -> out: (n,)
    pub fn matmul_vec(&self, w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
        let (rows, _) = w.dim();
        let mut output = Array1::zeros(rows);
        self.matmul_vec_into(w, x, &mut output);
        output
    }

    /// Matrix-vector multiplication with pre-allocated output.
    pub fn matmul_vec_into(&self, w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
        let (rows, cols) = w.dim();

        // Try to get contiguous slices
        let w_slice = w.as_slice();
        let x_slice = x.as_slice();
        let out_slice = out.as_slice_mut();

        match (w_slice, x_slice, out_slice) {
            (Some(w), Some(x), Some(out)) => {
                // Use architecture-specific kernels
                #[cfg(target_arch = "x86_64")]
                if self.features.avx512f {
                    unsafe {
                        avx512::matmul_vec_avx512(w, x, out, rows, cols);
                    }
                    return;
                }

                #[cfg(target_arch = "aarch64")]
                if self.features.neon {
                    unsafe {
                        neon::matmul_vec_neon(w, x, out, rows, cols);
                    }
                    return;
                }

                // Fall back to scalar
                self.matmul_vec_scalar(w, x, out, rows, cols);
            }
            _ => {
                // Non-contiguous: use ndarray
                out.assign(&w.dot(x));
            }
        }
    }

    fn matmul_vec_scalar(&self, w: &[f32], x: &[f32], out: &mut [f32], rows: usize, cols: usize) {
        for row in 0..rows {
            let row_offset = row * cols;
            let mut sum = 0.0f32;
            for col in 0..cols {
                sum += w[row_offset + col] * x[col];
            }
            out[row] = sum;
        }
    }

    /// RMSNorm with automatic dispatch.
    pub fn rmsnorm(&self, x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
        let x_slice = x.as_slice_mut();
        let w_slice = weight.as_slice();

        match (x_slice, w_slice) {
            (Some(x), Some(w)) => {
                #[cfg(target_arch = "x86_64")]
                if self.features.avx512f {
                    unsafe {
                        avx512::rmsnorm_avx512(x, w, eps);
                    }
                    return;
                }

                #[cfg(target_arch = "aarch64")]
                if self.features.neon {
                    unsafe {
                        neon::rmsnorm_neon(x, w, eps);
                    }
                    return;
                }

                // Fall back to scalar
                self.rmsnorm_scalar(x, w, eps);
            }
            _ => {
                // Non-contiguous: use existing implementation
                super::rmsnorm(x, weight, eps);
            }
        }
    }

    fn rmsnorm_scalar(&self, x: &mut [f32], weight: &[f32], eps: f32) {
        let n = x.len();
        let sum_sq: f32 = x.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        for i in 0..n {
            x[i] = x[i] * inv_rms * weight[i];
        }
    }

    /// Softmax with automatic dispatch.
    pub fn softmax(&self, x: &mut Array1<f32>) {
        let x_slice = x.as_slice_mut();

        match x_slice {
            Some(x) => {
                #[cfg(target_arch = "x86_64")]
                if self.features.avx512f {
                    unsafe {
                        avx512::softmax_avx512(x);
                    }
                    return;
                }

                #[cfg(target_arch = "aarch64")]
                if self.features.neon {
                    unsafe {
                        neon::softmax_neon(x);
                    }
                    return;
                }

                // Fall back to scalar
                self.softmax_scalar(x);
            }
            None => {
                super::softmax(x);
            }
        }
    }

    fn softmax_scalar(&self, x: &mut [f32]) {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in x.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }

    /// SiLU activation with automatic dispatch.
    pub fn silu(&self, x: &Array1<f32>) -> Array1<f32> {
        let x_slice = x.as_slice();

        match x_slice {
            Some(x_s) => {
                let mut output = vec![0.0f32; x_s.len()];

                #[cfg(target_arch = "x86_64")]
                if self.features.avx512f {
                    unsafe {
                        avx512::silu_avx512(x_s, &mut output);
                    }
                    return Array1::from_vec(output);
                }

                #[cfg(target_arch = "aarch64")]
                if self.features.neon {
                    unsafe {
                        neon::silu_neon(x_s, &mut output);
                    }
                    return Array1::from_vec(output);
                }

                // Fall back to scalar
                for (i, &v) in x_s.iter().enumerate() {
                    output[i] = v / (1.0 + (-v).exp());
                }
                Array1::from_vec(output)
            }
            None => super::silu(x),
        }
    }

    /// Apply RoPE with automatic dispatch.
    pub fn apply_rope(&self, x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
        let (n_heads, head_dim) = x.dim();
        let x_slice = x.as_slice_mut();
        let cos_slice = cos.as_slice();
        let sin_slice = sin.as_slice();

        match (x_slice, cos_slice, sin_slice) {
            (Some(x), Some(cos), Some(sin)) => {
                #[cfg(target_arch = "x86_64")]
                if self.features.avx512f {
                    unsafe {
                        avx512::apply_rope_avx512(x, cos, sin, n_heads, head_dim);
                    }
                    return;
                }

                #[cfg(target_arch = "aarch64")]
                if self.features.neon {
                    unsafe {
                        neon::apply_rope_neon(x, cos, sin, n_heads, head_dim);
                    }
                    return;
                }

                // Fall back to scalar
                self.apply_rope_scalar(x, cos, sin, n_heads, head_dim);
            }
            _ => {
                super::apply_rope(x, cos, sin);
            }
        }
    }

    fn apply_rope_scalar(
        &self,
        x: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        n_heads: usize,
        head_dim: usize,
    ) {
        let half = head_dim / 2;
        for h in 0..n_heads {
            let offset = h * head_dim;
            for i in 0..half {
                let xi = x[offset + i];
                let yi = x[offset + half + i];
                let c = cos[i];
                let s = sin[i];
                x[offset + i] = xi * c - yi * s;
                x[offset + half + i] = xi * s + yi * c;
            }
        }
    }

    /// Compute attention scores with automatic dispatch.
    pub fn compute_attention_scores(
        &self,
        query: ArrayView1<f32>,
        keys: ArrayView2<f32>,
        scores: &mut ArrayViewMut1<f32>,
        scale: f32,
    ) {
        let n_keys = keys.nrows();
        let key_dim = keys.ncols();

        let q_slice = query.as_slice();
        let k_slice = keys.as_slice();
        let s_slice = scores.as_slice_mut();

        match (q_slice, k_slice, s_slice) {
            (Some(q), Some(k), Some(s)) => {
                #[cfg(target_arch = "x86_64")]
                if self.features.avx512f {
                    unsafe {
                        avx512::compute_attention_scores_avx512(q, k, s, n_keys, key_dim, scale);
                    }
                    return;
                }

                #[cfg(target_arch = "aarch64")]
                if self.features.neon {
                    unsafe {
                        neon::compute_attention_scores_neon(q, k, s, n_keys, key_dim, scale);
                    }
                    return;
                }

                // Fall back to scalar
                for i in 0..n_keys {
                    let key_offset = i * key_dim;
                    let mut dot = 0.0f32;
                    for j in 0..key_dim {
                        dot += q[j] * k[key_offset + j];
                    }
                    s[i] = dot * scale;
                }
            }
            _ => {
                super::compute_attention_scores(query, keys, scores, scale);
            }
        }
    }

    /// Weighted sum of rows with automatic dispatch.
    pub fn weighted_sum_rows(
        &self,
        weights: ArrayView1<f32>,
        matrix: ArrayView2<f32>,
        out: &mut ArrayViewMut1<f32>,
    ) {
        let n_rows = matrix.nrows();
        let n_cols = matrix.ncols();

        let w_slice = weights.as_slice();
        let m_slice = matrix.as_slice();
        let o_slice = out.as_slice_mut();

        match (w_slice, m_slice, o_slice) {
            (Some(w), Some(m), Some(o)) => {
                #[cfg(target_arch = "x86_64")]
                if self.features.avx512f {
                    unsafe {
                        avx512::weighted_sum_rows_avx512(w, m, o, n_rows, n_cols);
                    }
                    return;
                }

                #[cfg(target_arch = "aarch64")]
                if self.features.neon {
                    unsafe {
                        neon::weighted_sum_rows_neon(w, m, o, n_rows, n_cols);
                    }
                    return;
                }

                // Fall back to scalar
                o.fill(0.0);
                for i in 0..n_rows {
                    let weight = w[i];
                    let row_offset = i * n_cols;
                    for j in 0..n_cols {
                        o[j] += weight * m[row_offset + j];
                    }
                }
            }
            _ => {
                super::weighted_sum_rows(weights, matrix, out);
            }
        }
    }
}

/// Get the global dispatcher instance.
static GLOBAL_DISPATCHER: OnceLock<Dispatcher> = OnceLock::new();

/// Get a reference to the global dispatcher.
pub fn global_dispatcher() -> &'static Dispatcher {
    GLOBAL_DISPATCHER.get_or_init(Dispatcher::new)
}

// =============================================================================
// Convenience functions that use the global dispatcher
// =============================================================================

/// Auto-dispatched matrix-vector multiplication.
pub fn dispatch_matmul_vec(w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
    global_dispatcher().matmul_vec(w, x)
}

/// Auto-dispatched matrix-vector multiplication with pre-allocated output.
pub fn dispatch_matmul_vec_into(w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
    global_dispatcher().matmul_vec_into(w, x, out);
}

/// Auto-dispatched RMSNorm.
pub fn dispatch_rmsnorm(x: &mut Array1<f32>, weight: &Array1<f32>, eps: f32) {
    global_dispatcher().rmsnorm(x, weight, eps);
}

/// Auto-dispatched softmax.
pub fn dispatch_softmax(x: &mut Array1<f32>) {
    global_dispatcher().softmax(x);
}

/// Auto-dispatched SiLU activation.
pub fn dispatch_silu(x: &Array1<f32>) -> Array1<f32> {
    global_dispatcher().silu(x)
}

/// Auto-dispatched RoPE application.
pub fn dispatch_apply_rope(x: &mut Array2<f32>, cos: &Array1<f32>, sin: &Array1<f32>) {
    global_dispatcher().apply_rope(x, cos, sin);
}

/// Auto-dispatched attention scores computation.
pub fn dispatch_compute_attention_scores(
    query: ArrayView1<f32>,
    keys: ArrayView2<f32>,
    scores: &mut ArrayViewMut1<f32>,
    scale: f32,
) {
    global_dispatcher().compute_attention_scores(query, keys, scores, scale);
}

/// Auto-dispatched weighted sum of rows.
pub fn dispatch_weighted_sum_rows(
    weights: ArrayView1<f32>,
    matrix: ArrayView2<f32>,
    out: &mut ArrayViewMut1<f32>,
) {
    global_dispatcher().weighted_sum_rows(weights, matrix, out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cpu_features_detection() {
        let features = CpuFeatures::detect();
        let desc = features.describe();
        println!("Detected CPU features: {}", desc);

        // At least one of these should be true on modern hardware
        // (or scalar on very old/exotic hardware)
        assert!(
            features.avx512f || features.avx2 || features.neon || desc == "scalar only"
        );
    }

    #[test]
    fn test_dispatcher_matmul_vec() {
        let dispatcher = Dispatcher::new();

        let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = array![1.0, 2.0, 3.0];

        let result = dispatcher.matmul_vec(&w, &x);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 14.0).abs() < 1e-5); // 1*1 + 2*2 + 3*3 = 14
        assert!((result[1] - 32.0).abs() < 1e-5); // 4*1 + 5*2 + 6*3 = 32
    }

    #[test]
    fn test_dispatcher_rmsnorm() {
        let dispatcher = Dispatcher::new();

        let mut x = array![1.0, 2.0, 3.0];
        let weight = array![1.0, 1.0, 1.0];
        let eps = 1e-5;

        dispatcher.rmsnorm(&mut x, &weight, eps);

        // After normalization, RMS should be approximately 1
        let squares: f32 = x.iter().map(|v| v * v).sum();
        let rms = (squares / x.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_dispatcher_softmax() {
        let dispatcher = Dispatcher::new();

        let mut x = array![1.0, 2.0, 3.0];
        dispatcher.softmax(&mut x);

        // Check sum is 1.0
        let sum: f32 = x.sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check monotonicity
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_dispatcher_silu() {
        let dispatcher = Dispatcher::new();

        let x = array![0.0, 1.0, -1.0];
        let result = dispatcher.silu(&x);

        // SiLU(0) = 0
        assert!(result[0].abs() < 1e-6);
        // SiLU(x) > 0 for x > 0
        assert!(result[1] > 0.0);
        // SiLU(x) < 0 for x < 0
        assert!(result[2] < 0.0);
    }

    #[test]
    fn test_dispatcher_apply_rope() {
        let dispatcher = Dispatcher::new();

        let n_heads = 2;
        let head_dim = 4;
        let mut x = Array2::from_shape_fn((n_heads, head_dim), |(i, j)| (i * head_dim + j) as f32);
        let x_original = x.clone();

        let inv_freq = super::super::init_rope_freqs(head_dim, 10000.0);
        let (cos, sin) = super::super::rope_embeddings(&inv_freq, 1);

        dispatcher.apply_rope(&mut x, &cos, &sin);

        // Should modify the tensor
        assert_ne!(x, x_original);
    }

    #[test]
    fn test_global_dispatcher() {
        // Ensure global dispatcher works
        let features = cpu_features();
        println!("Global CPU features: {}", features.describe());

        let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = array![1.0, 2.0, 3.0];

        let result = dispatch_matmul_vec(&w, &x);
        assert!((result[0] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_dispatch_functions() {
        // Test all dispatch convenience functions

        // matmul_vec
        let w = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let x = array![1.0, 2.0, 3.0];
        let result = dispatch_matmul_vec(&w, &x);
        assert_eq!(result.len(), 2);

        // matmul_vec_into
        let mut out = Array1::zeros(2);
        dispatch_matmul_vec_into(&w, &x, &mut out);
        assert!((out[0] - 14.0).abs() < 1e-5);

        // rmsnorm
        let mut y = array![1.0, 2.0, 3.0];
        let weight = array![1.0, 1.0, 1.0];
        dispatch_rmsnorm(&mut y, &weight, 1e-5);

        // softmax
        let mut z = array![1.0, 2.0, 3.0];
        dispatch_softmax(&mut z);
        assert!((z.sum() - 1.0).abs() < 1e-5);

        // silu
        let a = array![0.0, 1.0, -1.0];
        let b = dispatch_silu(&a);
        assert!(b[0].abs() < 1e-6);

        // apply_rope
        let mut rope_x = Array2::from_shape_fn((2, 4), |(i, j)| (i * 4 + j) as f32);
        let cos = array![1.0, 1.0];
        let sin = array![0.0, 0.0];
        dispatch_apply_rope(&mut rope_x, &cos, &sin);
    }
}
