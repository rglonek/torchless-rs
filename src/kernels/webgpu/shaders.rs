//! WebGPU Shader Library
//!
//! This module contains the WGSL source code for compute shaders used in
//! WebGPU-accelerated transformer inference.
//!
//! # Kernel Overview
//!
//! - `rmsnorm_kernel`: RMSNorm normalization
//! - `softmax_kernel`: Numerically stable softmax
//! - `silu_kernel`: SiLU (Swish) activation function
//! - `rope_kernel`: Rotary Position Embedding
//! - `attention_scores_kernel`: Compute Q @ K^T attention scores
//! - `weighted_sum_kernel`: Compute attention-weighted sum of values
//! - `matmul_vec_kernel`: Matrix-vector multiplication
//! - `elementwise_mul_kernel`: Element-wise multiplication
//! - `elementwise_add_kernel`: Element-wise addition

/// WGSL source code for all compute kernels.
pub const WGSL_SHADERS_SOURCE: &str = r#"
// =============================================================================
// RMSNorm Kernel
// =============================================================================
// Computes: x = x * weight / sqrt(mean(x^2) + eps)
// Uses shared memory for parallel reduction of sum of squares.

struct RMSNormParams {
    n: u32,
    eps: f32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: RMSNormParams;

var<workgroup> wg_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm_kernel(@builtin(local_invocation_index) tid: u32) {
    let block_size = 256u;
    let n = params.n;
    
    // Compute sum of squares in parallel
    var local_sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < n) {
        let val = x[i];
        local_sum += val * val;
        i += block_size;
    }
    
    wg_shared[tid] = local_sum;
    workgroupBarrier();
    
    // Parallel reduction for sum
    var stride: u32 = block_size / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            wg_shared[tid] += wg_shared[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    let rms = sqrt(wg_shared[0u] / f32(n) + params.eps);
    let inv_rms = 1.0 / rms;
    workgroupBarrier();
    
    // Apply normalization and weight
    i = tid;
    while (i < n) {
        x[i] = x[i] * inv_rms * weight[i];
        i += block_size;
    }
}

// =============================================================================
// Softmax Kernel
// =============================================================================
// Computes: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// Numerically stable implementation with parallel max finding and sum.

struct SoftmaxParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<uniform> params: SoftmaxParams;

// Shared memory: first 256 for max, second 256 for sum
var<workgroup> max_shared: array<f32, 256>;
var<workgroup> sum_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_kernel(@builtin(local_invocation_index) tid: u32) {
    let block_size = 256u;
    let n = params.n;
    
    // Find max in parallel
    var local_max: f32 = -3.402823e+38;  // -INF
    var i: u32 = tid;
    while (i < n) {
        local_max = max(local_max, x[i]);
        i += block_size;
    }
    max_shared[tid] = local_max;
    workgroupBarrier();
    
    // Parallel reduction for max
    var stride: u32 = block_size / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            max_shared[tid] = max(max_shared[tid], max_shared[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let max_val = max_shared[0u];
    workgroupBarrier();
    
    // Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    i = tid;
    while (i < n) {
        let exp_val = exp(x[i] - max_val);
        x[i] = exp_val;
        local_sum += exp_val;
        i += block_size;
    }
    sum_shared[tid] = local_sum;
    workgroupBarrier();
    
    // Parallel reduction for sum
    stride = block_size / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let sum_val = sum_shared[0u];
    let inv_sum = 1.0 / sum_val;
    workgroupBarrier();
    
    // Normalize
    i = tid;
    while (i < n) {
        x[i] *= inv_sum;
        i += block_size;
    }
}

// =============================================================================
// SiLU Activation Kernel
// =============================================================================
// Computes: y = x * sigmoid(x) = x / (1 + exp(-x))

struct SiluParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: SiluParams;

@compute @workgroup_size(256)
fn silu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.n) {
        let val = x[idx];
        y[idx] = val / (1.0 + exp(-val));
    }
}

// =============================================================================
// RoPE (Rotary Position Embedding) Kernel
// =============================================================================
// Uses half-split layout: rotate dim i with dim i + head_dim/2
// x: [n_heads, head_dim], cos/sin: [half_dim]

struct RopeParams {
    n_heads: u32,
    head_dim: u32,
    half_dim: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<uniform> params: RopeParams;

@compute @workgroup_size(256)
fn rope_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.n_heads * params.half_dim;
    if (idx < total) {
        let h = idx / params.half_dim;
        let i = idx % params.half_dim;
        
        let base = h * params.head_dim;
        let xi = x[base + i];
        let yi = x[base + i + params.half_dim];
        let c = cos_table[i];
        let s = sin_table[i];
        
        x[base + i] = xi * c - yi * s;
        x[base + i + params.half_dim] = xi * s + yi * c;
    }
}

// =============================================================================
// Attention Scores Kernel
// =============================================================================
// Computes: scores[i] = keys[i, :].dot(query) * scale
// query: [head_dim], keys: [seq_len, head_dim] -> scores: [seq_len]

struct AttentionScoresParams {
    seq_len: u32,
    head_dim: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: AttentionScoresParams;

@compute @workgroup_size(256)
fn attention_scores_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < params.seq_len) {
        var dot: f32 = 0.0;
        let key_offset = i * params.head_dim;
        for (var j = 0u; j < params.head_dim; j++) {
            dot += query[j] * keys[key_offset + j];
        }
        scores[i] = dot * params.scale;
    }
}

// =============================================================================
// Weighted Sum Kernel
// =============================================================================
// Computes: out[j] = sum_i(weights[i] * matrix[i, j])
// weights: [n], matrix: [n, d] -> out: [d]

struct WeightedSumParams {
    n: u32,
    d: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> matrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: WeightedSumParams;

@compute @workgroup_size(256)
fn weighted_sum_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j < params.d) {
        var sum: f32 = 0.0;
        for (var i = 0u; i < params.n; i++) {
            sum += weights[i] * matrix[i * params.d + j];
        }
        out[j] = sum;
    }
}

// =============================================================================
// Matrix-Vector Multiplication Kernel
// =============================================================================
// Computes: out = W @ x
// W: [rows, cols], x: [cols] -> out: [rows]

struct MatmulVecParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: MatmulVecParams;

@compute @workgroup_size(256)
fn matmul_vec_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row < params.rows) {
        var sum: f32 = 0.0;
        let w_offset = row * params.cols;
        for (var j = 0u; j < params.cols; j++) {
            sum += weights[w_offset + j] * x[j];
        }
        out[row] = sum;
    }
}

// =============================================================================
// Element-wise Operations
// =============================================================================

struct ElementwiseParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: ElementwiseParams;

@compute @workgroup_size(256)
fn elementwise_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.n) {
        c[idx] = a[idx] * b[idx];
    }
}

@compute @workgroup_size(256)
fn elementwise_add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < params.n) {
        c[idx] = a[idx] + b[idx];
    }
}
"#;

/// Kernel entry point names for pipeline creation.
pub mod kernel_names {
    pub const RMSNORM: &str = "rmsnorm_kernel";
    pub const SOFTMAX: &str = "softmax_kernel";
    pub const SILU: &str = "silu_kernel";
    pub const ROPE: &str = "rope_kernel";
    pub const ATTENTION_SCORES: &str = "attention_scores_kernel";
    pub const WEIGHTED_SUM: &str = "weighted_sum_kernel";
    pub const MATMUL_VEC: &str = "matmul_vec_kernel";
    pub const ELEMENTWISE_MUL: &str = "elementwise_mul_kernel";
    pub const ELEMENTWISE_ADD: &str = "elementwise_add_kernel";
}
