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
///
/// Each kernel's global bindings are prefixed with a unique kernel abbreviation
/// to avoid name collisions, since all kernels share a single WGSL module.
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

@group(0) @binding(0) var<storage, read_write> rms_x: array<f32>;
@group(0) @binding(1) var<storage, read> rms_weight: array<f32>;
@group(0) @binding(2) var<uniform> rms_params: RMSNormParams;

var<workgroup> rms_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm_kernel(@builtin(local_invocation_index) tid: u32) {
    let block_size = 256u;
    let n = rms_params.n;
    
    // Compute sum of squares in parallel
    var local_sum: f32 = 0.0;
    var i: u32 = tid;
    while (i < n) {
        let val = rms_x[i];
        local_sum += val * val;
        i += block_size;
    }
    
    rms_shared[tid] = local_sum;
    workgroupBarrier();
    
    // Parallel reduction for sum
    var stride: u32 = block_size / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            rms_shared[tid] += rms_shared[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    let rms = sqrt(rms_shared[0u] / f32(n) + rms_params.eps);
    let inv_rms = 1.0 / rms;
    workgroupBarrier();
    
    // Apply normalization and weight
    i = tid;
    while (i < n) {
        rms_x[i] = rms_x[i] * inv_rms * rms_weight[i];
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

@group(0) @binding(0) var<storage, read_write> sm_x: array<f32>;
@group(0) @binding(1) var<uniform> sm_params: SoftmaxParams;

// Shared memory: first 256 for max, second 256 for sum
var<workgroup> sm_max_shared: array<f32, 256>;
var<workgroup> sm_sum_shared: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax_kernel(@builtin(local_invocation_index) tid: u32) {
    let block_size = 256u;
    let n = sm_params.n;
    
    // Find max in parallel
    var local_max: f32 = -3.402823e+38;  // -INF
    var i: u32 = tid;
    while (i < n) {
        local_max = max(local_max, sm_x[i]);
        i += block_size;
    }
    sm_max_shared[tid] = local_max;
    workgroupBarrier();
    
    // Parallel reduction for max
    var stride: u32 = block_size / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            sm_max_shared[tid] = max(sm_max_shared[tid], sm_max_shared[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let max_val = sm_max_shared[0u];
    workgroupBarrier();
    
    // Compute exp(x - max) and sum
    var local_sum: f32 = 0.0;
    i = tid;
    while (i < n) {
        let exp_val = exp(sm_x[i] - max_val);
        sm_x[i] = exp_val;
        local_sum += exp_val;
        i += block_size;
    }
    sm_sum_shared[tid] = local_sum;
    workgroupBarrier();
    
    // Parallel reduction for sum
    stride = block_size / 2u;
    while (stride > 0u) {
        if (tid < stride) {
            sm_sum_shared[tid] += sm_sum_shared[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let sum_val = sm_sum_shared[0u];
    let inv_sum = 1.0 / sum_val;
    workgroupBarrier();
    
    // Normalize
    i = tid;
    while (i < n) {
        sm_x[i] *= inv_sum;
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

@group(0) @binding(0) var<storage, read> silu_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> silu_y: array<f32>;
@group(0) @binding(2) var<uniform> silu_params: SiluParams;

@compute @workgroup_size(256)
fn silu_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < silu_params.n) {
        let val = silu_x[idx];
        silu_y[idx] = val / (1.0 + exp(-val));
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

@group(0) @binding(0) var<storage, read_write> rope_x: array<f32>;
@group(0) @binding(1) var<storage, read> rope_cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> rope_sin_table: array<f32>;
@group(0) @binding(3) var<uniform> rope_params: RopeParams;

@compute @workgroup_size(256)
fn rope_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = rope_params.n_heads * rope_params.half_dim;
    if (idx < total) {
        let h = idx / rope_params.half_dim;
        let i = idx % rope_params.half_dim;
        
        let base = h * rope_params.head_dim;
        let xi = rope_x[base + i];
        let yi = rope_x[base + i + rope_params.half_dim];
        let c = rope_cos_table[i];
        let s = rope_sin_table[i];
        
        rope_x[base + i] = xi * c - yi * s;
        rope_x[base + i + rope_params.half_dim] = xi * s + yi * c;
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

@group(0) @binding(0) var<storage, read> attn_query: array<f32>;
@group(0) @binding(1) var<storage, read> attn_keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> attn_scores: array<f32>;
@group(0) @binding(3) var<uniform> attn_params: AttentionScoresParams;

@compute @workgroup_size(256)
fn attention_scores_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < attn_params.seq_len) {
        var dot: f32 = 0.0;
        let key_offset = i * attn_params.head_dim;
        for (var j = 0u; j < attn_params.head_dim; j++) {
            dot += attn_query[j] * attn_keys[key_offset + j];
        }
        attn_scores[i] = dot * attn_params.scale;
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

@group(0) @binding(0) var<storage, read> ws_weights: array<f32>;
@group(0) @binding(1) var<storage, read> ws_matrix: array<f32>;
@group(0) @binding(2) var<storage, read_write> ws_out: array<f32>;
@group(0) @binding(3) var<uniform> ws_params: WeightedSumParams;

@compute @workgroup_size(256)
fn weighted_sum_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j < ws_params.d) {
        var sum: f32 = 0.0;
        for (var i = 0u; i < ws_params.n; i++) {
            sum += ws_weights[i] * ws_matrix[i * ws_params.d + j];
        }
        ws_out[j] = sum;
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

@group(0) @binding(0) var<storage, read> mv_weights: array<f32>;
@group(0) @binding(1) var<storage, read> mv_x: array<f32>;
@group(0) @binding(2) var<storage, read_write> mv_out: array<f32>;
@group(0) @binding(3) var<uniform> mv_params: MatmulVecParams;

@compute @workgroup_size(256)
fn matmul_vec_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row < mv_params.rows) {
        var sum: f32 = 0.0;
        let w_offset = row * mv_params.cols;
        for (var j = 0u; j < mv_params.cols; j++) {
            sum += mv_weights[w_offset + j] * mv_x[j];
        }
        mv_out[row] = sum;
    }
}

// =============================================================================
// Element-wise Operations
// =============================================================================

struct ElementwiseParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> ew_a: array<f32>;
@group(0) @binding(1) var<storage, read> ew_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> ew_c: array<f32>;
@group(0) @binding(3) var<uniform> ew_params: ElementwiseParams;

@compute @workgroup_size(256)
fn elementwise_mul_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < ew_params.n) {
        ew_c[idx] = ew_a[idx] * ew_b[idx];
    }
}

@compute @workgroup_size(256)
fn elementwise_add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < ew_params.n) {
        ew_c[idx] = ew_a[idx] + ew_b[idx];
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
