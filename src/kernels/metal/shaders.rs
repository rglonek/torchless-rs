//! Metal Shader Library
//!
//! This module contains the Metal Shading Language (MSL) source code for
//! GPU compute kernels used in transformer inference.
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

/// Metal Shading Language source code for all compute kernels.
pub const METAL_SHADERS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// =============================================================================
// RMSNorm Kernel
// =============================================================================
// Computes: x = x * weight / sqrt(mean(x^2) + eps)
// Uses shared memory for parallel reduction of sum of squares.

kernel void rmsnorm_kernel(
    device float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant int& n [[buffer(2)]],
    constant float& eps [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    // Compute sum of squares in parallel
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float val = x[i];
        local_sum += val * val;
    }
    
    // Store in shared memory
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for sum
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Compute RMS and apply normalization
    float rms = sqrt(shared[0] / float(n) + eps);
    float inv_rms = 1.0f / rms;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Apply normalization and weight
    for (int i = tid; i < n; i += block_size) {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

// =============================================================================
// Softmax Kernel
// =============================================================================
// Computes: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// Numerically stable implementation with parallel max finding and sum.

kernel void softmax_kernel(
    device float* x [[buffer(0)]],
    constant int& n [[buffer(1)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    // Shared memory layout: first half for max, second half for sum
    threadgroup float* max_shared = shared;
    threadgroup float* sum_shared = shared + block_size;
    
    // Find max in parallel
    float local_max = -INFINITY;
    for (int i = tid; i < n; i += block_size) {
        local_max = max(local_max, x[i]);
    }
    max_shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for max
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_shared[tid] = max(max_shared[tid], max_shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = max_shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float exp_val = exp(x[i] - max_val);
        x[i] = exp_val;
        local_sum += exp_val;
    }
    sum_shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction for sum
    for (uint stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = sum_shared[0];
    float inv_sum = 1.0f / sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize
    for (int i = tid; i < n; i += block_size) {
        x[i] *= inv_sum;
    }
}

// =============================================================================
// SiLU Activation Kernel
// =============================================================================
// Computes: y = x * sigmoid(x) = x / (1 + exp(-x))

kernel void silu_kernel(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant int& n [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        float val = x[idx];
        y[idx] = val / (1.0f + exp(-val));
    }
}

// =============================================================================
// RoPE (Rotary Position Embedding) Kernel
// =============================================================================
// Uses half-split layout: rotate dim i with dim i + head_dim/2
// x: [n_heads, head_dim], cos/sin: [half_dim]

kernel void rope_kernel(
    device float* x [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    constant int& n_heads [[buffer(3)]],
    constant int& head_dim [[buffer(4)]],
    constant int& half_dim [[buffer(5)]],
    uint idx [[thread_position_in_grid]]
) {
    int total = n_heads * half_dim;
    if (idx < uint(total)) {
        int h = idx / half_dim;      // head index
        int i = idx % half_dim;      // position within half
        
        int base = h * head_dim;
        float xi = x[base + i];
        float yi = x[base + i + half_dim];
        float c = cos_table[i];
        float s = sin_table[i];
        
        // Rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
        x[base + i] = xi * c - yi * s;
        x[base + i + half_dim] = xi * s + yi * c;
    }
}

// =============================================================================
// Attention Scores Kernel
// =============================================================================
// Computes: scores[i] = keys[i, :].dot(query) * scale
// query: [head_dim], keys: [seq_len, head_dim] -> scores: [seq_len]

kernel void attention_scores_kernel(
    device const float* query [[buffer(0)]],
    device const float* keys [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant int& seq_len [[buffer(3)]],
    constant int& head_dim [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < uint(seq_len)) {
        float dot = 0.0f;
        device const float* key_row = keys + i * head_dim;
        for (int j = 0; j < head_dim; j++) {
            dot += query[j] * key_row[j];
        }
        scores[i] = dot * scale;
    }
}

// =============================================================================
// Weighted Sum Kernel
// =============================================================================
// Computes: out[j] = sum_i(weights[i] * matrix[i, j])
// weights: [n], matrix: [n, d] -> out: [d]

kernel void weighted_sum_kernel(
    device const float* weights [[buffer(0)]],
    device const float* matrix [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& n [[buffer(3)]],
    constant int& d [[buffer(4)]],
    uint j [[thread_position_in_grid]]
) {
    if (j < uint(d)) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += weights[i] * matrix[i * d + j];
        }
        out[j] = sum;
    }
}

// =============================================================================
// Matrix-Vector Multiplication Kernel
// =============================================================================
// Computes: out = W @ x
// W: [rows, cols], x: [cols] -> out: [rows]
// Each thread computes one output element.

kernel void matmul_vec_kernel(
    device const float* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant int& cols [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row < uint(rows)) {
        float sum = 0.0f;
        device const float* w_row = weights + row * cols;
        for (int j = 0; j < cols; j++) {
            sum += w_row[j] * x[j];
        }
        out[row] = sum;
    }
}

// =============================================================================
// Tiled Matrix-Vector Multiplication Kernel
// =============================================================================
// More efficient implementation using threadgroup memory for shared x access.
// Uses TILE_SIZE threads per threadgroup, each computing partial dot products.

constant int TILE_SIZE = 256;

kernel void matmul_vec_tiled_kernel(
    device const float* weights [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& rows [[buffer(3)]],
    constant int& cols [[buffer(4)]],
    threadgroup float* shared_x [[threadgroup(0)]],
    uint row [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint block_size [[threads_per_threadgroup]]
) {
    if (row >= uint(rows)) return;
    
    float sum = 0.0f;
    device const float* w_row = weights + row * cols;
    
    // Process in tiles to improve cache utilization
    for (int tile_start = 0; tile_start < cols; tile_start += TILE_SIZE) {
        // Load x values into shared memory cooperatively
        int tile_end = min(tile_start + TILE_SIZE, cols);
        int tile_len = tile_end - tile_start;
        
        // Each thread loads its portion of x
        for (int i = tid; i < tile_len; i += int(block_size)) {
            shared_x[i] = x[tile_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product using shared x
        for (int j = 0; j < tile_len; j++) {
            sum += w_row[tile_start + j] * shared_x[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    out[row] = sum;
}

// =============================================================================
// Element-wise Operations
// =============================================================================

kernel void elementwise_mul_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        c[idx] = a[idx] * b[idx];
    }
}

kernel void elementwise_add_kernel(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        c[idx] = a[idx] + b[idx];
    }
}

// =============================================================================
// Fused Operations (for better performance)
// =============================================================================

// Fused SiLU + Element-wise Multiply
// Computes: out = silu(gate) * up
// Used in SwiGLU activation: gate_proj and up_proj outputs
kernel void fused_silu_mul_kernel(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant int& n [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx < uint(n)) {
        float g = gate[idx];
        float silu_g = g / (1.0f + exp(-g));
        out[idx] = silu_g * up[idx];
    }
}

// =============================================================================
// Matrix-Matrix Multiplication Kernel (simple implementation)
// =============================================================================
// Computes: C = A @ B
// A: [M, K], B: [K, N] -> C: [M, N]
// Note: For production, use Metal Performance Shaders (MPS) instead.

kernel void matmul_kernel(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row < uint(M) && col < uint(N)) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"#;

/// Kernel names for lookup in compiled library
pub mod kernel_names {
    pub const RMSNORM: &str = "rmsnorm_kernel";
    pub const SOFTMAX: &str = "softmax_kernel";
    pub const SILU: &str = "silu_kernel";
    pub const ROPE: &str = "rope_kernel";
    pub const ATTENTION_SCORES: &str = "attention_scores_kernel";
    pub const WEIGHTED_SUM: &str = "weighted_sum_kernel";
    pub const MATMUL_VEC: &str = "matmul_vec_kernel";
    pub const MATMUL_VEC_TILED: &str = "matmul_vec_tiled_kernel";
    pub const ELEMENTWISE_MUL: &str = "elementwise_mul_kernel";
    pub const ELEMENTWISE_ADD: &str = "elementwise_add_kernel";
    pub const FUSED_SILU_MUL: &str = "fused_silu_mul_kernel";
    pub const MATMUL: &str = "matmul_kernel";
}
