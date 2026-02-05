//! OpenCL Kernel Sources
//!
//! This module contains the OpenCL C source code for GPU compute kernels
//! used in transformer inference.
//!
//! # Kernel Overview
//!
//! - `rmsnorm_kernel`: RMSNorm normalization with parallel reduction
//! - `softmax_kernel`: Numerically stable softmax with parallel max/sum
//! - `silu_kernel`: SiLU (Swish) activation function
//! - `rope_kernel`: Rotary Position Embedding
//! - `attention_scores_kernel`: Compute Q @ K^T attention scores
//! - `weighted_sum_kernel`: Compute attention-weighted sum of values
//! - `matmul_vec_kernel`: Matrix-vector multiplication

/// OpenCL C source code for all compute kernels.
pub const OPENCL_KERNELS_SOURCE: &str = r#"
// =============================================================================
// Utility Macros
// =============================================================================

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// =============================================================================
// RMSNorm Kernel
// =============================================================================
// Computes: x = x * weight / sqrt(mean(x^2) + eps)
// Uses local memory for parallel reduction of sum of squares.
//
// Work-group setup:
// - Global: 1 work-group
// - Local: block_size threads (should be power of 2, e.g., 256)

__kernel void rmsnorm_kernel(
    __global float* x,
    __global const float* weight,
    const int n,
    const float eps,
    __local float* shared
) {
    int tid = get_local_id(0);
    int block_size = get_local_size(0);
    
    // Compute sum of squares in parallel
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float val = x[i];
        local_sum += val * val;
    }
    
    // Store in local memory
    shared[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction for sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Compute RMS
    float rms = sqrt(shared[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;
    barrier(CLK_LOCAL_MEM_FENCE);
    
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
//
// Work-group setup:
// - Global: 1 work-group  
// - Local: block_size threads (should be power of 2)
// - Local memory: 2 * block_size * sizeof(float)

__kernel void softmax_kernel(
    __global float* x,
    const int n,
    __local float* shared
) {
    int tid = get_local_id(0);
    int block_size = get_local_size(0);
    
    // Shared memory layout: first half for max, second half for sum
    __local float* max_shared = shared;
    __local float* sum_shared = shared + block_size;
    
    // Find max in parallel
    float local_max = -INFINITY;
    for (int i = tid; i < n; i += block_size) {
        local_max = fmax(local_max, x[i]);
    }
    max_shared[tid] = local_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction for max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_shared[tid] = fmax(max_shared[tid], max_shared[tid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = max_shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float exp_val = exp(x[i] - max_val);
        x[i] = exp_val;
        local_sum += exp_val;
    }
    sum_shared[tid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Parallel reduction for sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sum = sum_shared[0];
    float inv_sum = 1.0f / sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Normalize
    for (int i = tid; i < n; i += block_size) {
        x[i] *= inv_sum;
    }
}

// =============================================================================
// SiLU Activation Kernel
// =============================================================================
// Computes: y = x * sigmoid(x) = x / (1 + exp(-x))
//
// Work-group setup:
// - Global: n threads (or more, kernel handles bounds)
// - Local: any reasonable size (e.g., 256)

__kernel void silu_kernel(
    __global const float* x,
    __global float* y,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + exp(-val));
    }
}

// =============================================================================
// RoPE (Rotary Position Embedding) Kernel
// =============================================================================
// Uses half-split layout: rotate dim i with dim i + head_dim/2
// x: [n_heads, head_dim], cos/sin: [half_dim]
//
// Work-group setup:
// - Global: n_heads * half_dim threads
// - Local: any reasonable size

__kernel void rope_kernel(
    __global float* x,
    __global const float* cos_table,
    __global const float* sin_table,
    const int n_heads,
    const int head_dim,
    const int half_dim
) {
    int idx = get_global_id(0);
    int total = n_heads * half_dim;
    
    if (idx < total) {
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
//
// Work-group setup:
// - Global: seq_len threads
// - Local: any reasonable size

__kernel void attention_scores_kernel(
    __global const float* query,
    __global const float* keys,
    __global float* scores,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    int i = get_global_id(0);
    if (i < seq_len) {
        float dot = 0.0f;
        __global const float* key_row = keys + i * head_dim;
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
//
// Work-group setup:
// - Global: d threads
// - Local: any reasonable size

__kernel void weighted_sum_kernel(
    __global const float* weights,
    __global const float* matrix,
    __global float* out,
    const int n,
    const int d
) {
    int j = get_global_id(0);
    if (j < d) {
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
// Each work-item computes one output element.
//
// Work-group setup:
// - Global: rows threads
// - Local: any reasonable size (e.g., 256)

__kernel void matmul_vec_kernel(
    __global const float* weights,
    __global const float* x,
    __global float* out,
    const int rows,
    const int cols
) {
    int row = get_global_id(0);
    if (row < rows) {
        float sum = 0.0f;
        __global const float* w_row = weights + row * cols;
        for (int j = 0; j < cols; j++) {
            sum += w_row[j] * x[j];
        }
        out[row] = sum;
    }
}

// =============================================================================
// Tiled Matrix-Vector Multiplication Kernel
// =============================================================================
// More efficient implementation using local memory for shared x access.
// Uses TILE_SIZE work-items per work-group.
//
// Work-group setup:
// - Global: rows threads
// - Local: TILE_SIZE (256)

#define TILE_SIZE 256

__kernel void matmul_vec_tiled_kernel(
    __global const float* weights,
    __global const float* x,
    __global float* out,
    const int rows,
    const int cols,
    __local float* shared_x
) {
    int row = get_global_id(0);
    int tid = get_local_id(0);
    int block_size = get_local_size(0);
    
    if (row >= rows) return;
    
    float sum = 0.0f;
    __global const float* w_row = weights + row * cols;
    
    // Process in tiles to improve cache utilization
    for (int tile_start = 0; tile_start < cols; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, cols);
        int tile_len = tile_end - tile_start;
        
        // Each work-item loads its portion of x
        for (int i = tid; i < tile_len; i += block_size) {
            shared_x[i] = x[tile_start + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial dot product using shared x
        for (int j = 0; j < tile_len; j++) {
            sum += w_row[tile_start + j] * shared_x[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    out[row] = sum;
}

// =============================================================================
// Element-wise Operations
// =============================================================================

__kernel void elementwise_mul_kernel(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__kernel void elementwise_add_kernel(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// =============================================================================
// Fused Operations (for better performance)
// =============================================================================

// Fused SiLU + Element-wise Multiply
// Computes: out = silu(gate) * up
// Used in SwiGLU activation: gate_proj and up_proj outputs
__kernel void fused_silu_mul_kernel(
    __global const float* gate,
    __global const float* up,
    __global float* out,
    const int n
) {
    int idx = get_global_id(0);
    if (idx < n) {
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
// Note: For production, use vendor-specific BLAS libraries (clBLAS, etc.)

__kernel void matmul_kernel(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K
) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"#;

/// Kernel names for lookup in compiled program
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
