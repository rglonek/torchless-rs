//! Additional ROCm Kernel Utilities
//!
//! This module provides helper functions for launching custom HIP kernels
//! and additional kernel implementations for advanced operations.

// =============================================================================
// Kernel Launch Configuration
// =============================================================================

/// Launch configuration for HIP kernels.
#[derive(Debug, Clone, Copy)]
pub struct HipLaunchConfig {
    /// Block dimensions (threads per block)
    pub block_dim: (u32, u32, u32),
    /// Grid dimensions (number of blocks)
    pub grid_dim: (u32, u32, u32),
    /// Shared memory bytes per block
    pub shared_mem_bytes: u32,
}

impl Default for HipLaunchConfig {
    fn default() -> Self {
        Self {
            block_dim: (256, 1, 1),
            grid_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

// =============================================================================
// Kernel Launch Helpers
// =============================================================================

/// Calculate optimal block size for a kernel.
///
/// Uses a simple heuristic: 256 threads per block is generally good
/// for compute-bound kernels on modern AMD GPUs.
pub fn optimal_block_size(n_elements: usize) -> u32 {
    // AMD GPUs typically work well with 64-256 threads per block
    // Wavefront size is 64 on AMD (vs 32 on NVIDIA)
    if n_elements < 64 {
        64 // Minimum one wavefront
    } else if n_elements < 256 {
        64
    } else if n_elements < 1024 {
        128
    } else {
        256
    }
}

/// Calculate number of blocks needed to process n elements.
pub fn num_blocks(n_elements: usize, block_size: u32) -> u32 {
    n_elements.div_ceil(block_size as usize) as u32
}

/// Create a launch configuration for a 1D kernel.
pub fn launch_config_1d(n_elements: usize) -> HipLaunchConfig {
    let block_size = optimal_block_size(n_elements);
    let num_blocks = num_blocks(n_elements, block_size);

    HipLaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (num_blocks, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Create a launch configuration for a 1D kernel with shared memory.
pub fn launch_config_1d_shared(n_elements: usize, shared_bytes_per_thread: u32) -> HipLaunchConfig {
    let block_size = optimal_block_size(n_elements);
    let num_blocks = num_blocks(n_elements, block_size);
    let shared_mem = block_size * shared_bytes_per_thread;

    HipLaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (num_blocks, 1, 1),
        shared_mem_bytes: shared_mem,
    }
}

/// Create a launch configuration for a 2D kernel.
pub fn launch_config_2d(rows: usize, cols: usize) -> HipLaunchConfig {
    // Use 16x16 thread blocks for 2D kernels
    let block_x = 16u32;
    let block_y = 16u32;

    let grid_x = cols.div_ceil(block_x as usize) as u32;
    let grid_y = rows.div_ceil(block_y as usize) as u32;

    HipLaunchConfig {
        block_dim: (block_x, block_y, 1),
        grid_dim: (grid_x, grid_y, 1),
        shared_mem_bytes: 0,
    }
}

// =============================================================================
// HIP Kernel Source Code
// =============================================================================

/// HIP kernel source code for core operations.
///
/// HIP uses CUDA-like syntax and is mostly source-compatible with CUDA.
/// These kernels can be compiled with hipcc (HIP compiler).
pub const HIP_KERNELS_SOURCE: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// RMSNorm kernel
// x = x * weight / sqrt(mean(x^2) + eps)
extern "C" __global__ void rmsnorm_kernel(float* x, const float* weight, int n, float eps) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Compute sum of squares in parallel
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float val = x[i];
        local_sum += val * val;
    }
    
    // Store in shared memory
    shared[tid] = local_sum;
    __syncthreads();
    
    // Reduction to compute total sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute RMS
    float rms = sqrtf(shared[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;
    __syncthreads();
    
    // Apply normalization and weight
    for (int i = tid; i < n; i += block_size) {
        x[i] = x[i] * inv_rms * weight[i];
    }
}

// Softmax kernel with numerical stability
extern "C" __global__ void softmax_kernel(float* x, int n) {
    extern __shared__ float shared[];
    float* max_shared = shared;
    float* sum_shared = &shared[blockDim.x];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Find max in parallel
    float local_max = -INFINITY;
    for (int i = tid; i < n; i += block_size) {
        local_max = fmaxf(local_max, x[i]);
    }
    max_shared[tid] = local_max;
    __syncthreads();
    
    // Reduction for max
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_shared[tid] = fmaxf(max_shared[tid], max_shared[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = max_shared[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float exp_val = expf(x[i] - max_val);
        x[i] = exp_val;
        local_sum += exp_val;
    }
    sum_shared[tid] = local_sum;
    __syncthreads();
    
    // Reduction for sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_shared[tid] += sum_shared[tid + stride];
        }
        __syncthreads();
    }
    float sum = sum_shared[0];
    float inv_sum = 1.0f / sum;
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < n; i += block_size) {
        x[i] *= inv_sum;
    }
}

// SiLU activation kernel: y = x * sigmoid(x) = x / (1 + exp(-x))
extern "C" __global__ void silu_kernel(const float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = val / (1.0f + expf(-val));
    }
}

// RoPE (Rotary Position Embedding) kernel
// Uses half-split layout: rotate dim i with dim i + head_dim/2
extern "C" __global__ void rope_kernel(
    float* x,           // [n_heads, head_dim]
    const float* cos,   // [half]
    const float* sin,   // [half]
    int n_heads,
    int head_dim,
    int half
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * half;
    
    if (idx < total) {
        int h = idx / half;      // head index
        int i = idx % half;      // position within half
        
        int base = h * head_dim;
        float xi = x[base + i];
        float yi = x[base + i + half];
        float c = cos[i];
        float s = sin[i];
        
        // Rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
        x[base + i] = xi * c - yi * s;
        x[base + i + half] = xi * s + yi * c;
    }
}

// Attention scores kernel: scores[i] = keys[i, :].dot(query) * scale
extern "C" __global__ void attention_scores_kernel(
    const float* query,   // [head_dim]
    const float* keys,    // [seq_len, head_dim]
    float* scores,        // [seq_len]
    int seq_len,
    int head_dim,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < seq_len) {
        float dot = 0.0f;
        const float* key_row = keys + i * head_dim;
        for (int j = 0; j < head_dim; j++) {
            dot += query[j] * key_row[j];
        }
        scores[i] = dot * scale;
    }
}

// Weighted sum kernel: out[j] = sum_i(weights[i] * matrix[i, j])
extern "C" __global__ void weighted_sum_kernel(
    const float* weights,  // [n]
    const float* matrix,   // [n, d]
    float* out,            // [d]
    int n,
    int d
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j < d) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += weights[i] * matrix[i * d + j];
        }
        out[j] = sum;
    }
}

// Element-wise multiply kernel
extern "C" __global__ void elementwise_mul_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Element-wise add kernel
extern "C" __global__ void elementwise_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Matrix-vector multiplication kernel: y = W @ x
// W: [rows, cols], x: [cols], y: [rows]
extern "C" __global__ void matmul_vec_kernel(
    const float* W,
    const float* x,
    float* y,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        const float* W_row = W + row * cols;
        for (int j = 0; j < cols; j++) {
            sum += W_row[j] * x[j];
        }
        y[row] = sum;
    }
}
"#;

/// FP16 kernel source for memory-efficient operations.
pub const FP16_KERNELS_SOURCE: &str = r#"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Convert FP16 to FP32 kernel
extern "C" __global__ void fp16_to_fp32_kernel(
    const __half* input,
    float* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

// Convert FP32 to FP16 kernel
extern "C" __global__ void fp32_to_fp16_kernel(
    const float* input,
    __half* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

// FP16 SiLU kernel
extern "C" __global__ void silu_fp16_kernel(
    const __half* x,
    __half* y,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(x[idx]);
        float result = val / (1.0f + expf(-val));
        y[idx] = __float2half(result);
    }
}

// FP16 RMSNorm kernel
extern "C" __global__ void rmsnorm_fp16_kernel(
    __half* x,
    const __half* weight,
    int n,
    float eps
) {
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Compute sum of squares (in FP32 for precision)
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += block_size) {
        float val = __half2float(x[i]);
        local_sum += val * val;
    }
    
    shared[tid] = local_sum;
    __syncthreads();
    
    // Reduction
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    float rms = sqrtf(shared[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;
    __syncthreads();
    
    // Apply normalization
    for (int i = tid; i < n; i += block_size) {
        float val = __half2float(x[i]) * inv_rms * __half2float(weight[i]);
        x[i] = __float2half(val);
    }
}
"#;

/// Quantized kernel source for INT4/INT8 operations.
pub const QUANTIZED_KERNELS_SOURCE: &str = r#"
#include <hip/hip_runtime.h>

// Fused INT8 dequantize + matmul kernel
extern "C" __global__ void int8_matmul_kernel(
    const char* weights,   // [rows, cols] in INT8
    const float* scales,   // [rows] scale per row
    const float* input,    // [cols]
    float* output,         // [rows]
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        float scale = scales[row];
        const char* weight_row = weights + row * cols;
        
        for (int j = 0; j < cols; j++) {
            float w = (float)weight_row[j] * scale;
            sum += w * input[j];
        }
        
        output[row] = sum;
    }
}

// Fused INT4 dequantize + matmul kernel (packed format)
extern "C" __global__ void int4_matmul_kernel(
    const unsigned char* weights,  // [rows, cols/2] packed INT4
    const float* scales,           // [rows * (cols / group_size)] scales
    const float* input,            // [cols]
    float* output,                 // [rows]
    int rows,
    int cols,
    int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        int packed_cols = cols / 2;
        const unsigned char* weight_row = weights + row * packed_cols;
        int groups_per_row = cols / group_size;
        
        for (int j = 0; j < cols; j++) {
            // Unpack INT4 value
            int packed_idx = j / 2;
            unsigned char packed = weight_row[packed_idx];
            signed char q_val;
            if (j % 2 == 0) {
                q_val = (signed char)(packed & 0x0F) - 8;  // Lower nibble
            } else {
                q_val = (signed char)(packed >> 4) - 8;   // Upper nibble
            }
            
            // Get scale for this group
            int group_idx = j / group_size;
            float scale = scales[row * groups_per_row + group_idx];
            
            // Dequantize and accumulate
            float w = (float)q_val * scale;
            sum += w * input[j];
        }
        
        output[row] = sum;
    }
}
"#;

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_block_size() {
        assert_eq!(optimal_block_size(10), 64);
        assert_eq!(optimal_block_size(100), 64);
        assert_eq!(optimal_block_size(500), 128);
        assert_eq!(optimal_block_size(10000), 256);
    }

    #[test]
    fn test_num_blocks() {
        assert_eq!(num_blocks(100, 64), 2);
        assert_eq!(num_blocks(256, 256), 1);
        assert_eq!(num_blocks(257, 256), 2);
        assert_eq!(num_blocks(1000, 256), 4);
    }

    #[test]
    fn test_launch_config_1d() {
        let cfg = launch_config_1d(1000);
        assert_eq!(cfg.block_dim.0, 256);
        assert_eq!(cfg.grid_dim.0, 4);
        assert_eq!(cfg.shared_mem_bytes, 0);
    }

    #[test]
    fn test_launch_config_2d() {
        let cfg = launch_config_2d(100, 200);
        assert_eq!(cfg.block_dim.0, 16);
        assert_eq!(cfg.block_dim.1, 16);
        // grid_x = ceil(200/16) = 13
        // grid_y = ceil(100/16) = 7
        assert_eq!(cfg.grid_dim.0, 13);
        assert_eq!(cfg.grid_dim.1, 7);
    }
}
