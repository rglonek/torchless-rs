//! Additional CUDA Kernel Utilities
//!
//! This module provides helper functions for launching custom CUDA kernels
//! and additional kernel implementations for advanced operations.

use cudarc::driver::LaunchConfig;

// =============================================================================
// Kernel Launch Helpers
// =============================================================================

/// Calculate optimal block size for a kernel.
///
/// Uses a simple heuristic: 256 threads per block is generally good
/// for compute-bound kernels on modern GPUs.
pub fn optimal_block_size(n_elements: usize) -> u32 {
    // Common block sizes: 64, 128, 256, 512, 1024
    // 256 is a good default for most compute-bound kernels
    if n_elements < 64 {
        32
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
pub fn launch_config_1d(n_elements: usize) -> LaunchConfig {
    let block_size = optimal_block_size(n_elements);
    let num_blocks = num_blocks(n_elements, block_size);

    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (num_blocks, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Create a launch configuration for a 1D kernel with shared memory.
pub fn launch_config_1d_shared(n_elements: usize, shared_bytes_per_thread: u32) -> LaunchConfig {
    let block_size = optimal_block_size(n_elements);
    let num_blocks = num_blocks(n_elements, block_size);
    let shared_mem = block_size * shared_bytes_per_thread;

    LaunchConfig {
        block_dim: (block_size, 1, 1),
        grid_dim: (num_blocks, 1, 1),
        shared_mem_bytes: shared_mem,
    }
}

/// Create a launch configuration for a 2D kernel.
pub fn launch_config_2d(rows: usize, cols: usize) -> LaunchConfig {
    // Use 16x16 thread blocks for 2D kernels
    let block_x = 16u32;
    let block_y = 16u32;

    let grid_x = cols.div_ceil(block_x as usize) as u32;
    let grid_y = rows.div_ceil(block_y as usize) as u32;

    LaunchConfig {
        block_dim: (block_x, block_y, 1),
        grid_dim: (grid_x, grid_y, 1),
        shared_mem_bytes: 0,
    }
}

// =============================================================================
// FP16 Kernel Support (for future half-precision operations)
// =============================================================================

/// Additional CUDA kernel source for FP16 operations.
///
/// These kernels use CUDA's half-precision types for memory-efficient
/// operations. They are compiled separately when FP16 support is needed.
pub const FP16_KERNELS_SOURCE: &str = r#"
#include <cuda_fp16.h>

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

// FP16 SiLU kernel (for memory-efficient activation)
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

// =============================================================================
// Quantized Kernel Support (for future INT4/INT8 operations)
// =============================================================================

/// Additional CUDA kernel source for quantized operations.
///
/// These kernels support INT4 and INT8 quantized weights with fused
/// dequantization during matrix multiplication.
pub const QUANTIZED_KERNELS_SOURCE: &str = r#"
// Fused INT8 dequantize + matmul kernel
// Dequantizes INT8 weights on-the-fly during matrix-vector multiply
extern "C" __global__ void int8_matmul_kernel(
    const int8_t* weights,  // [rows, cols] in INT8
    const float* scales,    // [rows] scale per row
    const float* input,     // [cols]
    float* output,          // [rows]
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        float scale = scales[row];
        const int8_t* weight_row = weights + row * cols;
        
        for (int j = 0; j < cols; j++) {
            float w = (float)weight_row[j] * scale;
            sum += w * input[j];
        }
        
        output[row] = sum;
    }
}

// Fused INT4 dequantize + matmul kernel (packed format)
// Two INT4 values per byte, lower nibble first
extern "C" __global__ void int4_matmul_kernel(
    const uint8_t* weights,  // [rows, cols/2] packed INT4
    const float* scales,     // [rows * (cols / group_size)] scales
    const float* input,      // [cols]
    float* output,           // [rows]
    int rows,
    int cols,
    int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        float sum = 0.0f;
        int packed_cols = cols / 2;
        const uint8_t* weight_row = weights + row * packed_cols;
        int groups_per_row = cols / group_size;
        
        for (int j = 0; j < cols; j++) {
            // Unpack INT4 value
            int packed_idx = j / 2;
            uint8_t packed = weight_row[packed_idx];
            int8_t q_val;
            if (j % 2 == 0) {
                q_val = (int8_t)(packed & 0x0F) - 8;  // Lower nibble
            } else {
                q_val = (int8_t)(packed >> 4) - 8;   // Upper nibble
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

// Q4_0 block dequantization kernel (GGUF format compatible)
// Q4_0 block: 2 bytes scale (f16) + 16 bytes weights (32 x 4-bit)
extern "C" __global__ void q4_0_dequantize_kernel(
    const uint8_t* blocks,    // Packed Q4_0 blocks
    float* output,            // [total_elements]
    int num_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (block_idx < num_blocks) {
        // Each Q4_0 block is 18 bytes: 2 bytes scale + 16 bytes data
        const uint8_t* block = blocks + block_idx * 18;
        
        // Read scale (stored as f16, interpret bytes)
        uint16_t scale_bits = block[0] | (block[1] << 8);
        // Simple f16 to f32 conversion (ignoring denormals/inf/nan)
        int sign = (scale_bits >> 15) & 1;
        int exp = (scale_bits >> 10) & 0x1F;
        int mant = scale_bits & 0x3FF;
        float scale;
        if (exp == 0) {
            scale = 0.0f;  // Simplified: treat denormals as 0
        } else {
            scale = (sign ? -1.0f : 1.0f) * ldexpf(1.0f + mant / 1024.0f, exp - 15);
        }
        
        // Dequantize 32 values
        float* out = output + block_idx * 32;
        for (int i = 0; i < 16; i++) {
            uint8_t packed = block[2 + i];
            int8_t q0 = (int8_t)(packed & 0x0F) - 8;
            int8_t q1 = (int8_t)(packed >> 4) - 8;
            out[i * 2] = q0 * scale;
            out[i * 2 + 1] = q1 * scale;
        }
    }
}
"#;

// =============================================================================
// Flash Attention Kernel Support
// =============================================================================

/// CUDA kernel source for flash attention.
///
/// Flash attention computes attention with O(N) memory instead of O(N^2)
/// by using tiled computation with online softmax.
pub const FLASH_ATTENTION_KERNELS_SOURCE: &str = r#"
// Constants for flash attention
#define TILE_SIZE 64

// Flash attention kernel (simplified version for single head)
// Computes: softmax(Q @ K^T / sqrt(d)) @ V
extern "C" __global__ void flash_attention_kernel(
    const float* Q,       // [seq_len, head_dim]
    const float* K,       // [seq_len, head_dim]
    const float* V,       // [seq_len, head_dim]
    float* output,        // [seq_len, head_dim]
    int seq_len,
    int head_dim,
    float scale
) {
    // Shared memory for tiles
    extern __shared__ float shared_mem[];
    float* K_tile = shared_mem;                     // [TILE_SIZE, head_dim]
    float* V_tile = K_tile + TILE_SIZE * head_dim;  // [TILE_SIZE, head_dim]
    
    int query_idx = blockIdx.x;  // Which query row we're processing
    int tid = threadIdx.x;
    
    if (query_idx >= seq_len) return;
    
    // Load query row into registers
    float q[128];  // Assuming head_dim <= 128
    for (int d = 0; d < head_dim; d++) {
        q[d] = Q[query_idx * head_dim + d];
    }
    
    // Running statistics for online softmax
    float m_prev = -INFINITY;  // Max so far
    float l_prev = 0.0f;       // Sum of exp so far
    float o[128] = {0};        // Accumulated output
    
    // Process K/V in tiles
    int num_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_SIZE;
        int tile_end = min(tile_start + TILE_SIZE, seq_len);
        int tile_len = tile_end - tile_start;
        
        // Collaboratively load K and V tiles into shared memory
        for (int i = tid; i < tile_len * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int global_row = tile_start + row;
            K_tile[row * head_dim + col] = K[global_row * head_dim + col];
            V_tile[row * head_dim + col] = V[global_row * head_dim + col];
        }
        __syncthreads();
        
        // Compute attention scores for this tile
        float scores[TILE_SIZE];
        float m_new = m_prev;
        
        for (int j = 0; j < tile_len; j++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q[d] * K_tile[j * head_dim + d];
            }
            score *= scale;
            scores[j] = score;
            m_new = fmaxf(m_new, score);
        }
        
        // Online softmax update
        float l_new = l_prev * expf(m_prev - m_new);
        
        for (int j = 0; j < tile_len; j++) {
            float p = expf(scores[j] - m_new);
            l_new += p;
            
            // Update output
            for (int d = 0; d < head_dim; d++) {
                o[d] = o[d] * expf(m_prev - m_new) + p * V_tile[j * head_dim + d];
            }
        }
        
        m_prev = m_new;
        l_prev = l_new;
        
        __syncthreads();
    }
    
    // Write final output (normalized by softmax denominator)
    float inv_l = 1.0f / l_prev;
    for (int d = 0; d < head_dim; d++) {
        output[query_idx * head_dim + d] = o[d] * inv_l;
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
        assert_eq!(optimal_block_size(10), 32);
        assert_eq!(optimal_block_size(100), 64);
        assert_eq!(optimal_block_size(500), 128);
        assert_eq!(optimal_block_size(10000), 256);
    }

    #[test]
    fn test_num_blocks() {
        assert_eq!(num_blocks(100, 32), 4);
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
