use super::quantization::{Q4KMBlock, Q4KSBlock, Q4_0Block, Q8_0Block, QK4_0, QK8_0, QK_K};
use super::{Config, Header, TensorInfo};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

/// Tensor data type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum TensorDtype {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 8-bit integer with group quantization (legacy format)
    Int8,
    /// 8-bit quantization (GGUF Q8_0 compatible)
    Q8_0,
    /// 4-bit quantization (GGUF Q4_0 compatible)
    Q4_0,
    /// 4-bit K-quantization medium (GGUF Q4_K_M compatible)
    Q4_K_M,
    /// 4-bit K-quantization small (GGUF Q4_K_S compatible)
    Q4_K_S,
}

#[allow(clippy::should_implement_trait)]
impl TensorDtype {
    /// Parse dtype from string representation
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "float32" => Some(TensorDtype::F32),
            "f16" | "float16" => Some(TensorDtype::F16),
            "bf16" | "bfloat16" => Some(TensorDtype::BF16),
            "int8" => Some(TensorDtype::Int8),
            "q8_0" => Some(TensorDtype::Q8_0),
            "q4_0" => Some(TensorDtype::Q4_0),
            "q4_k_m" | "q4_k" => Some(TensorDtype::Q4_K_M),
            "q4_k_s" => Some(TensorDtype::Q4_K_S),
            _ => None,
        }
    }

    /// Get block size for quantized formats
    pub fn block_size(&self) -> usize {
        match self {
            TensorDtype::F32 | TensorDtype::F16 | TensorDtype::BF16 => 1,
            TensorDtype::Int8 => 64, // Legacy format uses 64 elements per group
            TensorDtype::Q8_0 => QK8_0,
            TensorDtype::Q4_0 => QK4_0,
            TensorDtype::Q4_K_M | TensorDtype::Q4_K_S => QK_K,
        }
    }

    /// Get bytes per block for this dtype
    pub fn bytes_per_block(&self) -> usize {
        match self {
            TensorDtype::F32 => 4,
            TensorDtype::F16 | TensorDtype::BF16 => 2,
            TensorDtype::Int8 => 1,
            TensorDtype::Q8_0 => Q8_0Block::SIZE,
            TensorDtype::Q4_0 => Q4_0Block::SIZE,
            TensorDtype::Q4_K_M => Q4KMBlock::SIZE,
            TensorDtype::Q4_K_S => Q4KSBlock::SIZE,
        }
    }

    /// Returns true if this dtype is quantized
    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            TensorDtype::F32 | TensorDtype::F16 | TensorDtype::BF16
        )
    }
}

/// A lazy view into a tensor stored in the memory-mapped file.
/// This struct holds references to the raw data without copying or dequantizing
/// until the data is actually needed.
#[derive(Debug)]
pub struct TensorView<'a> {
    /// Raw data bytes (varies by dtype)
    pub data: &'a [u8],
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: TensorDtype,
    /// Quantization scales (only for legacy int8 tensors)
    pub scales: Option<&'a [u8]>,
    /// Number of elements per quantization group (for legacy int8, typically 64)
    pub group_size: usize,
}

impl TensorDtype {
    /// Get the data size in bytes for a tensor with given number of elements
    pub fn data_bytes(&self, numel: usize) -> usize {
        match self {
            TensorDtype::F32 => numel * 4,
            TensorDtype::F16 | TensorDtype::BF16 => numel * 2,
            TensorDtype::Int8 => numel, // 1 byte per element, scales stored separately
            TensorDtype::Q8_0 => numel.div_ceil(QK8_0) * Q8_0Block::SIZE,
            TensorDtype::Q4_0 => numel.div_ceil(QK4_0) * Q4_0Block::SIZE,
            TensorDtype::Q4_K_M => numel.div_ceil(QK_K) * Q4KMBlock::SIZE,
            TensorDtype::Q4_K_S => numel.div_ceil(QK_K) * Q4KSBlock::SIZE,
        }
    }
}

impl TensorView<'_> {
    /// Get the number of rows in the tensor.
    /// For 1D tensors, treats them as a single row (nrows = 1).
    /// For 2D tensors, returns the first dimension.
    pub fn nrows(&self) -> usize {
        if self.shape.len() == 1 {
            1 // 1D tensor is treated as single row
        } else {
            self.shape[0]
        }
    }

    /// Get the number of columns in the tensor (row length).
    /// For 1D tensors, returns the total number of elements.
    /// For 2D tensors, returns the second dimension.
    pub fn ncols(&self) -> usize {
        if self.shape.len() == 1 {
            self.shape[0] // 1D tensor: all elements are in one row
        } else {
            self.shape[1]
        }
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Dequantize a single row and return as Vec<f32>
    /// For f32/f16 tensors, this converts to f32.
    /// For quantized tensors, this dequantizes on-the-fly.
    pub fn get_row(&self, row: usize) -> Vec<f32> {
        let ncols = self.ncols();
        let row_start = row * ncols;

        match self.dtype {
            TensorDtype::F32 => {
                // Read f32 values directly from bytes
                let byte_start = row_start * 4;
                let byte_end = byte_start + ncols * 4;
                let row_bytes = &self.data[byte_start..byte_end];
                let mut result = Vec::with_capacity(ncols);
                let mut cursor = Cursor::new(row_bytes);
                for _ in 0..ncols {
                    result.push(cursor.read_f32::<LittleEndian>().unwrap());
                }
                result
            }
            TensorDtype::F16 => {
                // Read f16 values and convert to f32
                let byte_start = row_start * 2;
                let byte_end = byte_start + ncols * 2;
                let row_bytes = &self.data[byte_start..byte_end];
                let mut result = Vec::with_capacity(ncols);
                for i in 0..ncols {
                    let f16_val = f16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
                    result.push(f16_val.to_f32());
                }
                result
            }
            TensorDtype::BF16 => {
                // Read bf16 values and convert to f32
                let byte_start = row_start * 2;
                let byte_end = byte_start + ncols * 2;
                let row_bytes = &self.data[byte_start..byte_end];
                let mut result = Vec::with_capacity(ncols);
                for i in 0..ncols {
                    let bits = u16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
                    result.push(f32::from_bits((bits as u32) << 16));
                }
                result
            }
            TensorDtype::Int8 => {
                // Dequantize int8 values using scales (legacy format)
                let scales_bytes = self.scales.expect("Int8 tensor must have scales");
                let row_bytes = &self.data[row_start..row_start + ncols];
                let mut result = Vec::with_capacity(ncols);

                for (i, &q_val) in row_bytes.iter().enumerate() {
                    let global_idx = row_start + i;
                    let group_idx = global_idx / self.group_size;
                    // Read scale as f32 from bytes
                    let scale_byte_offset = group_idx * 4;
                    let scale = f32::from_le_bytes([
                        scales_bytes[scale_byte_offset],
                        scales_bytes[scale_byte_offset + 1],
                        scales_bytes[scale_byte_offset + 2],
                        scales_bytes[scale_byte_offset + 3],
                    ]);
                    result.push(q_val as i8 as f32 / scale);
                }
                result
            }
            TensorDtype::Q8_0 => {
                // Dequantize Q8_0 blocks
                let mut result = Vec::with_capacity(ncols);
                let n_blocks = ncols.div_ceil(QK8_0);
                let block_offset = row * n_blocks * Q8_0Block::SIZE;

                for b in 0..n_blocks {
                    let block_start = block_offset + b * Q8_0Block::SIZE;
                    let block_end = block_start + Q8_0Block::SIZE;
                    let block = Q8_0Block::from_bytes(&self.data[block_start..block_end]);
                    let values = block.dequantize_block();
                    let take = (ncols - result.len()).min(QK8_0);
                    result.extend_from_slice(&values[..take]);
                }
                result
            }
            TensorDtype::Q4_0 => {
                // Dequantize Q4_0 blocks
                let mut result = Vec::with_capacity(ncols);
                let n_blocks = ncols.div_ceil(QK4_0);
                let block_offset = row * n_blocks * Q4_0Block::SIZE;

                for b in 0..n_blocks {
                    let block_start = block_offset + b * Q4_0Block::SIZE;
                    let block_end = block_start + Q4_0Block::SIZE;
                    let block = Q4_0Block::from_bytes(&self.data[block_start..block_end]);
                    let values = block.dequantize_block();
                    let take = (ncols - result.len()).min(QK4_0);
                    result.extend_from_slice(&values[..take]);
                }
                result
            }
            TensorDtype::Q4_K_M => {
                // Dequantize Q4_K_M blocks
                let mut result = Vec::with_capacity(ncols);
                let n_blocks = ncols.div_ceil(QK_K);
                let block_offset = row * n_blocks * Q4KMBlock::SIZE;

                for b in 0..n_blocks {
                    let block_start = block_offset + b * Q4KMBlock::SIZE;
                    let block_end = block_start + Q4KMBlock::SIZE;
                    let block = Q4KMBlock::from_bytes(&self.data[block_start..block_end]);
                    let values = block.dequantize_block();
                    let take = (ncols - result.len()).min(QK_K);
                    result.extend_from_slice(&values[..take]);
                }
                result
            }
            TensorDtype::Q4_K_S => {
                // Dequantize Q4_K_S blocks
                let mut result = Vec::with_capacity(ncols);
                let n_blocks = ncols.div_ceil(QK_K);
                let block_offset = row * n_blocks * Q4KSBlock::SIZE;

                for b in 0..n_blocks {
                    let block_start = block_offset + b * Q4KSBlock::SIZE;
                    let block_end = block_start + Q4KSBlock::SIZE;
                    let block = Q4KSBlock::from_bytes(&self.data[block_start..block_end]);
                    let values = block.dequantize_block();
                    let take = (ncols - result.len()).min(QK_K);
                    result.extend_from_slice(&values[..take]);
                }
                result
            }
        }
    }

    /// Get a single element from the tensor (for embedding lookup)
    /// row: row index, col: column index within the row
    pub fn get_element(&self, row: usize, col: usize) -> f32 {
        let ncols = self.ncols();
        let idx = row * ncols + col;

        match self.dtype {
            TensorDtype::F32 => {
                let byte_offset = idx * 4;
                f32::from_le_bytes([
                    self.data[byte_offset],
                    self.data[byte_offset + 1],
                    self.data[byte_offset + 2],
                    self.data[byte_offset + 3],
                ])
            }
            TensorDtype::F16 => {
                let byte_offset = idx * 2;
                let f16_val =
                    f16::from_le_bytes([self.data[byte_offset], self.data[byte_offset + 1]]);
                f16_val.to_f32()
            }
            TensorDtype::BF16 => {
                let byte_offset = idx * 2;
                let bits = u16::from_le_bytes([self.data[byte_offset], self.data[byte_offset + 1]]);
                f32::from_bits((bits as u32) << 16)
            }
            TensorDtype::Int8 => {
                let scales_bytes = self.scales.expect("Int8 tensor must have scales");
                let q_val = self.data[idx] as i8;
                let group_idx = idx / self.group_size;
                let scale_byte_offset = group_idx * 4;
                let scale = f32::from_le_bytes([
                    scales_bytes[scale_byte_offset],
                    scales_bytes[scale_byte_offset + 1],
                    scales_bytes[scale_byte_offset + 2],
                    scales_bytes[scale_byte_offset + 3],
                ]);
                q_val as f32 / scale
            }
            TensorDtype::Q8_0 => {
                let block_idx = idx / QK8_0;
                let within_block = idx % QK8_0;
                let block_start = block_idx * Q8_0Block::SIZE;
                let block_end = block_start + Q8_0Block::SIZE;
                let block = Q8_0Block::from_bytes(&self.data[block_start..block_end]);
                block.dequantize(within_block)
            }
            TensorDtype::Q4_0 => {
                let block_idx = idx / QK4_0;
                let within_block = idx % QK4_0;
                let block_start = block_idx * Q4_0Block::SIZE;
                let block_end = block_start + Q4_0Block::SIZE;
                let block = Q4_0Block::from_bytes(&self.data[block_start..block_end]);
                block.dequantize(within_block)
            }
            TensorDtype::Q4_K_M => {
                let block_idx = idx / QK_K;
                let within_block = idx % QK_K;
                let block_start = block_idx * Q4KMBlock::SIZE;
                let block_end = block_start + Q4KMBlock::SIZE;
                let block = Q4KMBlock::from_bytes(&self.data[block_start..block_end]);
                block.dequantize(within_block)
            }
            TensorDtype::Q4_K_S => {
                let block_idx = idx / QK_K;
                let within_block = idx % QK_K;
                let block_start = block_idx * Q4KSBlock::SIZE;
                let block_end = block_start + Q4KSBlock::SIZE;
                let block = Q4KSBlock::from_bytes(&self.data[block_start..block_end]);
                block.dequantize(within_block)
            }
        }
    }

    /// Iterate over rows lazily, returning dequantized f32 vectors
    pub fn rows(&self) -> impl Iterator<Item = Vec<f32>> + '_ {
        (0..self.nrows()).map(move |row| self.get_row(row))
    }

    /// Perform matrix-vector multiplication: out = self @ x
    /// where self is [nrows, ncols] and x is [ncols]
    /// Returns [nrows] output vector.
    /// This fuses dequantization with the dot product for efficiency.
    pub fn matmul_vec(&self, x: &[f32]) -> Vec<f32> {
        let nrows = self.nrows();
        let ncols = self.ncols();
        assert_eq!(x.len(), ncols, "Input vector size mismatch");

        let mut result = vec![0.0f32; nrows];

        match self.dtype {
            TensorDtype::F32 => {
                // Direct f32 dot products
                for (row, result_elem) in result.iter_mut().enumerate() {
                    let byte_start = row * ncols * 4;
                    let mut cursor = Cursor::new(&self.data[byte_start..]);
                    let mut dot = 0.0f32;
                    for x_elem in x.iter() {
                        let w = cursor.read_f32::<LittleEndian>().unwrap();
                        dot += w * x_elem;
                    }
                    *result_elem = dot;
                }
            }
            TensorDtype::F16 => {
                // Fused f16->f32 conversion + dot product
                for (row, result_elem) in result.iter_mut().enumerate() {
                    let byte_start = row * ncols * 2;
                    let row_bytes = &self.data[byte_start..byte_start + ncols * 2];
                    let mut dot = 0.0f32;
                    for (i, x_elem) in x.iter().enumerate() {
                        let f16_val = f16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
                        dot += f16_val.to_f32() * x_elem;
                    }
                    *result_elem = dot;
                }
            }
            TensorDtype::BF16 => {
                // Fused bf16->f32 conversion + dot product
                for (row, result_elem) in result.iter_mut().enumerate() {
                    let byte_start = row * ncols * 2;
                    let row_bytes = &self.data[byte_start..byte_start + ncols * 2];
                    let mut dot = 0.0f32;
                    for (i, x_elem) in x.iter().enumerate() {
                        let bits = u16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
                        let w = f32::from_bits((bits as u32) << 16);
                        dot += w * x_elem;
                    }
                    *result_elem = dot;
                }
            }
            TensorDtype::Int8 => {
                // Fused dequantize + dot product (legacy format)
                let scales_bytes = self.scales.expect("Int8 tensor must have scales");

                for (row, result_elem) in result.iter_mut().enumerate() {
                    let row_start = row * ncols;
                    let row_bytes = &self.data[row_start..row_start + ncols];
                    let mut dot = 0.0f32;

                    for (i, (&q_val, &x_val)) in row_bytes.iter().zip(x.iter()).enumerate() {
                        let global_idx = row_start + i;
                        let group_idx = global_idx / self.group_size;
                        let scale_byte_offset = group_idx * 4;
                        let scale = f32::from_le_bytes([
                            scales_bytes[scale_byte_offset],
                            scales_bytes[scale_byte_offset + 1],
                            scales_bytes[scale_byte_offset + 2],
                            scales_bytes[scale_byte_offset + 3],
                        ]);
                        let w = q_val as i8 as f32 / scale;
                        dot += w * x_val;
                    }
                    *result_elem = dot;
                }
            }
            TensorDtype::Q8_0 => {
                // Fused Q8_0 dequantize + dot product
                let n_blocks_per_row = ncols.div_ceil(QK8_0);

                for (row, result_elem) in result.iter_mut().enumerate() {
                    let row_offset = row * n_blocks_per_row * Q8_0Block::SIZE;
                    let mut dot = 0.0f32;
                    let mut x_idx = 0;

                    for b in 0..n_blocks_per_row {
                        let block_start = row_offset + b * Q8_0Block::SIZE;
                        let block = Q8_0Block::from_bytes(
                            &self.data[block_start..block_start + Q8_0Block::SIZE],
                        );
                        let scale = block.scale_f32();

                        let elements_in_block = (ncols - x_idx).min(QK8_0);
                        for i in 0..elements_in_block {
                            let w = block.qs[i] as f32 * scale;
                            dot += w * x[x_idx];
                            x_idx += 1;
                        }
                    }
                    *result_elem = dot;
                }
            }
            TensorDtype::Q4_0 => {
                // Fused Q4_0 dequantize + dot product
                let n_blocks_per_row = ncols.div_ceil(QK4_0);

                for (row, result_elem) in result.iter_mut().enumerate() {
                    let row_offset = row * n_blocks_per_row * Q4_0Block::SIZE;
                    let mut dot = 0.0f32;
                    let mut x_idx = 0;

                    for b in 0..n_blocks_per_row {
                        let block_start = row_offset + b * Q4_0Block::SIZE;
                        let block = Q4_0Block::from_bytes(
                            &self.data[block_start..block_start + Q4_0Block::SIZE],
                        );
                        let scale = block.scale_f32();

                        let elements_in_block = (ncols - x_idx).min(QK4_0);
                        for i in 0..elements_in_block {
                            let byte_idx = i / 2;
                            let nibble = if i % 2 == 0 {
                                block.qs[byte_idx] & 0x0F
                            } else {
                                block.qs[byte_idx] >> 4
                            };
                            let w = (nibble as i8 - 8) as f32 * scale;
                            dot += w * x[x_idx];
                            x_idx += 1;
                        }
                    }
                    *result_elem = dot;
                }
            }
            TensorDtype::Q4_K_M => {
                // Fused Q4_K_M dequantize + dot product
                let n_blocks_per_row = ncols.div_ceil(QK_K);

                for (row, result_elem) in result.iter_mut().enumerate() {
                    let row_offset = row * n_blocks_per_row * Q4KMBlock::SIZE;
                    let mut dot = 0.0f32;
                    let mut x_idx = 0;

                    for b in 0..n_blocks_per_row {
                        let block_start = row_offset + b * Q4KMBlock::SIZE;
                        let block = Q4KMBlock::from_bytes(
                            &self.data[block_start..block_start + Q4KMBlock::SIZE],
                        );

                        let elements_in_block = (ncols - x_idx).min(QK_K);
                        for i in 0..elements_in_block {
                            let w = block.dequantize(i);
                            dot += w * x[x_idx];
                            x_idx += 1;
                        }
                    }
                    *result_elem = dot;
                }
            }
            TensorDtype::Q4_K_S => {
                // Fused Q4_K_S dequantize + dot product
                let n_blocks_per_row = ncols.div_ceil(QK_K);

                for (row, result_elem) in result.iter_mut().enumerate() {
                    let row_offset = row * n_blocks_per_row * Q4KSBlock::SIZE;
                    let mut dot = 0.0f32;
                    let mut x_idx = 0;

                    for b in 0..n_blocks_per_row {
                        let block_start = row_offset + b * Q4KSBlock::SIZE;
                        let block = Q4KSBlock::from_bytes(
                            &self.data[block_start..block_start + Q4KSBlock::SIZE],
                        );

                        let elements_in_block = (ncols - x_idx).min(QK_K);
                        for i in 0..elements_in_block {
                            let w = block.dequantize(i);
                            dot += w * x[x_idx];
                            x_idx += 1;
                        }
                    }
                    *result_elem = dot;
                }
            }
        }
        result
    }

    /// Perform matrix-vector multiplication into a pre-allocated buffer
    /// out = self @ x, avoiding allocation
    pub fn matmul_vec_into(&self, x: &[f32], out: &mut [f32]) {
        let nrows = self.nrows();
        let ncols = self.ncols();
        assert_eq!(x.len(), ncols, "Input vector size mismatch");
        assert_eq!(out.len(), nrows, "Output vector size mismatch");

        match self.dtype {
            TensorDtype::F32 => {
                for (row, out_elem) in out.iter_mut().enumerate() {
                    let byte_start = row * ncols * 4;
                    let mut cursor = Cursor::new(&self.data[byte_start..]);
                    let mut dot = 0.0f32;
                    for x_elem in x.iter() {
                        let w = cursor.read_f32::<LittleEndian>().unwrap();
                        dot += w * x_elem;
                    }
                    *out_elem = dot;
                }
            }
            TensorDtype::F16 => {
                for (row, out_elem) in out.iter_mut().enumerate() {
                    let byte_start = row * ncols * 2;
                    let row_bytes = &self.data[byte_start..byte_start + ncols * 2];
                    let mut dot = 0.0f32;
                    for (i, x_elem) in x.iter().enumerate() {
                        let f16_val = f16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
                        dot += f16_val.to_f32() * x_elem;
                    }
                    *out_elem = dot;
                }
            }
            TensorDtype::BF16 => {
                for (row, out_elem) in out.iter_mut().enumerate() {
                    let byte_start = row * ncols * 2;
                    let row_bytes = &self.data[byte_start..byte_start + ncols * 2];
                    let mut dot = 0.0f32;
                    for (i, x_elem) in x.iter().enumerate() {
                        let bits = u16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
                        let w = f32::from_bits((bits as u32) << 16);
                        dot += w * x_elem;
                    }
                    *out_elem = dot;
                }
            }
            TensorDtype::Int8 => {
                let scales_bytes = self.scales.expect("Int8 tensor must have scales");

                for (row, out_elem) in out.iter_mut().enumerate() {
                    let row_start = row * ncols;
                    let row_bytes = &self.data[row_start..row_start + ncols];
                    let mut dot = 0.0f32;

                    for (i, (&q_val, &x_val)) in row_bytes.iter().zip(x.iter()).enumerate() {
                        let global_idx = row_start + i;
                        let group_idx = global_idx / self.group_size;
                        let scale_byte_offset = group_idx * 4;
                        let scale = f32::from_le_bytes([
                            scales_bytes[scale_byte_offset],
                            scales_bytes[scale_byte_offset + 1],
                            scales_bytes[scale_byte_offset + 2],
                            scales_bytes[scale_byte_offset + 3],
                        ]);
                        let w = q_val as i8 as f32 / scale;
                        dot += w * x_val;
                    }
                    *out_elem = dot;
                }
            }
            TensorDtype::Q8_0 => {
                let n_blocks_per_row = ncols.div_ceil(QK8_0);

                for (row, out_elem) in out.iter_mut().enumerate() {
                    let row_offset = row * n_blocks_per_row * Q8_0Block::SIZE;
                    let mut dot = 0.0f32;
                    let mut x_idx = 0;

                    for b in 0..n_blocks_per_row {
                        let block_start = row_offset + b * Q8_0Block::SIZE;
                        let block = Q8_0Block::from_bytes(
                            &self.data[block_start..block_start + Q8_0Block::SIZE],
                        );
                        let scale = block.scale_f32();

                        let elements_in_block = (ncols - x_idx).min(QK8_0);
                        for i in 0..elements_in_block {
                            let w = block.qs[i] as f32 * scale;
                            dot += w * x[x_idx];
                            x_idx += 1;
                        }
                    }
                    *out_elem = dot;
                }
            }
            TensorDtype::Q4_0 => {
                let n_blocks_per_row = ncols.div_ceil(QK4_0);

                for (row, out_elem) in out.iter_mut().enumerate() {
                    let row_offset = row * n_blocks_per_row * Q4_0Block::SIZE;
                    let mut dot = 0.0f32;
                    let mut x_idx = 0;

                    for b in 0..n_blocks_per_row {
                        let block_start = row_offset + b * Q4_0Block::SIZE;
                        let block = Q4_0Block::from_bytes(
                            &self.data[block_start..block_start + Q4_0Block::SIZE],
                        );
                        let scale = block.scale_f32();

                        let elements_in_block = (ncols - x_idx).min(QK4_0);
                        for i in 0..elements_in_block {
                            let byte_idx = i / 2;
                            let nibble = if i % 2 == 0 {
                                block.qs[byte_idx] & 0x0F
                            } else {
                                block.qs[byte_idx] >> 4
                            };
                            let w = (nibble as i8 - 8) as f32 * scale;
                            dot += w * x[x_idx];
                            x_idx += 1;
                        }
                    }
                    *out_elem = dot;
                }
            }
            TensorDtype::Q4_K_M => {
                let n_blocks_per_row = ncols.div_ceil(QK_K);

                for (row, out_elem) in out.iter_mut().enumerate() {
                    let row_offset = row * n_blocks_per_row * Q4KMBlock::SIZE;
                    let mut dot = 0.0f32;
                    let mut x_idx = 0;

                    for b in 0..n_blocks_per_row {
                        let block_start = row_offset + b * Q4KMBlock::SIZE;
                        let block = Q4KMBlock::from_bytes(
                            &self.data[block_start..block_start + Q4KMBlock::SIZE],
                        );

                        let elements_in_block = (ncols - x_idx).min(QK_K);
                        for i in 0..elements_in_block {
                            let w = block.dequantize(i);
                            dot += w * x[x_idx];
                            x_idx += 1;
                        }
                    }
                    *out_elem = dot;
                }
            }
            TensorDtype::Q4_K_S => {
                let n_blocks_per_row = ncols.div_ceil(QK_K);

                for (row, out_elem) in out.iter_mut().enumerate() {
                    let row_offset = row * n_blocks_per_row * Q4KSBlock::SIZE;
                    let mut dot = 0.0f32;
                    let mut x_idx = 0;

                    for b in 0..n_blocks_per_row {
                        let block_start = row_offset + b * Q4KSBlock::SIZE;
                        let block = Q4KSBlock::from_bytes(
                            &self.data[block_start..block_start + Q4KSBlock::SIZE],
                        );

                        let elements_in_block = (ncols - x_idx).min(QK_K);
                        for i in 0..elements_in_block {
                            let w = block.dequantize(i);
                            dot += w * x[x_idx];
                            x_idx += 1;
                        }
                    }
                    *out_elem = dot;
                }
            }
        }
    }
}

pub struct Parameters {
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub tensors: HashMap<String, TensorInfo>,
    mmap: Mmap,
    payload_offset: usize,
}

impl Parameters {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref()).context("Failed to open model file")?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Read header size (8 bytes)
        let mut cursor = Cursor::new(&mmap[..8]);
        let header_size = cursor.read_u64::<LittleEndian>()? as usize;

        // Read and parse JSON header
        let header_start = 8;
        let header_end = header_start + header_size;
        let mut header_bytes = &mmap[header_start..header_end];

        // Trim trailing null bytes (padding)
        while header_bytes.last() == Some(&0) {
            header_bytes = &header_bytes[..header_bytes.len() - 1];
        }

        let header: Header =
            serde_json::from_slice(header_bytes).context("Failed to parse JSON header")?;

        // Parse config from metadata
        let config = Config {
            hidden_size: header.metadata["hidden_size"].parse()?,
            intermediate_size: header.metadata["intermediate_size"].parse()?,
            n_layers: header.metadata["n_layers"].parse()?,
            n_heads: header.metadata["n_heads"].parse()?,
            n_kv_heads: header.metadata["n_kv_heads"].parse()?,
            vocab_size: header.metadata["vocab_size"].parse()?,
            max_position_embeddings: header.metadata["max_position_embeddings"].parse()?,
            sliding_window: header.metadata["sliding_window"].parse()?,
            rope_theta: header.metadata["rope_theta"].parse()?,
            norm_eps: header.metadata["norm_eps"].parse()?,
            act_type: header.metadata["act_type"].clone(),
            quant: header.metadata["quant"].clone(),
        };

        let tokenizer = Tokenizer::new(header.tokenizer.vocab, header.tokenizer.merges);

        // Calculate payload offset (aligned to 64 bytes)
        let payload_offset = header_end.div_ceil(64) * 64;

        Ok(Self {
            config,
            tokenizer,
            tensors: header.tensors,
            mmap,
            payload_offset,
        })
    }

    /// Get a view of a tensor's data as f32
    pub fn get_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .context(format!("Tensor '{}' not found", name))?;

        let numel: usize = info.shape.iter().product();
        let data_offset = self.payload_offset + info.offset;

        match info.dtype.as_str() {
            "f32" => {
                // Read f32 directly
                let data_end = data_offset + numel * 4;
                let bytes = &self.mmap[data_offset..data_end];
                let mut result = Vec::with_capacity(numel);
                let mut cursor = Cursor::new(bytes);

                for _ in 0..numel {
                    result.push(cursor.read_f32::<LittleEndian>()?);
                }

                Ok(result)
            }
            "f16" => {
                // Read f16 and convert to f32
                let data_end = data_offset + numel * 2;
                let bytes = &self.mmap[data_offset..data_end];
                let mut result = Vec::with_capacity(numel);

                for i in 0..numel {
                    let f16_val = f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    result.push(f16_val.to_f32());
                }

                Ok(result)
            }
            "bf16" => {
                // Read bf16 and convert to f32
                let data_end = data_offset + numel * 2;
                let bytes = &self.mmap[data_offset..data_end];
                let mut result = Vec::with_capacity(numel);

                for i in 0..numel {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    result.push(f32::from_bits((bits as u32) << 16));
                }

                Ok(result)
            }
            "int8" => {
                // Read int8 and dequantize (legacy format)
                let scale_offset = self.payload_offset + info.scale_offset.unwrap();
                let scale_size = info.scale_size.unwrap();
                let group_size = 64;

                // Read scales
                let scale_end = scale_offset + scale_size * 4;
                let scale_bytes = &self.mmap[scale_offset..scale_end];
                let mut scales = Vec::with_capacity(scale_size);
                let mut cursor = Cursor::new(scale_bytes);

                for _ in 0..scale_size {
                    scales.push(cursor.read_f32::<LittleEndian>()?);
                }

                // Read quantized data
                let data_end = data_offset + numel;
                let quant_data = &self.mmap[data_offset..data_end];

                // Dequantize
                let mut result = Vec::with_capacity(numel);
                for (i, &q_val) in quant_data.iter().enumerate() {
                    let group_idx = i / group_size;
                    let scale = scales[group_idx];
                    result.push(q_val as f32 / scale);
                }

                Ok(result)
            }
            "q8_0" => {
                // Read Q8_0 blocks and dequantize
                let n_blocks = numel.div_ceil(QK8_0);
                let data_end = data_offset + n_blocks * Q8_0Block::SIZE;
                let bytes = &self.mmap[data_offset..data_end];
                let mut result = Vec::with_capacity(numel);

                for block_data in bytes.chunks(Q8_0Block::SIZE) {
                    let block = Q8_0Block::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);

                Ok(result)
            }
            "q4_0" => {
                // Read Q4_0 blocks and dequantize
                let n_blocks = numel.div_ceil(QK4_0);
                let data_end = data_offset + n_blocks * Q4_0Block::SIZE;
                let bytes = &self.mmap[data_offset..data_end];
                let mut result = Vec::with_capacity(numel);

                for block_data in bytes.chunks(Q4_0Block::SIZE) {
                    let block = Q4_0Block::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);

                Ok(result)
            }
            "q4_k_m" | "q4_k" => {
                // Read Q4_K_M blocks and dequantize
                let n_blocks = numel.div_ceil(QK_K);
                let data_end = data_offset + n_blocks * Q4KMBlock::SIZE;
                let bytes = &self.mmap[data_offset..data_end];
                let mut result = Vec::with_capacity(numel);

                for block_data in bytes.chunks(Q4KMBlock::SIZE) {
                    let block = Q4KMBlock::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);

                Ok(result)
            }
            "q4_k_s" => {
                // Read Q4_K_S blocks and dequantize
                let n_blocks = numel.div_ceil(QK_K);
                let data_end = data_offset + n_blocks * Q4KSBlock::SIZE;
                let bytes = &self.mmap[data_offset..data_end];
                let mut result = Vec::with_capacity(numel);

                for block_data in bytes.chunks(Q4KSBlock::SIZE) {
                    let block = Q4KSBlock::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);

                Ok(result)
            }
            _ => anyhow::bail!("Unsupported dtype: {}", info.dtype),
        }
    }

    pub fn get_tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|info| info.shape.as_slice())
    }

    /// Get a lazy view of a tensor's data without copying or dequantizing.
    /// The data remains in the memory-mapped file until accessed.
    pub fn get_tensor_view(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self
            .tensors
            .get(name)
            .context(format!("Tensor '{}' not found", name))?;

        let numel: usize = info.shape.iter().product();
        let data_offset = self.payload_offset + info.offset;

        match info.dtype.as_str() {
            "f32" => {
                // View f32 data directly as bytes
                let data_end = data_offset + numel * 4;
                let data = &self.mmap[data_offset..data_end];

                Ok(TensorView {
                    data,
                    shape: info.shape.clone(),
                    dtype: TensorDtype::F32,
                    scales: None,
                    group_size: 0,
                })
            }
            "f16" => {
                // View f16 data directly as bytes
                let data_end = data_offset + numel * 2;
                let data = &self.mmap[data_offset..data_end];

                Ok(TensorView {
                    data,
                    shape: info.shape.clone(),
                    dtype: TensorDtype::F16,
                    scales: None,
                    group_size: 0,
                })
            }
            "bf16" => {
                // View bf16 data directly as bytes
                let data_end = data_offset + numel * 2;
                let data = &self.mmap[data_offset..data_end];

                Ok(TensorView {
                    data,
                    shape: info.shape.clone(),
                    dtype: TensorDtype::BF16,
                    scales: None,
                    group_size: 0,
                })
            }
            "int8" => {
                // View int8 data and scales (legacy format)
                let scale_offset = self.payload_offset + info.scale_offset.unwrap();
                let scale_size = info.scale_size.unwrap();
                let group_size = 64;

                let data_end = data_offset + numel;
                let data = &self.mmap[data_offset..data_end];

                let scale_end = scale_offset + scale_size * 4;
                let scales = &self.mmap[scale_offset..scale_end];

                Ok(TensorView {
                    data,
                    shape: info.shape.clone(),
                    dtype: TensorDtype::Int8,
                    scales: Some(scales),
                    group_size,
                })
            }
            "q8_0" => {
                // View Q8_0 data (blocks are self-contained with scales)
                let n_blocks = numel.div_ceil(QK8_0);
                let data_end = data_offset + n_blocks * Q8_0Block::SIZE;
                let data = &self.mmap[data_offset..data_end];

                Ok(TensorView {
                    data,
                    shape: info.shape.clone(),
                    dtype: TensorDtype::Q8_0,
                    scales: None,
                    group_size: QK8_0,
                })
            }
            "q4_0" => {
                // View Q4_0 data (blocks are self-contained with scales)
                let n_blocks = numel.div_ceil(QK4_0);
                let data_end = data_offset + n_blocks * Q4_0Block::SIZE;
                let data = &self.mmap[data_offset..data_end];

                Ok(TensorView {
                    data,
                    shape: info.shape.clone(),
                    dtype: TensorDtype::Q4_0,
                    scales: None,
                    group_size: QK4_0,
                })
            }
            "q4_k_m" | "q4_k" => {
                // View Q4_K_M data (super-blocks are self-contained)
                let n_blocks = numel.div_ceil(QK_K);
                let data_end = data_offset + n_blocks * Q4KMBlock::SIZE;
                let data = &self.mmap[data_offset..data_end];

                Ok(TensorView {
                    data,
                    shape: info.shape.clone(),
                    dtype: TensorDtype::Q4_K_M,
                    scales: None,
                    group_size: QK_K,
                })
            }
            "q4_k_s" => {
                // View Q4_K_S data (super-blocks are self-contained)
                let n_blocks = numel.div_ceil(QK_K);
                let data_end = data_offset + n_blocks * Q4KSBlock::SIZE;
                let data = &self.mmap[data_offset..data_end];

                Ok(TensorView {
                    data,
                    shape: info.shape.clone(),
                    dtype: TensorDtype::Q4_K_S,
                    scales: None,
                    group_size: QK_K,
                })
            }
            _ => anyhow::bail!("Unsupported dtype: {}", info.dtype),
        }
    }
}
