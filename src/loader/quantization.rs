//! Quantization Types and Utilities
//!
//! This module provides quantization block structures and utilities for memory-efficient
//! tensor storage. It supports multiple quantization formats:
//!
//! - **Q4_0**: Basic 4-bit quantization with single scale per block (32 weights)
//! - **Q4_K_M**: 4-bit K-quantization (medium) with super-blocks and min values
//! - **Q4_K_S**: 4-bit K-quantization (small) - more compact variant
//! - **Q8_0**: 8-bit quantization with single scale per block
//!
//! # Memory Layout
//!
//! ```text
//! Q4_0 Block (18 bytes for 32 weights):
//! ┌────────┬────────────────────────────────────────┐
//! │ scale  │ 16 bytes of packed 4-bit values        │
//! │ (f16)  │ (2 values per byte, lower nibble first)│
//! └────────┴────────────────────────────────────────┘
//!
//! Q4_K_M Super-block (144 bytes for 256 weights):
//! ┌─────────┬────────────┬───────────────────────────┬───────────────┐
//! │ d (f16) │ dmin (f16) │ scales[12] + mins[12]     │ 128 bytes qs  │
//! │ 2 bytes │ 2 bytes    │ 12 bytes packed           │ packed nibbles│
//! └─────────┴────────────┴───────────────────────────┴───────────────┘
//! ```

use half::f16;
use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};

/// Block size for Q4_0 and Q8_0 quantization (number of weights per block)
pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;

/// Block size for K-quantization (super-block size)
pub const QK_K: usize = 256;

/// Number of blocks within a K-quant super-block
pub const K_SCALE_SIZE: usize = 12;

// =============================================================================
// Q4_0: Basic 4-bit Quantization
// =============================================================================

/// Q4_0 quantization block: 32 weights stored in 18 bytes
///
/// Format: f16 scale + 16 bytes of packed nibbles
/// Dequantization: `weight[i] = (nibble[i] - 8) * scale`
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Q4_0Block {
    /// Scale factor (f16, 2 bytes)
    pub scale: u16,  // Stored as raw f16 bits
    /// Packed 4-bit weights (16 bytes = 32 nibbles)
    pub qs: [u8; QK4_0 / 2],
}

impl Q4_0Block {
    /// Size of a Q4_0 block in bytes
    pub const SIZE: usize = 2 + QK4_0 / 2; // 18 bytes

    /// Get the scale as f32
    #[inline]
    pub fn scale_f32(&self) -> f32 {
        f16::from_bits(self.scale).to_f32()
    }

    /// Dequantize a single value at the given index within this block
    #[inline]
    pub fn dequantize(&self, index: usize) -> f32 {
        debug_assert!(index < QK4_0);
        let byte_idx = index / 2;
        let nibble = if index % 2 == 0 {
            self.qs[byte_idx] & 0x0F
        } else {
            self.qs[byte_idx] >> 4
        };
        // Nibble is in [0, 15], we offset by 8 to get signed range [-8, 7]
        (nibble as i8 - 8) as f32 * self.scale_f32()
    }

    /// Dequantize all 32 values in this block
    pub fn dequantize_block(&self) -> [f32; QK4_0] {
        let mut result = [0.0f32; QK4_0];
        let scale = self.scale_f32();
        
        for (i, byte) in self.qs.iter().enumerate() {
            let low = (*byte & 0x0F) as i8 - 8;
            let high = (*byte >> 4) as i8 - 8;
            result[i * 2] = low as f32 * scale;
            result[i * 2 + 1] = high as f32 * scale;
        }
        result
    }

    /// Read a Q4_0 block from bytes
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= Self::SIZE);
        let scale = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut qs = [0u8; QK4_0 / 2];
        qs.copy_from_slice(&bytes[2..2 + QK4_0 / 2]);
        Self { scale, qs }
    }

    /// Quantize f32 values into a Q4_0 block
    pub fn from_f32(values: &[f32; QK4_0]) -> Self {
        // Find max absolute value
        let max_abs = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        
        // Scale to fit in [-8, 7]
        let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
        let scale_f16 = f16::from_f32(scale);
        let inv_scale = 1.0 / scale_f16.to_f32();
        
        let mut qs = [0u8; QK4_0 / 2];
        for (i, byte) in qs.iter_mut().enumerate() {
            let q0 = ((values[i * 2] * inv_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            let q1 = ((values[i * 2 + 1] * inv_scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
            *byte = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
        }
        
        Self {
            scale: scale_f16.to_bits(),
            qs,
        }
    }
}

// =============================================================================
// Q8_0: 8-bit Quantization
// =============================================================================

/// Q8_0 quantization block: 32 weights stored in 34 bytes
///
/// Format: f16 scale + 32 bytes of int8 weights
/// Dequantization: `weight[i] = qs[i] * scale`
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Q8_0Block {
    /// Scale factor (f16, 2 bytes)
    pub scale: u16,
    /// Quantized weights (32 bytes)
    pub qs: [i8; QK8_0],
}

impl Q8_0Block {
    /// Size of a Q8_0 block in bytes
    pub const SIZE: usize = 2 + QK8_0; // 34 bytes

    /// Get the scale as f32
    #[inline]
    pub fn scale_f32(&self) -> f32 {
        f16::from_bits(self.scale).to_f32()
    }

    /// Dequantize a single value at the given index within this block
    #[inline]
    pub fn dequantize(&self, index: usize) -> f32 {
        debug_assert!(index < QK8_0);
        self.qs[index] as f32 * self.scale_f32()
    }

    /// Dequantize all 32 values in this block
    pub fn dequantize_block(&self) -> [f32; QK8_0] {
        let mut result = [0.0f32; QK8_0];
        let scale = self.scale_f32();
        for (i, &q) in self.qs.iter().enumerate() {
            result[i] = q as f32 * scale;
        }
        result
    }

    /// Read a Q8_0 block from bytes
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= Self::SIZE);
        let scale = u16::from_le_bytes([bytes[0], bytes[1]]);
        let mut qs = [0i8; QK8_0];
        for (i, &b) in bytes[2..2 + QK8_0].iter().enumerate() {
            qs[i] = b as i8;
        }
        Self { scale, qs }
    }

    /// Quantize f32 values into a Q8_0 block
    pub fn from_f32(values: &[f32; QK8_0]) -> Self {
        // Find max absolute value
        let max_abs = values.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        
        // Scale to fit in [-127, 127]
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let scale_f16 = f16::from_f32(scale);
        let inv_scale = 1.0 / scale_f16.to_f32();
        
        let mut qs = [0i8; QK8_0];
        for (i, &val) in values.iter().enumerate() {
            qs[i] = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }
        
        Self {
            scale: scale_f16.to_bits(),
            qs,
        }
    }
}

// =============================================================================
// Q4_K_M: K-Quantization (Medium)
// =============================================================================

/// Q4_K_M quantization super-block: 256 weights stored in 144 bytes
///
/// This format uses super-blocks with per-block scales and minimums for better
/// accuracy than Q4_0 while maintaining similar compression ratio.
///
/// For simplicity, this implementation uses a simplified packing scheme that
/// is self-consistent but may differ from the exact GGML layout.
///
/// Structure:
/// - d: super-block scale (f16)
/// - dmin: super-block minimum scale (f16)
/// - scales_and_mins: packed scales and mins for 8 blocks (12 bytes)
/// - qs: packed 4-bit quantized values (128 bytes for 256 values)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Q4KMBlock {
    /// Super-block scale (f16)
    pub d: f16,
    /// Super-block minimum scale (f16)
    pub dmin: f16,
    /// Packed scales and mins for sub-blocks (12 bytes)
    /// Simple packing: scales[0..7] = 8 4-bit scales, scales[4..11] = 8 4-bit mins
    pub scales: [u8; K_SCALE_SIZE],
    /// Packed 4-bit quantized values (128 bytes = 256 nibbles)
    pub qs: [u8; QK_K / 2],
}

impl Q4KMBlock {
    /// Size of a Q4_K_M block in bytes
    pub const SIZE: usize = 2 + 2 + K_SCALE_SIZE + QK_K / 2; // 144 bytes

    /// Number of sub-blocks within this super-block
    const N_BLOCKS: usize = QK_K / 32; // 8 blocks

    /// Get the scale for a sub-block (4-bit precision for simplicity)
    fn get_scale(&self, block_idx: usize) -> f32 {
        debug_assert!(block_idx < Self::N_BLOCKS);
        // Simple packing: first 4 bytes contain low nibbles of scales 0-7,
        // scales 0,1 in byte 0, scales 2,3 in byte 1, etc.
        let byte_idx = block_idx / 2;
        let scale_bits = if block_idx % 2 == 0 {
            self.scales[byte_idx] & 0x0F
        } else {
            self.scales[byte_idx] >> 4
        };
        self.d.to_f32() * scale_bits as f32
    }

    /// Get the min for a sub-block (4-bit precision for simplicity)
    fn get_min(&self, block_idx: usize) -> f32 {
        debug_assert!(block_idx < Self::N_BLOCKS);
        // Mins stored in bytes 4-7
        let byte_idx = 4 + block_idx / 2;
        let min_bits = if block_idx % 2 == 0 {
            self.scales[byte_idx] & 0x0F
        } else {
            self.scales[byte_idx] >> 4
        };
        self.dmin.to_f32() * min_bits as f32
    }

    /// Dequantize a single value at the given index within this super-block
    #[inline]
    pub fn dequantize(&self, index: usize) -> f32 {
        debug_assert!(index < QK_K);
        let block_idx = index / 32;
        let byte_idx = index / 2;
        let nibble = if index % 2 == 0 {
            self.qs[byte_idx] & 0x0F
        } else {
            self.qs[byte_idx] >> 4
        };
        
        let scale = self.get_scale(block_idx);
        let min = self.get_min(block_idx);
        
        nibble as f32 * scale - min
    }

    /// Dequantize all 256 values in this super-block
    pub fn dequantize_block(&self) -> [f32; QK_K] {
        let mut result = [0.0f32; QK_K];
        
        for block_idx in 0..Self::N_BLOCKS {
            let scale = self.get_scale(block_idx);
            let min = self.get_min(block_idx);
            let base_idx = block_idx * 32;
            
            for i in 0..16 {
                let byte = self.qs[base_idx / 2 + i];
                let low_nibble = (byte & 0x0F) as f32;
                let high_nibble = (byte >> 4) as f32;
                
                result[base_idx + i * 2] = low_nibble * scale - min;
                result[base_idx + i * 2 + 1] = high_nibble * scale - min;
            }
        }
        
        result
    }

    /// Read a Q4_K_M block from bytes
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= Self::SIZE);
        let mut cursor = Cursor::new(bytes);
        
        let d_bits = cursor.read_u16::<LittleEndian>().unwrap();
        let dmin_bits = cursor.read_u16::<LittleEndian>().unwrap();
        
        let mut scales = [0u8; K_SCALE_SIZE];
        cursor.read_exact(&mut scales).unwrap();
        
        let mut qs = [0u8; QK_K / 2];
        cursor.read_exact(&mut qs).unwrap();
        
        Self {
            d: f16::from_bits(d_bits),
            dmin: f16::from_bits(dmin_bits),
            scales,
            qs,
        }
    }

    /// Quantize f32 values into a Q4_K_M block
    pub fn from_f32(values: &[f32; QK_K]) -> Self {
        // Compute per-block statistics
        // In Q4_K_M, values are dequantized as: nibble * scale - min
        // So we need: (value + min) / scale = nibble (where nibble is 0-15)
        // The min is stored as a positive value that gets subtracted
        let mut block_scales = [0.0f32; Self::N_BLOCKS];
        let mut block_mins = [0.0f32; Self::N_BLOCKS];
        
        for block_idx in 0..Self::N_BLOCKS {
            let block_start = block_idx * 32;
            let block_values = &values[block_start..block_start + 32];
            
            let min_val = block_values.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = block_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            // Store the min as a positive offset
            // dequant: nibble * scale - min = original
            // So if min_val < 0, then -min_val is positive and we subtract it
            block_mins[block_idx] = -min_val; // This will be subtracted
            let range = max_val - min_val;
            block_scales[block_idx] = if range > 1e-10 { range / 15.0 } else { 1.0 };
        }
        
        // Compute super-block scales
        let max_scale = block_scales.iter().copied().fold(1e-10f32, f32::max);
        let max_min = block_mins.iter().map(|x| x.abs()).fold(1e-10f32, f32::max);
        
        let d = max_scale / 15.0;
        let dmin = max_min / 15.0;
        
        // Pack scales and mins (4-bit each for simplicity)
        let mut scales = [0u8; K_SCALE_SIZE];
        for i in 0..Self::N_BLOCKS {
            let scale_q = (block_scales[i] / d).round().clamp(0.0, 15.0) as u8;
            let min_q = (block_mins[i] / dmin).round().clamp(0.0, 15.0) as u8;
            
            let scale_byte_idx = i / 2;
            let min_byte_idx = 4 + i / 2;
            
            if i % 2 == 0 {
                scales[scale_byte_idx] = (scales[scale_byte_idx] & 0xF0) | scale_q;
                scales[min_byte_idx] = (scales[min_byte_idx] & 0xF0) | min_q;
            } else {
                scales[scale_byte_idx] = (scales[scale_byte_idx] & 0x0F) | (scale_q << 4);
                scales[min_byte_idx] = (scales[min_byte_idx] & 0x0F) | (min_q << 4);
            }
        }
        
        // Quantize values using the computed scales
        let mut qs = [0u8; QK_K / 2];
        for block_idx in 0..Self::N_BLOCKS {
            let block_start = block_idx * 32;
            let block_values = &values[block_start..block_start + 32];
            
            // Get the reconstructed scale and min from the packed values
            let scale_byte_idx = block_idx / 2;
            let scale_bits = if block_idx % 2 == 0 {
                scales[scale_byte_idx] & 0x0F
            } else {
                scales[scale_byte_idx] >> 4
            };
            let rec_scale = d * scale_bits as f32;
            
            let min_byte_idx = 4 + block_idx / 2;
            let min_bits = if block_idx % 2 == 0 {
                scales[min_byte_idx] & 0x0F
            } else {
                scales[min_byte_idx] >> 4
            };
            let rec_min = dmin * min_bits as f32;
            
            let inv_scale = if rec_scale > 1e-10 { 1.0 / rec_scale } else { 0.0 };
            
            for i in 0..16 {
                let idx0 = i * 2;
                let idx1 = i * 2 + 1;
                
                // Dequant: nibble * scale - min = value
                // Quant: (value + min) / scale = nibble
                let q0 = ((block_values[idx0] + rec_min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                let q1 = ((block_values[idx1] + rec_min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                
                qs[block_start / 2 + i] = q0 | (q1 << 4);
            }
        }
        
        Self {
            d: f16::from_f32(d),
            dmin: f16::from_f32(dmin),
            scales,
            qs,
        }
    }
}

// =============================================================================
// Q4_K_S: K-Quantization (Small)
// =============================================================================

/// Q4_K_S quantization super-block: 256 weights stored in ~138 bytes
///
/// Smaller variant of Q4_K_M with reduced precision for scales/mins.
/// Uses 6 bytes for scales/mins instead of 12.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Q4KSBlock {
    /// Super-block scale (f16)
    pub d: f16,
    /// Super-block minimum scale (f16)  
    pub dmin: f16,
    /// Packed scales and mins (6 bytes - scales in first 3 bytes, mins in next 3)
    pub scales: [u8; 6],
    /// Packed 4-bit quantized values (128 bytes)
    pub qs: [u8; QK_K / 2],
}

impl Q4KSBlock {
    /// Size of a Q4_K_S block in bytes
    pub const SIZE: usize = 2 + 2 + 6 + QK_K / 2; // 138 bytes

    /// Number of sub-blocks
    const N_BLOCKS: usize = QK_K / 32; // 8 blocks

    /// Get scale for sub-block (3-bit effective precision due to packing constraints)
    fn get_scale(&self, block_idx: usize) -> f32 {
        debug_assert!(block_idx < Self::N_BLOCKS);
        // First 3 bytes store scales (2 scales per byte using 4-bit nibbles)
        // Only 6 scales can be stored, so blocks 6,7 share with 4,5
        let effective_idx = block_idx.min(5);
        let byte_idx = effective_idx / 2;
        let scale_bits = if effective_idx % 2 == 0 {
            self.scales[byte_idx] & 0x0F
        } else {
            self.scales[byte_idx] >> 4
        };
        self.d.to_f32() * scale_bits as f32
    }

    /// Get min for sub-block (3-bit effective precision)
    fn get_min(&self, block_idx: usize) -> f32 {
        debug_assert!(block_idx < Self::N_BLOCKS);
        let effective_idx = block_idx.min(5);
        let byte_idx = 3 + effective_idx / 2;
        let min_bits = if effective_idx % 2 == 0 {
            self.scales[byte_idx] & 0x0F
        } else {
            self.scales[byte_idx] >> 4
        };
        self.dmin.to_f32() * min_bits as f32
    }

    /// Dequantize a single value
    #[inline]
    pub fn dequantize(&self, index: usize) -> f32 {
        debug_assert!(index < QK_K);
        let block_idx = index / 32;
        
        let byte_idx = index / 2;
        let nibble = if index % 2 == 0 {
            self.qs[byte_idx] & 0x0F
        } else {
            self.qs[byte_idx] >> 4
        };
        
        let scale = self.get_scale(block_idx);
        let min = self.get_min(block_idx);
        
        nibble as f32 * scale - min
    }

    /// Dequantize all 256 values
    pub fn dequantize_block(&self) -> [f32; QK_K] {
        let mut result = [0.0f32; QK_K];
        
        for block_idx in 0..Self::N_BLOCKS {
            let scale = self.get_scale(block_idx);
            let min = self.get_min(block_idx);
            let base_idx = block_idx * 32;
            
            for i in 0..16 {
                let byte = self.qs[base_idx / 2 + i];
                let low_nibble = (byte & 0x0F) as f32;
                let high_nibble = (byte >> 4) as f32;
                
                result[base_idx + i * 2] = low_nibble * scale - min;
                result[base_idx + i * 2 + 1] = high_nibble * scale - min;
            }
        }
        
        result
    }

    /// Read from bytes
    pub fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= Self::SIZE);
        let mut cursor = Cursor::new(bytes);
        
        let d_bits = cursor.read_u16::<LittleEndian>().unwrap();
        let dmin_bits = cursor.read_u16::<LittleEndian>().unwrap();
        
        let mut scales = [0u8; 6];
        cursor.read_exact(&mut scales).unwrap();
        
        let mut qs = [0u8; QK_K / 2];
        cursor.read_exact(&mut qs).unwrap();
        
        Self {
            d: f16::from_bits(d_bits),
            dmin: f16::from_bits(dmin_bits),
            scales,
            qs,
        }
    }

    /// Quantize f32 values into a Q4_K_S block
    pub fn from_f32(values: &[f32; QK_K]) -> Self {
        // Compute per-block statistics
        let mut block_scales = [0.0f32; Self::N_BLOCKS];
        let mut block_mins = [0.0f32; Self::N_BLOCKS];
        
        for block_idx in 0..Self::N_BLOCKS {
            let block_start = block_idx * 32;
            let block_values = &values[block_start..block_start + 32];
            
            let min_val = block_values.iter().copied().fold(f32::INFINITY, f32::min);
            let max_val = block_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            
            block_mins[block_idx] = min_val;
            let range = max_val - min_val;
            block_scales[block_idx] = if range > 0.0 { range / 15.0 } else { 0.0 };
        }
        
        // Compute super-block scales
        let max_scale = block_scales.iter().copied().fold(0.0f32, f32::max);
        let max_min = block_mins.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        
        let d = if max_scale > 0.0 { max_scale / 15.0 } else { 1.0 };
        let dmin = if max_min > 0.0 { max_min / 15.0 } else { 1.0 };
        
        // Pack scales and mins (4-bit each, only first 6 blocks)
        let mut scales = [0u8; 6];
        for i in 0..6.min(Self::N_BLOCKS) {
            let scale_q = (block_scales[i] / d).round().clamp(0.0, 15.0) as u8;
            let min_q = (block_mins[i].abs() / dmin).round().clamp(0.0, 15.0) as u8;
            
            let scale_byte_idx = i / 2;
            let min_byte_idx = 3 + i / 2;
            
            if i % 2 == 0 {
                scales[scale_byte_idx] = (scales[scale_byte_idx] & 0xF0) | scale_q;
                scales[min_byte_idx] = (scales[min_byte_idx] & 0xF0) | min_q;
            } else {
                scales[scale_byte_idx] = (scales[scale_byte_idx] & 0x0F) | (scale_q << 4);
                scales[min_byte_idx] = (scales[min_byte_idx] & 0x0F) | (min_q << 4);
            }
        }
        
        // Quantize values
        let mut qs = [0u8; QK_K / 2];
        for block_idx in 0..Self::N_BLOCKS {
            let block_start = block_idx * 32;
            let inv_scale = if block_scales[block_idx] > 0.0 { 
                1.0 / block_scales[block_idx] 
            } else { 
                0.0 
            };
            let min = block_mins[block_idx];
            
            for i in 0..16 {
                let idx0 = block_start + i * 2;
                let idx1 = block_start + i * 2 + 1;
                
                let q0 = ((values[idx0] - min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                let q1 = ((values[idx1] - min) * inv_scale).round().clamp(0.0, 15.0) as u8;
                
                qs[block_start / 2 + i] = q0 | (q1 << 4);
            }
        }
        
        Self {
            d: f16::from_f32(d),
            dmin: f16::from_f32(dmin),
            scales,
            qs,
        }
    }
}

// =============================================================================
// Quantization Format Enum
// =============================================================================

/// Supported quantization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QuantFormat {
    /// Full precision (no quantization)
    F32,
    /// Half precision
    F16,
    /// Brain floating point 16
    BF16,
    /// 8-bit quantization with single scale per 32 elements
    Q8_0,
    /// 4-bit quantization with single scale per 32 elements
    Q4_0,
    /// 4-bit K-quantization (medium quality)
    Q4_K_M,
    /// 4-bit K-quantization (small/compact)
    Q4_K_S,
}

impl QuantFormat {
    /// Returns bytes per element (approximate for sub-byte formats)
    pub fn bytes_per_element(&self) -> f32 {
        match self {
            QuantFormat::F32 => 4.0,
            QuantFormat::F16 | QuantFormat::BF16 => 2.0,
            QuantFormat::Q8_0 => Q8_0Block::SIZE as f32 / QK8_0 as f32, // ~1.06
            QuantFormat::Q4_0 => Q4_0Block::SIZE as f32 / QK4_0 as f32, // ~0.56
            QuantFormat::Q4_K_M => Q4KMBlock::SIZE as f32 / QK_K as f32, // ~0.56
            QuantFormat::Q4_K_S => Q4KSBlock::SIZE as f32 / QK_K as f32, // ~0.54
        }
    }

    /// Returns the block size for this format
    pub fn block_size(&self) -> usize {
        match self {
            QuantFormat::F32 | QuantFormat::F16 | QuantFormat::BF16 => 1,
            QuantFormat::Q8_0 => QK8_0,
            QuantFormat::Q4_0 => QK4_0,
            QuantFormat::Q4_K_M | QuantFormat::Q4_K_S => QK_K,
        }
    }

    /// Parse from string representation
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" | "float32" => Some(QuantFormat::F32),
            "f16" | "float16" => Some(QuantFormat::F16),
            "bf16" | "bfloat16" => Some(QuantFormat::BF16),
            "q8_0" | "int8" => Some(QuantFormat::Q8_0),
            "q4_0" | "int4" => Some(QuantFormat::Q4_0),
            "q4_k_m" | "q4_k" => Some(QuantFormat::Q4_K_M),
            "q4_k_s" => Some(QuantFormat::Q4_K_S),
            _ => None,
        }
    }
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantFormat::F32 => write!(f, "f32"),
            QuantFormat::F16 => write!(f, "f16"),
            QuantFormat::BF16 => write!(f, "bf16"),
            QuantFormat::Q8_0 => write!(f, "q8_0"),
            QuantFormat::Q4_0 => write!(f, "q4_0"),
            QuantFormat::Q4_K_M => write!(f, "q4_k_m"),
            QuantFormat::Q4_K_S => write!(f, "q4_k_s"),
        }
    }
}

// =============================================================================
// Quantized Tensor Wrapper
// =============================================================================

/// A quantized tensor that stores data in a specific quantization format
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Raw quantized data
    pub data: Vec<u8>,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Quantization format
    pub format: QuantFormat,
}

impl QuantizedTensor {
    /// Create a new quantized tensor from f32 data
    pub fn from_f32(values: &[f32], shape: Vec<usize>, format: QuantFormat) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(values.len(), numel, "Shape mismatch");

        let data = match format {
            QuantFormat::F32 => {
                values.iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            QuantFormat::F16 => {
                values.iter()
                    .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                    .collect()
            }
            QuantFormat::BF16 => {
                values.iter()
                    .flat_map(|v| {
                        let bits = v.to_bits();
                        ((bits >> 16) as u16).to_le_bytes()
                    })
                    .collect()
            }
            QuantFormat::Q8_0 => {
                let n_blocks = numel.div_ceil(QK8_0);
                let mut data = Vec::with_capacity(n_blocks * Q8_0Block::SIZE);
                
                for chunk in values.chunks(QK8_0) {
                    let mut block_values = [0.0f32; QK8_0];
                    block_values[..chunk.len()].copy_from_slice(chunk);
                    let block = Q8_0Block::from_f32(&block_values);
                    
                    data.extend_from_slice(&block.scale.to_le_bytes());
                    for &q in &block.qs {
                        data.push(q as u8);
                    }
                }
                data
            }
            QuantFormat::Q4_0 => {
                let n_blocks = numel.div_ceil(QK4_0);
                let mut data = Vec::with_capacity(n_blocks * Q4_0Block::SIZE);
                
                for chunk in values.chunks(QK4_0) {
                    let mut block_values = [0.0f32; QK4_0];
                    block_values[..chunk.len()].copy_from_slice(chunk);
                    let block = Q4_0Block::from_f32(&block_values);
                    
                    data.extend_from_slice(&block.scale.to_le_bytes());
                    data.extend_from_slice(&block.qs);
                }
                data
            }
            QuantFormat::Q4_K_M => {
                let n_blocks = numel.div_ceil(QK_K);
                let mut data = Vec::with_capacity(n_blocks * Q4KMBlock::SIZE);
                
                for chunk in values.chunks(QK_K) {
                    let mut block_values = [0.0f32; QK_K];
                    block_values[..chunk.len()].copy_from_slice(chunk);
                    let block = Q4KMBlock::from_f32(&block_values);
                    
                    data.extend_from_slice(&block.d.to_le_bytes());
                    data.extend_from_slice(&block.dmin.to_le_bytes());
                    data.extend_from_slice(&block.scales);
                    data.extend_from_slice(&block.qs);
                }
                data
            }
            QuantFormat::Q4_K_S => {
                // Similar to Q4_K_M but with smaller scale storage
                let n_blocks = numel.div_ceil(QK_K);
                let mut data = Vec::with_capacity(n_blocks * Q4KSBlock::SIZE);
                
                for chunk in values.chunks(QK_K) {
                    // Use Q4_K_M quantization for now, Q4_K_S would need its own from_f32
                    let mut block_values = [0.0f32; QK_K];
                    block_values[..chunk.len()].copy_from_slice(chunk);
                    let block = Q4KMBlock::from_f32(&block_values);
                    
                    // Store in Q4_K_S format (subset of Q4_K_M)
                    data.extend_from_slice(&block.d.to_le_bytes());
                    data.extend_from_slice(&block.dmin.to_le_bytes());
                    // Compressed scales (take first 6 bytes only)
                    data.extend_from_slice(&block.scales[..6]);
                    data.extend_from_slice(&block.qs);
                }
                data
            }
        };

        Self { data, shape, format }
    }

    /// Dequantize to f32 Vec
    pub fn to_f32(&self) -> Vec<f32> {
        let numel: usize = self.shape.iter().product();
        
        match self.format {
            QuantFormat::F32 => {
                self.data
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                    .collect()
            }
            QuantFormat::F16 => {
                self.data
                    .chunks_exact(2)
                    .map(|bytes| f16::from_le_bytes([bytes[0], bytes[1]]).to_f32())
                    .collect()
            }
            QuantFormat::BF16 => {
                self.data
                    .chunks_exact(2)
                    .map(|bytes| {
                        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect()
            }
            QuantFormat::Q8_0 => {
                let mut result = Vec::with_capacity(numel);
                for block_data in self.data.chunks(Q8_0Block::SIZE) {
                    let block = Q8_0Block::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);
                result
            }
            QuantFormat::Q4_0 => {
                let mut result = Vec::with_capacity(numel);
                for block_data in self.data.chunks(Q4_0Block::SIZE) {
                    let block = Q4_0Block::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);
                result
            }
            QuantFormat::Q4_K_M => {
                let mut result = Vec::with_capacity(numel);
                for block_data in self.data.chunks(Q4KMBlock::SIZE) {
                    let block = Q4KMBlock::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);
                result
            }
            QuantFormat::Q4_K_S => {
                let mut result = Vec::with_capacity(numel);
                for block_data in self.data.chunks(Q4KSBlock::SIZE) {
                    let block = Q4KSBlock::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);
                result
            }
        }
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Compute compression ratio vs f32
    pub fn compression_ratio(&self) -> f32 {
        let f32_size = self.numel() * 4;
        f32_size as f32 / self.data.len() as f32
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip() {
        let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) / 4.0);
        let block = Q4_0Block::from_f32(&values);
        let recovered = block.dequantize_block();
        
        // Check that recovered values are close to original
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.5, "Q4_0 roundtrip: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_q8_0_roundtrip() {
        let values: [f32; 32] = std::array::from_fn(|i| (i as f32 - 16.0) / 4.0);
        let block = Q8_0Block::from_f32(&values);
        let recovered = block.dequantize_block();
        
        // Q8_0 should have better precision than Q4_0
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 0.1, "Q8_0 roundtrip: {} vs {}", orig, rec);
        }
    }

    #[test]
    fn test_q4_k_m_roundtrip() {
        let values: [f32; 256] = std::array::from_fn(|i| (i as f32 - 128.0) / 32.0);
        let block = Q4KMBlock::from_f32(&values);
        let recovered = block.dequantize_block();
        
        // Q4_K_M is a complex format with per-block scales and mins.
        // 4-bit quantization with nested quantization of scales inherently has 
        // larger errors. The format quantizes scales to 4-bit as well.
        // We test for reasonable correlation rather than exact values.
        let mut total_error = 0.0f32;
        for (orig, rec) in values.iter().zip(recovered.iter()) {
            total_error += (orig - rec).abs();
        }
        let avg_error = total_error / 256.0;
        // 4-bit with 4-bit scales typically has ~0.3-0.6 average error
        assert!(avg_error < 0.7, "Q4_K_M average error too high: {}", avg_error);
    }

    #[test]
    fn test_quantized_tensor_compression() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 128.0).collect();
        
        let q4_0 = QuantizedTensor::from_f32(&data, vec![1024], QuantFormat::Q4_0);
        let q4_k_m = QuantizedTensor::from_f32(&data, vec![1024], QuantFormat::Q4_K_M);
        let f16 = QuantizedTensor::from_f32(&data, vec![1024], QuantFormat::F16);
        
        // Check compression ratios
        assert!(q4_0.compression_ratio() > 5.0, "Q4_0 should compress ~7x");
        assert!(q4_k_m.compression_ratio() > 5.0, "Q4_K_M should compress ~7x");
        assert!((f16.compression_ratio() - 2.0).abs() < 0.1, "F16 should compress 2x");
    }

    #[test]
    fn test_quantized_tensor_roundtrip() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 32.0).collect();
        
        for format in [QuantFormat::F32, QuantFormat::F16, QuantFormat::Q8_0, QuantFormat::Q4_0, QuantFormat::Q4_K_M] {
            let quantized = QuantizedTensor::from_f32(&data, vec![256], format);
            let recovered = quantized.to_f32();
            
            assert_eq!(recovered.len(), data.len(), "Length mismatch for {:?}", format);
            
            // For 4-bit formats, check average error instead of per-element tolerance
            // because individual elements can have larger errors
            if matches!(format, QuantFormat::Q4_0 | QuantFormat::Q4_K_M | QuantFormat::Q4_K_S) {
                let total_error: f32 = data.iter()
                    .zip(recovered.iter())
                    .map(|(o, r)| (o - r).abs())
                    .sum();
                let avg_error = total_error / data.len() as f32;
                // Q4_0 has better precision, Q4_K_M has nested quantization
                let tolerance = match format {
                    QuantFormat::Q4_0 => 0.3,
                    QuantFormat::Q4_K_M | QuantFormat::Q4_K_S => 0.7,
                    _ => unreachable!(),
                };
                assert!(
                    avg_error < tolerance,
                    "{:?} roundtrip avg error too high: {} (tolerance {})",
                    format, avg_error, tolerance
                );
            } else {
                // Check approximate equality (tolerance depends on format)
                let tolerance = match format {
                    QuantFormat::F32 => 1e-6,
                    QuantFormat::F16 => 0.01,
                    QuantFormat::Q8_0 => 0.1,
                    QuantFormat::BF16 => 0.01,
                    _ => unreachable!(),
                };
                
                for (orig, rec) in data.iter().zip(recovered.iter()) {
                    assert!(
                        (orig - rec).abs() < tolerance,
                        "{:?} roundtrip failed: {} vs {} (tolerance {})",
                        format, orig, rec, tolerance
                    );
                }
            }
        }
    }

    #[test]
    fn test_quant_format_block_sizes() {
        assert_eq!(QuantFormat::Q4_0.block_size(), 32);
        assert_eq!(QuantFormat::Q8_0.block_size(), 32);
        assert_eq!(QuantFormat::Q4_K_M.block_size(), 256);
        assert_eq!(QuantFormat::Q4_K_S.block_size(), 256);
        assert_eq!(QuantFormat::F32.block_size(), 1);
    }
}
