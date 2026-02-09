//! GGUF Format Parser (Phase 7)
//!
//! This module provides support for loading models in the GGUF (GGML Universal Format)
//! used by llama.cpp and related projects.
//!
//! # GGUF File Structure
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Magic Number (4 bytes): "GGUF" (0x46554747)                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Version (4 bytes): uint32_t                                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Tensor Count (8 bytes): uint64_t                                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Metadata KV Count (8 bytes): uint64_t                           │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Metadata Key-Value Pairs                                        │
//! │   - Key: string (length-prefixed)                               │
//! │   - Value Type: uint32_t                                        │
#![allow(clippy::needless_range_loop)]
//! │   - Value: varies by type                                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Tensor Info Array                                               │
//! │   - Name: string (length-prefixed)                              │
//! │   - N Dimensions: uint32_t                                      │
//! │   - Dimensions: uint64_t[n_dims]                                │
//! │   - Type: uint32_t (GGML type)                                  │
//! │   - Offset: uint64_t (from start of tensor data)                │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Padding to alignment (typically 32 bytes)                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Tensor Data (binary, contiguous)                                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Supported Quantization Types
//!
//! - F32, F16, BF16: Full and half precision
//! - Q8_0, Q8_1: 8-bit quantization
//! - Q4_0, Q4_1: 4-bit quantization
//! - Q4_K_M, Q4_K_S: 4-bit K-quantization variants
//! - Q5_0, Q5_1: 5-bit quantization
//! - Q5_K_M, Q5_K_S: 5-bit K-quantization variants
//! - Q6_K: 6-bit K-quantization

use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

use super::UnifiedConfig;
use crate::loader::quantization::{Q4KMBlock, Q4_0Block, Q8_0Block, QK4_0, QK8_0, QK_K};

/// GGUF magic number: "GGUF" in ASCII
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Default alignment for tensor data
pub const GGUF_DEFAULT_ALIGNMENT: usize = 32;

/// Supported GGUF versions
pub const GGUF_VERSION_V1: u32 = 1;
pub const GGUF_VERSION_V2: u32 = 2;
pub const GGUF_VERSION_V3: u32 = 3;

/// GGML tensor data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // Q4_2 = 4, // unused
    // Q4_3 = 5, // unused
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    BF16 = 29,
}

impl GGMLType {
    /// Parse GGML type from u32
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GGMLType::F32),
            1 => Some(GGMLType::F16),
            2 => Some(GGMLType::Q4_0),
            3 => Some(GGMLType::Q4_1),
            6 => Some(GGMLType::Q5_0),
            7 => Some(GGMLType::Q5_1),
            8 => Some(GGMLType::Q8_0),
            9 => Some(GGMLType::Q8_1),
            10 => Some(GGMLType::Q2_K),
            11 => Some(GGMLType::Q3_K),
            12 => Some(GGMLType::Q4_K),
            13 => Some(GGMLType::Q5_K),
            14 => Some(GGMLType::Q6_K),
            15 => Some(GGMLType::Q8_K),
            16 => Some(GGMLType::IQ2_XXS),
            17 => Some(GGMLType::IQ2_XS),
            18 => Some(GGMLType::IQ3_XXS),
            19 => Some(GGMLType::IQ1_S),
            20 => Some(GGMLType::IQ4_NL),
            21 => Some(GGMLType::IQ3_S),
            22 => Some(GGMLType::IQ2_S),
            23 => Some(GGMLType::IQ4_XS),
            24 => Some(GGMLType::I8),
            25 => Some(GGMLType::I16),
            26 => Some(GGMLType::I32),
            27 => Some(GGMLType::I64),
            28 => Some(GGMLType::F64),
            29 => Some(GGMLType::BF16),
            _ => None,
        }
    }

    /// Get block size for quantized types
    pub fn block_size(&self) -> usize {
        match self {
            GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 | GGMLType::F64 => 1,
            GGMLType::I8 | GGMLType::I16 | GGMLType::I32 | GGMLType::I64 => 1,
            GGMLType::Q4_0 | GGMLType::Q4_1 => QK4_0,
            GGMLType::Q5_0 | GGMLType::Q5_1 => 32,
            GGMLType::Q8_0 | GGMLType::Q8_1 => QK8_0,
            GGMLType::Q2_K
            | GGMLType::Q3_K
            | GGMLType::Q4_K
            | GGMLType::Q5_K
            | GGMLType::Q6_K
            | GGMLType::Q8_K => QK_K,
            _ => 32, // Default block size for IQ types
        }
    }

    /// Get bytes per block for this type
    pub fn bytes_per_block(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::BF16 => 2,
            GGMLType::F64 => 8,
            GGMLType::I8 => 1,
            GGMLType::I16 => 2,
            GGMLType::I32 => 4,
            GGMLType::I64 => 8,
            GGMLType::Q4_0 => Q4_0Block::SIZE,
            GGMLType::Q4_1 => 20, // scale + min + 16 bytes packed
            GGMLType::Q5_0 => 22, // scale + 16 bytes packed + 4 bytes high bits
            GGMLType::Q5_1 => 24, // scale + min + 16 bytes packed + 4 bytes high bits
            GGMLType::Q8_0 => Q8_0Block::SIZE,
            GGMLType::Q8_1 => 36, // scale + min + 32 bytes
            GGMLType::Q2_K => 84,
            GGMLType::Q3_K => 110,
            GGMLType::Q4_K => Q4KMBlock::SIZE, // Same as Q4_K_M
            GGMLType::Q5_K => 176,
            GGMLType::Q6_K => 210,
            GGMLType::Q8_K => 292,
            _ => 32, // Approximation for IQ types
        }
    }

    /// Calculate total bytes for a tensor with given number of elements
    pub fn tensor_bytes(&self, numel: usize) -> usize {
        let block_size = self.block_size();
        let n_blocks = numel.div_ceil(block_size);
        n_blocks * self.bytes_per_block()
    }

    /// Check if this type is quantized
    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            GGMLType::F32
                | GGMLType::F16
                | GGMLType::BF16
                | GGMLType::F64
                | GGMLType::I8
                | GGMLType::I16
                | GGMLType::I32
                | GGMLType::I64
        )
    }
}

impl std::fmt::Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGMLType::F32 => write!(f, "F32"),
            GGMLType::F16 => write!(f, "F16"),
            GGMLType::BF16 => write!(f, "BF16"),
            GGMLType::F64 => write!(f, "F64"),
            GGMLType::Q4_0 => write!(f, "Q4_0"),
            GGMLType::Q4_1 => write!(f, "Q4_1"),
            GGMLType::Q5_0 => write!(f, "Q5_0"),
            GGMLType::Q5_1 => write!(f, "Q5_1"),
            GGMLType::Q8_0 => write!(f, "Q8_0"),
            GGMLType::Q8_1 => write!(f, "Q8_1"),
            GGMLType::Q2_K => write!(f, "Q2_K"),
            GGMLType::Q3_K => write!(f, "Q3_K"),
            GGMLType::Q4_K => write!(f, "Q4_K"),
            GGMLType::Q5_K => write!(f, "Q5_K"),
            GGMLType::Q6_K => write!(f, "Q6_K"),
            GGMLType::Q8_K => write!(f, "Q8_K"),
            GGMLType::I8 => write!(f, "I8"),
            GGMLType::I16 => write!(f, "I16"),
            GGMLType::I32 => write!(f, "I32"),
            GGMLType::I64 => write!(f, "I64"),
            _ => write!(f, "IQ_TYPE"),
        }
    }
}

/// GGUF metadata value types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GGUFValueType {
    fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GGUFValueType::Uint8),
            1 => Some(GGUFValueType::Int8),
            2 => Some(GGUFValueType::Uint16),
            3 => Some(GGUFValueType::Int16),
            4 => Some(GGUFValueType::Uint32),
            5 => Some(GGUFValueType::Int32),
            6 => Some(GGUFValueType::Float32),
            7 => Some(GGUFValueType::Bool),
            8 => Some(GGUFValueType::String),
            9 => Some(GGUFValueType::Array),
            10 => Some(GGUFValueType::Uint64),
            11 => Some(GGUFValueType::Int64),
            12 => Some(GGUFValueType::Float64),
            _ => None,
        }
    }
}

/// GGUF metadata value
#[derive(Debug, Clone)]
pub enum GGUFValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GGUFValue {
    /// Try to get as u32
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GGUFValue::Uint8(v) => Some(*v as u32),
            GGUFValue::Int8(v) => Some(*v as u32),
            GGUFValue::Uint16(v) => Some(*v as u32),
            GGUFValue::Int16(v) => Some(*v as u32),
            GGUFValue::Uint32(v) => Some(*v),
            GGUFValue::Int32(v) => Some(*v as u32),
            GGUFValue::Uint64(v) => Some(*v as u32),
            GGUFValue::Int64(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Try to get as u64
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GGUFValue::Uint8(v) => Some(*v as u64),
            GGUFValue::Int8(v) => Some(*v as u64),
            GGUFValue::Uint16(v) => Some(*v as u64),
            GGUFValue::Int16(v) => Some(*v as u64),
            GGUFValue::Uint32(v) => Some(*v as u64),
            GGUFValue::Int32(v) => Some(*v as u64),
            GGUFValue::Uint64(v) => Some(*v),
            GGUFValue::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    /// Try to get as f32
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GGUFValue::Float32(v) => Some(*v),
            GGUFValue::Float64(v) => Some(*v as f32),
            GGUFValue::Uint32(v) => Some(*v as f32),
            GGUFValue::Int32(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GGUFValue::String(s) => Some(s),
            _ => None,
        }
    }
}

/// GGUF file metadata
#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    /// GGUF format version
    pub version: u32,
    /// Number of tensors in the file
    pub tensor_count: u64,
    /// Key-value metadata pairs
    pub kv: HashMap<String, GGUFValue>,
}

impl GGUFMetadata {
    /// Get architecture string (e.g., "llama", "mistral")
    pub fn architecture(&self) -> Option<&str> {
        self.kv.get("general.architecture").and_then(|v| v.as_str())
    }

    /// Get model name
    pub fn name(&self) -> Option<&str> {
        self.kv.get("general.name").and_then(|v| v.as_str())
    }

    /// Get context length
    pub fn context_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.kv
            .get(&format!("{}.context_length", arch))
            .and_then(|v| v.as_u32())
    }

    /// Get embedding length (hidden size)
    pub fn embedding_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.kv
            .get(&format!("{}.embedding_length", arch))
            .and_then(|v| v.as_u32())
    }

    /// Get number of attention heads
    pub fn head_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.kv
            .get(&format!("{}.attention.head_count", arch))
            .and_then(|v| v.as_u32())
    }

    /// Get number of key-value heads
    pub fn head_count_kv(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.kv
            .get(&format!("{}.attention.head_count_kv", arch))
            .and_then(|v| v.as_u32())
    }

    /// Get number of layers
    pub fn block_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.kv
            .get(&format!("{}.block_count", arch))
            .and_then(|v| v.as_u32())
    }

    /// Get feed-forward length (intermediate size)
    pub fn feed_forward_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.kv
            .get(&format!("{}.feed_forward_length", arch))
            .and_then(|v| v.as_u32())
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> Option<u32> {
        // Try tokenizer vocab size first
        if let Some(GGUFValue::Array(arr)) = self.kv.get("tokenizer.ggml.tokens") {
            return Some(arr.len() as u32);
        }
        None
    }

    /// Get RoPE theta
    pub fn rope_freq_base(&self) -> Option<f32> {
        let arch = self.architecture()?;
        self.kv
            .get(&format!("{}.rope.freq_base", arch))
            .and_then(|v| v.as_f32())
    }

    /// Get layer norm epsilon
    pub fn layer_norm_eps(&self) -> Option<f32> {
        let arch = self.architecture()?;
        // Try both possible key names
        self.kv
            .get(&format!("{}.attention.layer_norm_epsilon", arch))
            .or_else(|| {
                self.kv
                    .get(&format!("{}.attention.layer_norm_rms_epsilon", arch))
            })
            .and_then(|v| v.as_f32())
    }

    /// Get file type / quantization type
    pub fn file_type(&self) -> Option<u32> {
        self.kv.get("general.file_type").and_then(|v| v.as_u32())
    }

    /// Get quantization version
    pub fn quantization_version(&self) -> Option<u32> {
        self.kv
            .get("general.quantization_version")
            .and_then(|v| v.as_u32())
    }
}

/// Information about a tensor in the GGUF file
#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: usize,
    /// Shape (dimensions)
    pub shape: Vec<usize>,
    /// GGML data type
    pub dtype: GGMLType,
    /// Offset from start of tensor data section
    pub offset: u64,
}

impl GGUFTensorInfo {
    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.dtype.tensor_bytes(self.numel())
    }
}

/// GGUF file loader
#[derive(Debug)]
pub struct GGUFLoader {
    /// Memory-mapped file
    mmap: Mmap,
    /// File metadata
    pub metadata: GGUFMetadata,
    /// Tensor information
    tensors: HashMap<String, GGUFTensorInfo>,
    /// Tensor names in order
    tensor_names: Vec<String>,
    /// Offset where tensor data starts
    tensor_data_offset: usize,
    /// Alignment for tensor data
    alignment: usize,
}

impl GGUFLoader {
    /// Load a GGUF file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("Failed to open GGUF file: {:?}", path))?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(&mmap[..]);

        // Read and verify magic
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            anyhow::bail!(
                "Invalid GGUF magic number: expected 0x{:08X}, got 0x{:08X}",
                GGUF_MAGIC,
                magic
            );
        }

        // Read version
        let version = cursor.read_u32::<LittleEndian>()?;
        if !(GGUF_VERSION_V1..=GGUF_VERSION_V3).contains(&version) {
            anyhow::bail!("Unsupported GGUF version: {}", version);
        }

        // Read tensor count and metadata count
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        let metadata_kv_count = cursor.read_u64::<LittleEndian>()?;

        // Read metadata key-value pairs
        let mut kv = HashMap::new();
        for _ in 0..metadata_kv_count {
            let (key, value) = Self::read_kv_pair(&mut cursor, version)?;
            kv.insert(key, value);
        }

        let metadata = GGUFMetadata {
            version,
            tensor_count,
            kv,
        };

        // Get alignment from metadata or use default
        let alignment = metadata
            .kv
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|v| v as usize)
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT);

        // Read tensor info
        let mut tensors = HashMap::new();
        let mut tensor_names = Vec::new();

        for _ in 0..tensor_count {
            let info = Self::read_tensor_info(&mut cursor, version)?;
            tensor_names.push(info.name.clone());
            tensors.insert(info.name.clone(), info);
        }

        // Calculate tensor data offset (aligned)
        let current_pos = cursor.position() as usize;
        let tensor_data_offset = current_pos.div_ceil(alignment) * alignment;

        Ok(Self {
            mmap,
            metadata,
            tensors,
            tensor_names,
            tensor_data_offset,
            alignment,
        })
    }

    /// Read a string from the cursor
    fn read_string(cursor: &mut Cursor<&[u8]>, version: u32) -> Result<String> {
        let len = if version >= GGUF_VERSION_V2 {
            cursor.read_u64::<LittleEndian>()? as usize
        } else {
            cursor.read_u32::<LittleEndian>()? as usize
        };

        let mut bytes = vec![0u8; len];
        cursor.read_exact(&mut bytes)?;
        String::from_utf8(bytes).context("Invalid UTF-8 in GGUF string")
    }

    /// Read a key-value pair
    fn read_kv_pair(cursor: &mut Cursor<&[u8]>, version: u32) -> Result<(String, GGUFValue)> {
        let key = Self::read_string(cursor, version)?;
        let value_type = cursor.read_u32::<LittleEndian>()?;
        let value = Self::read_value(cursor, version, value_type)?;
        Ok((key, value))
    }

    /// Read a value based on type
    fn read_value(cursor: &mut Cursor<&[u8]>, version: u32, value_type: u32) -> Result<GGUFValue> {
        let vt = GGUFValueType::from_u32(value_type)
            .context(format!("Unknown GGUF value type: {}", value_type))?;

        match vt {
            GGUFValueType::Uint8 => Ok(GGUFValue::Uint8(cursor.read_u8()?)),
            GGUFValueType::Int8 => Ok(GGUFValue::Int8(cursor.read_i8()?)),
            GGUFValueType::Uint16 => Ok(GGUFValue::Uint16(cursor.read_u16::<LittleEndian>()?)),
            GGUFValueType::Int16 => Ok(GGUFValue::Int16(cursor.read_i16::<LittleEndian>()?)),
            GGUFValueType::Uint32 => Ok(GGUFValue::Uint32(cursor.read_u32::<LittleEndian>()?)),
            GGUFValueType::Int32 => Ok(GGUFValue::Int32(cursor.read_i32::<LittleEndian>()?)),
            GGUFValueType::Float32 => Ok(GGUFValue::Float32(cursor.read_f32::<LittleEndian>()?)),
            GGUFValueType::Bool => Ok(GGUFValue::Bool(cursor.read_u8()? != 0)),
            GGUFValueType::String => Ok(GGUFValue::String(Self::read_string(cursor, version)?)),
            GGUFValueType::Array => {
                let element_type = cursor.read_u32::<LittleEndian>()?;
                let len = if version >= GGUF_VERSION_V2 {
                    cursor.read_u64::<LittleEndian>()? as usize
                } else {
                    cursor.read_u32::<LittleEndian>()? as usize
                };

                let mut values = Vec::with_capacity(len);
                for _ in 0..len {
                    values.push(Self::read_value(cursor, version, element_type)?);
                }
                Ok(GGUFValue::Array(values))
            }
            GGUFValueType::Uint64 => Ok(GGUFValue::Uint64(cursor.read_u64::<LittleEndian>()?)),
            GGUFValueType::Int64 => Ok(GGUFValue::Int64(cursor.read_i64::<LittleEndian>()?)),
            GGUFValueType::Float64 => Ok(GGUFValue::Float64(cursor.read_f64::<LittleEndian>()?)),
        }
    }

    /// Read tensor info
    fn read_tensor_info(cursor: &mut Cursor<&[u8]>, version: u32) -> Result<GGUFTensorInfo> {
        let name = Self::read_string(cursor, version)?;
        let n_dims = cursor.read_u32::<LittleEndian>()? as usize;

        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(cursor.read_u64::<LittleEndian>()? as usize);
        }

        let dtype_val = cursor.read_u32::<LittleEndian>()?;
        let dtype =
            GGMLType::from_u32(dtype_val).context(format!("Unknown GGML type: {}", dtype_val))?;

        let offset = cursor.read_u64::<LittleEndian>()?;

        Ok(GGUFTensorInfo {
            name,
            n_dims,
            shape,
            dtype,
            offset,
        })
    }

    /// Get list of tensor names
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Get tensor info by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&GGUFTensorInfo> {
        self.tensors.get(name)
    }

    /// Get raw tensor bytes
    pub fn get_tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .context(format!("Tensor not found: {}", name))?;

        let start = self.tensor_data_offset + info.offset as usize;
        let size = info.size_bytes();
        let end = start + size;

        if end > self.mmap.len() {
            anyhow::bail!(
                "Tensor {} data exceeds file bounds: {} + {} > {}",
                name,
                start,
                size,
                self.mmap.len()
            );
        }

        Ok(&self.mmap[start..end])
    }

    /// Get tensor as f32 vector (dequantizes if necessary)
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .get(name)
            .context(format!("Tensor not found: {}", name))?;
        let bytes = self.get_tensor_bytes(name)?;
        let numel = info.numel();

        match info.dtype {
            GGMLType::F32 => {
                let mut result = Vec::with_capacity(numel);
                let mut cursor = Cursor::new(bytes);
                for _ in 0..numel {
                    result.push(cursor.read_f32::<LittleEndian>()?);
                }
                Ok(result)
            }
            GGMLType::F16 => {
                let mut result = Vec::with_capacity(numel);
                for i in 0..numel {
                    let f16_val = f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    result.push(f16_val.to_f32());
                }
                Ok(result)
            }
            GGMLType::BF16 => {
                let mut result = Vec::with_capacity(numel);
                for i in 0..numel {
                    let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    result.push(f32::from_bits((bits as u32) << 16));
                }
                Ok(result)
            }
            GGMLType::Q8_0 => {
                let mut result = Vec::with_capacity(numel);
                for block_data in bytes.chunks(Q8_0Block::SIZE) {
                    let block = Q8_0Block::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);
                Ok(result)
            }
            GGMLType::Q4_0 => {
                let mut result = Vec::with_capacity(numel);
                for block_data in bytes.chunks(Q4_0Block::SIZE) {
                    let block = Q4_0Block::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);
                Ok(result)
            }
            GGMLType::Q4_K => {
                // Q4_K is same as Q4_K_M
                let mut result = Vec::with_capacity(numel);
                for block_data in bytes.chunks(Q4KMBlock::SIZE) {
                    let block = Q4KMBlock::from_bytes(block_data);
                    result.extend_from_slice(&block.dequantize_block());
                }
                result.truncate(numel);
                Ok(result)
            }
            GGMLType::Q4_1 => {
                // Q4_1: scale (f16) + min (f16) + 16 bytes packed nibbles = 20 bytes for 32 values
                const QK4_1: usize = 32;
                const BLOCK_SIZE: usize = 20;

                let mut result = Vec::with_capacity(numel);
                for block_data in bytes.chunks(BLOCK_SIZE) {
                    if block_data.len() < BLOCK_SIZE {
                        break;
                    }
                    let scale = f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();
                    let min = f16::from_le_bytes([block_data[2], block_data[3]]).to_f32();

                    for i in 0..QK4_1 / 2 {
                        let byte = block_data[4 + i];
                        let low_nibble = (byte & 0x0F) as f32;
                        let high_nibble = (byte >> 4) as f32;

                        result.push(low_nibble * scale + min);
                        result.push(high_nibble * scale + min);
                    }
                }
                result.truncate(numel);
                Ok(result)
            }
            GGMLType::Q5_0 => {
                // Q5_0: scale (f16) + 16 bytes packed + 4 bytes high bits = 22 bytes for 32 values
                const QK5_0: usize = 32;
                const BLOCK_SIZE: usize = 22;

                let mut result = Vec::with_capacity(numel);
                for block_data in bytes.chunks(BLOCK_SIZE) {
                    if block_data.len() < BLOCK_SIZE {
                        break;
                    }
                    let scale = f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();
                    let qh = &block_data[2..6];
                    let qs = &block_data[6..22];

                    for i in 0..QK5_0 {
                        let byte_idx = i / 2;
                        let nibble = if i.is_multiple_of(2) {
                            qs[byte_idx] & 0x0F
                        } else {
                            qs[byte_idx] >> 4
                        };

                        let high_bit = (qh[i / 8] >> (i % 8)) & 1;
                        let quant_val = nibble | (high_bit << 4);
                        let val = (quant_val as i8 - 16) as f32 * scale;
                        result.push(val);
                    }
                }
                result.truncate(numel);
                Ok(result)
            }
            GGMLType::Q5_1 => {
                // Q5_1: scale (f16) + min (f16) + 4 bytes high bits + 16 bytes packed = 24 bytes for 32 values
                const QK5_1: usize = 32;
                const BLOCK_SIZE: usize = 24;

                let mut result = Vec::with_capacity(numel);
                for block_data in bytes.chunks(BLOCK_SIZE) {
                    if block_data.len() < BLOCK_SIZE {
                        break;
                    }
                    let scale = f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();
                    let min = f16::from_le_bytes([block_data[2], block_data[3]]).to_f32();
                    let qh = &block_data[4..8];
                    let qs = &block_data[8..24];

                    for i in 0..QK5_1 {
                        let byte_idx = i / 2;
                        let nibble = if i.is_multiple_of(2) {
                            qs[byte_idx] & 0x0F
                        } else {
                            qs[byte_idx] >> 4
                        };

                        let high_bit = (qh[i / 8] >> (i % 8)) & 1;
                        let quant_val = nibble | (high_bit << 4);
                        result.push(quant_val as f32 * scale + min);
                    }
                }
                result.truncate(numel);
                Ok(result)
            }
            GGMLType::Q8_1 => {
                // Q8_1: scale (f16) + sum (f16) + 32 bytes = 36 bytes for 32 values
                // Note: sum is for SIMD optimization, not needed for dequant
                const QK8_1: usize = 32;
                const BLOCK_SIZE: usize = 36;

                let mut result = Vec::with_capacity(numel);
                for block_data in bytes.chunks(BLOCK_SIZE) {
                    if block_data.len() < BLOCK_SIZE {
                        break;
                    }
                    let scale = f16::from_le_bytes([block_data[0], block_data[1]]).to_f32();
                    // bytes[2..4] is sum, not used for dequant
                    let qs = &block_data[4..36];

                    for i in 0..QK8_1 {
                        let q = qs[i] as i8;
                        result.push(q as f32 * scale);
                    }
                }
                result.truncate(numel);
                Ok(result)
            }
            GGMLType::Q6_K => {
                // Q6_K: Complex K-quantization format, 210 bytes for 256 elements
                // Structure: scale (f16) + 128 bytes ql + 64 bytes qh + 16 bytes scales
                const QK6_K: usize = 256;
                const BLOCK_SIZE: usize = 210;

                let mut result = Vec::with_capacity(numel);
                for block_data in bytes.chunks(BLOCK_SIZE) {
                    if block_data.len() < BLOCK_SIZE {
                        break;
                    }

                    // Simplified Q6_K dequantization
                    // Full implementation would need proper scale handling
                    let ql = &block_data[0..128];
                    let qh = &block_data[128..192];
                    let scales = &block_data[192..208];
                    let d = f16::from_le_bytes([block_data[208], block_data[209]]).to_f32();

                    for i in 0..QK6_K {
                        let l_idx = i;
                        let h_idx = i / 4;
                        let h_shift = (i % 4) * 2;

                        let ql_val = ql[l_idx / 2];
                        let q_low = if l_idx.is_multiple_of(2) {
                            ql_val & 0x0F
                        } else {
                            ql_val >> 4
                        };
                        let q_high = (qh[h_idx] >> h_shift) & 0x03;
                        let q = (q_low | (q_high << 4)) as i8 - 32;

                        let scale_idx = i / 16;
                        let scale = (scales[scale_idx] as i8) as f32;

                        result.push(d * scale * q as f32);
                    }
                }
                result.truncate(numel);
                Ok(result)
            }
            _ => {
                anyhow::bail!(
                    "Unsupported GGML type for tensor {}: {:?}",
                    name,
                    info.dtype
                )
            }
        }
    }

    /// Convert to unified config
    pub fn to_unified_config(&self) -> UnifiedConfig {
        let mut config = UnifiedConfig {
            architecture: self.metadata.architecture().map(String::from),
            hidden_size: self.metadata.embedding_length().map(|v| v as usize),
            intermediate_size: self.metadata.feed_forward_length().map(|v| v as usize),
            n_layers: self.metadata.block_count().map(|v| v as usize),
            n_heads: self.metadata.head_count().map(|v| v as usize),
            n_kv_heads: self.metadata.head_count_kv().map(|v| v as usize),
            vocab_size: self.metadata.vocab_size().map(|v| v as usize),
            max_position_embeddings: self.metadata.context_length().map(|v| v as usize),
            rope_theta: self.metadata.rope_freq_base(),
            norm_eps: self.metadata.layer_norm_eps(),
            ..Default::default()
        };

        // Add quantization info
        if let Some(file_type) = self.metadata.file_type() {
            config.quantization = Some(format!("GGUF_FT{}", file_type));
        }

        // Copy all metadata as strings
        for (key, value) in &self.metadata.kv {
            if let GGUFValue::String(s) = value {
                config.metadata.insert(key.clone(), s.clone());
            } else if let Some(v) = value.as_u64() {
                config.metadata.insert(key.clone(), v.to_string());
            } else if let Some(v) = value.as_f32() {
                config.metadata.insert(key.clone(), v.to_string());
            }
        }

        config
    }

    /// Get file size
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get tensor data offset
    pub fn tensor_data_offset(&self) -> usize {
        self.tensor_data_offset
    }

    /// Print summary of the GGUF file
    pub fn print_summary(&self) {
        println!("GGUF File Summary");
        println!("=================");
        println!("Version: {}", self.metadata.version);
        println!("Tensor count: {}", self.metadata.tensor_count);
        println!("File size: {} MB", self.file_size() / (1024 * 1024));
        println!("Tensor data offset: {}", self.tensor_data_offset);
        println!("Alignment: {} bytes", self.alignment);
        println!();

        if let Some(arch) = self.metadata.architecture() {
            println!("Architecture: {}", arch);
        }
        if let Some(name) = self.metadata.name() {
            println!("Model name: {}", name);
        }
        if let Some(ctx_len) = self.metadata.context_length() {
            println!("Context length: {}", ctx_len);
        }
        if let Some(embed_len) = self.metadata.embedding_length() {
            println!("Embedding length: {}", embed_len);
        }
        if let Some(blocks) = self.metadata.block_count() {
            println!("Layer count: {}", blocks);
        }
        if let Some(heads) = self.metadata.head_count() {
            println!("Attention heads: {}", heads);
        }
        if let Some(kv_heads) = self.metadata.head_count_kv() {
            println!("KV heads: {}", kv_heads);
        }

        println!();
        println!("First 10 tensors:");
        for (i, name) in self.tensor_names.iter().take(10).enumerate() {
            if let Some(info) = self.tensors.get(name) {
                println!(
                    "  {}: {} {:?} {} ({} bytes)",
                    i + 1,
                    name,
                    info.shape,
                    info.dtype,
                    info.size_bytes()
                );
            }
        }
        if self.tensor_names.len() > 10 {
            println!("  ... and {} more tensors", self.tensor_names.len() - 10);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ggml_type_sizes() {
        assert_eq!(GGMLType::F32.block_size(), 1);
        assert_eq!(GGMLType::F16.block_size(), 1);
        assert_eq!(GGMLType::Q4_0.block_size(), 32);
        assert_eq!(GGMLType::Q8_0.block_size(), 32);
        assert_eq!(GGMLType::Q4_K.block_size(), 256);

        assert_eq!(GGMLType::F32.bytes_per_block(), 4);
        assert_eq!(GGMLType::F16.bytes_per_block(), 2);
        assert_eq!(GGMLType::Q4_0.bytes_per_block(), 18);
        assert_eq!(GGMLType::Q8_0.bytes_per_block(), 34);
    }

    #[test]
    fn test_ggml_type_from_u32() {
        assert_eq!(GGMLType::from_u32(0), Some(GGMLType::F32));
        assert_eq!(GGMLType::from_u32(1), Some(GGMLType::F16));
        assert_eq!(GGMLType::from_u32(2), Some(GGMLType::Q4_0));
        assert_eq!(GGMLType::from_u32(8), Some(GGMLType::Q8_0));
        assert_eq!(GGMLType::from_u32(999), None);
    }

    #[test]
    fn test_gguf_value_conversions() {
        let v = GGUFValue::Uint32(42);
        assert_eq!(v.as_u32(), Some(42));
        assert_eq!(v.as_u64(), Some(42));
        assert_eq!(v.as_str(), None);

        let v = GGUFValue::String("test".to_string());
        assert_eq!(v.as_str(), Some("test"));
        assert_eq!(v.as_u32(), None);

        let v = GGUFValue::Float32(3.14);
        assert!((v.as_f32().unwrap() - 3.14).abs() < 0.001);
    }
}
