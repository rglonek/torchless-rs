use super::{Config, Header, TensorInfo};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

/// Tensor data type enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorDtype {
    F32,
    Int8,
}

/// A lazy view into a tensor stored in the memory-mapped file.
/// This struct holds references to the raw data without copying or dequantizing
/// until the data is actually needed.
#[derive(Debug)]
pub struct TensorView<'a> {
    /// Raw data bytes (f32 as bytes or int8 values)
    pub data: &'a [u8],
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: TensorDtype,
    /// Quantization scales (only for int8 tensors)
    pub scales: Option<&'a [u8]>,
    /// Number of elements per quantization group (typically 64)
    pub group_size: usize,
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
    /// For f32 tensors, this just copies the row.
    /// For int8 tensors, this dequantizes on-the-fly.
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
            TensorDtype::Int8 => {
                // Dequantize int8 values using scales
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
            TensorDtype::Int8 => {
                // Fused dequantize + dot product
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
            "int8" => {
                // Read int8 and dequantize
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
            "int8" => {
                // View int8 data and scales
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
            _ => anyhow::bail!("Unsupported dtype: {}", info.dtype),
        }
    }
}
