use super::{Config, Header, TensorInfo};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

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
}
