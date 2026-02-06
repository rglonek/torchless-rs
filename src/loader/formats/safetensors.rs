//! Safetensors Format Support (Phase 7)
//!
//! This module provides support for loading models in the Safetensors format
//! used by HuggingFace and related projects.
//!
//! # Safetensors File Structure
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Header Size (8 bytes): little-endian uint64                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ JSON Header (header_size bytes)                                 │
//! │   {                                                             │
//! │     "__metadata__": { ... },  // Optional metadata              │
//! │     "tensor_name": {                                            │
//! │       "dtype": "F32" | "F16" | "BF16" | ...,                    │
//! │       "shape": [dim1, dim2, ...],                               │
//! │       "data_offsets": [start, end]                              │
//! │     },                                                          │
//! │     ...                                                         │
//! │   }                                                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Tensor Data (binary, contiguous by offset order)                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Supported Data Types
//!
//! - F32: 32-bit floating point
//! - F16: 16-bit floating point (IEEE 754)
//! - BF16: 16-bit brain floating point
//! - F64: 64-bit floating point
//! - I8, I16, I32, I64: Signed integers
//! - U8, U16, U32, U64: Unsigned integers
//! - BOOL: Boolean values

use anyhow::{Context, Result};
use half::f16;
use memmap2::Mmap;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use super::UnifiedConfig;

/// Safetensors data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SafetensorsDtype {
    /// Boolean values (1 byte per element)
    Bool,
    /// Unsigned 8-bit integer
    U8,
    /// Signed 8-bit integer
    I8,
    /// Unsigned 16-bit integer
    U16,
    /// Signed 16-bit integer
    I16,
    /// 16-bit floating point (IEEE 754)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// Unsigned 32-bit integer
    U32,
    /// Signed 32-bit integer
    I32,
    /// 32-bit floating point
    F32,
    /// Unsigned 64-bit integer
    U64,
    /// Signed 64-bit integer
    I64,
    /// 64-bit floating point
    F64,
}

impl SafetensorsDtype {
    /// Parse dtype from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "BOOL" => Some(SafetensorsDtype::Bool),
            "U8" | "UINT8" => Some(SafetensorsDtype::U8),
            "I8" | "INT8" => Some(SafetensorsDtype::I8),
            "U16" | "UINT16" => Some(SafetensorsDtype::U16),
            "I16" | "INT16" => Some(SafetensorsDtype::I16),
            "F16" | "FLOAT16" => Some(SafetensorsDtype::F16),
            "BF16" | "BFLOAT16" => Some(SafetensorsDtype::BF16),
            "U32" | "UINT32" => Some(SafetensorsDtype::U32),
            "I32" | "INT32" => Some(SafetensorsDtype::I32),
            "F32" | "FLOAT32" => Some(SafetensorsDtype::F32),
            "U64" | "UINT64" => Some(SafetensorsDtype::U64),
            "I64" | "INT64" => Some(SafetensorsDtype::I64),
            "F64" | "FLOAT64" => Some(SafetensorsDtype::F64),
            _ => None,
        }
    }

    /// Get bytes per element for this dtype
    pub fn bytes_per_element(&self) -> usize {
        match self {
            SafetensorsDtype::Bool | SafetensorsDtype::U8 | SafetensorsDtype::I8 => 1,
            SafetensorsDtype::U16 | SafetensorsDtype::I16 | SafetensorsDtype::F16 | SafetensorsDtype::BF16 => 2,
            SafetensorsDtype::U32 | SafetensorsDtype::I32 | SafetensorsDtype::F32 => 4,
            SafetensorsDtype::U64 | SafetensorsDtype::I64 | SafetensorsDtype::F64 => 8,
        }
    }

    /// Check if this dtype is floating point
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            SafetensorsDtype::F16 | SafetensorsDtype::BF16 | SafetensorsDtype::F32 | SafetensorsDtype::F64
        )
    }
}

impl std::fmt::Display for SafetensorsDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafetensorsDtype::Bool => write!(f, "BOOL"),
            SafetensorsDtype::U8 => write!(f, "U8"),
            SafetensorsDtype::I8 => write!(f, "I8"),
            SafetensorsDtype::U16 => write!(f, "U16"),
            SafetensorsDtype::I16 => write!(f, "I16"),
            SafetensorsDtype::F16 => write!(f, "F16"),
            SafetensorsDtype::BF16 => write!(f, "BF16"),
            SafetensorsDtype::U32 => write!(f, "U32"),
            SafetensorsDtype::I32 => write!(f, "I32"),
            SafetensorsDtype::F32 => write!(f, "F32"),
            SafetensorsDtype::U64 => write!(f, "U64"),
            SafetensorsDtype::I64 => write!(f, "I64"),
            SafetensorsDtype::F64 => write!(f, "F64"),
        }
    }
}

/// Information about a tensor in the Safetensors file
#[derive(Debug, Clone)]
pub struct SafetensorsTensorInfo {
    /// Tensor name
    pub name: String,
    /// Data type
    pub dtype: SafetensorsDtype,
    /// Shape (dimensions)
    pub shape: Vec<usize>,
    /// Start offset in the data section
    pub start_offset: usize,
    /// End offset in the data section
    pub end_offset: usize,
}

impl SafetensorsTensorInfo {
    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.end_offset - self.start_offset
    }
}

/// Safetensors file loader
#[derive(Debug)]
pub struct SafetensorsLoader {
    /// Memory-mapped file
    mmap: Mmap,
    /// Tensor information
    tensors: HashMap<String, SafetensorsTensorInfo>,
    /// Tensor names in order (sorted by offset)
    tensor_names: Vec<String>,
    /// Metadata key-value pairs
    pub metadata: HashMap<String, String>,
    /// Offset where tensor data starts (after header)
    data_offset: usize,
}

/// Internal struct for deserializing tensor info from JSON
#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

impl SafetensorsLoader {
    /// Load a Safetensors file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open Safetensors file: {:?}", path))?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            anyhow::bail!("File too small to be a valid Safetensors file");
        }

        // Read header size (little-endian u64)
        let header_size = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3],
            mmap[4], mmap[5], mmap[6], mmap[7],
        ]) as usize;

        let data_offset = 8 + header_size;

        if data_offset > mmap.len() {
            anyhow::bail!(
                "Invalid header size: {} (file size: {})",
                header_size,
                mmap.len()
            );
        }

        // Parse JSON header
        let header_bytes = &mmap[8..data_offset];
        let header: Value = serde_json::from_slice(header_bytes)
            .context("Failed to parse Safetensors JSON header")?;

        let header_obj = header.as_object()
            .context("Safetensors header must be a JSON object")?;

        let mut tensors = HashMap::new();
        let mut tensor_names = Vec::new();
        let mut metadata = HashMap::new();

        for (key, value) in header_obj {
            if key == "__metadata__" {
                // Parse metadata
                if let Some(meta_obj) = value.as_object() {
                    for (mk, mv) in meta_obj {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        } else {
                            metadata.insert(mk.clone(), mv.to_string());
                        }
                    }
                }
            } else {
                // Parse tensor info
                let raw_info: RawTensorInfo = serde_json::from_value(value.clone())
                    .with_context(|| format!("Failed to parse tensor info for '{}'", key))?;

                let dtype = SafetensorsDtype::from_str(&raw_info.dtype)
                    .with_context(|| format!("Unknown dtype '{}' for tensor '{}'", raw_info.dtype, key))?;

                let info = SafetensorsTensorInfo {
                    name: key.clone(),
                    dtype,
                    shape: raw_info.shape,
                    start_offset: raw_info.data_offsets[0],
                    end_offset: raw_info.data_offsets[1],
                };

                tensor_names.push(key.clone());
                tensors.insert(key.clone(), info);
            }
        }

        // Sort tensor names by offset for deterministic ordering
        tensor_names.sort_by_key(|name| {
            tensors.get(name).map(|t| t.start_offset).unwrap_or(0)
        });

        Ok(Self {
            mmap,
            tensors,
            tensor_names,
            metadata,
            data_offset,
        })
    }

    /// Load multiple Safetensors files (e.g., sharded model)
    pub fn load_sharded<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        if paths.is_empty() {
            anyhow::bail!("No Safetensors files provided");
        }

        if paths.len() == 1 {
            return Self::load(&paths[0]);
        }

        // For sharded models, load the first file as base
        // and merge tensor info from other files
        // Note: This is a simplified implementation that requires all shards
        // to have non-overlapping tensor names
        
        let mut base = Self::load(&paths[0])?;

        for path in &paths[1..] {
            let shard = Self::load(path)?;
            
            // Merge tensor names
            for name in shard.tensor_names {
                if !base.tensors.contains_key(&name) {
                    base.tensor_names.push(name.clone());
                }
            }
            
            // Note: In a full implementation, we would need to handle
            // different file mmaps for different shards
            // For now, we just warn that sharded loading is limited
            eprintln!("Warning: Sharded Safetensors loading is limited. Consider using a merged model.");
        }

        Ok(base)
    }

    /// Get list of tensor names
    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    /// Get tensor info by name
    pub fn get_tensor_info(&self, name: &str) -> Option<&SafetensorsTensorInfo> {
        self.tensors.get(name)
    }

    /// Check if a tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Get raw tensor bytes
    pub fn get_tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self.tensors.get(name)
            .context(format!("Tensor not found: {}", name))?;

        let start = self.data_offset + info.start_offset;
        let end = self.data_offset + info.end_offset;

        if end > self.mmap.len() {
            anyhow::bail!(
                "Tensor {} data exceeds file bounds: {} + {} > {}",
                name,
                start,
                info.size_bytes(),
                self.mmap.len()
            );
        }

        Ok(&self.mmap[start..end])
    }

    /// Get tensor as f32 vector
    pub fn get_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self.tensors.get(name)
            .context(format!("Tensor not found: {}", name))?;
        let bytes = self.get_tensor_bytes(name)?;
        let numel = info.numel();

        match info.dtype {
            SafetensorsDtype::F32 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(4) {
                    result.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                Ok(result)
            }
            SafetensorsDtype::F16 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(2) {
                    let f16_val = f16::from_le_bytes([chunk[0], chunk[1]]);
                    result.push(f16_val.to_f32());
                }
                Ok(result)
            }
            SafetensorsDtype::BF16 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    result.push(f32::from_bits((bits as u32) << 16));
                }
                Ok(result)
            }
            SafetensorsDtype::F64 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(8) {
                    let val = f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ]);
                    result.push(val as f32);
                }
                Ok(result)
            }
            SafetensorsDtype::I8 => {
                let mut result = Vec::with_capacity(numel);
                for &byte in bytes {
                    result.push((byte as i8) as f32);
                }
                Ok(result)
            }
            SafetensorsDtype::U8 => {
                let mut result = Vec::with_capacity(numel);
                for &byte in bytes {
                    result.push(byte as f32);
                }
                Ok(result)
            }
            SafetensorsDtype::I16 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(2) {
                    let val = i16::from_le_bytes([chunk[0], chunk[1]]);
                    result.push(val as f32);
                }
                Ok(result)
            }
            SafetensorsDtype::U16 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(2) {
                    let val = u16::from_le_bytes([chunk[0], chunk[1]]);
                    result.push(val as f32);
                }
                Ok(result)
            }
            SafetensorsDtype::I32 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(4) {
                    let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    result.push(val as f32);
                }
                Ok(result)
            }
            SafetensorsDtype::U32 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(4) {
                    let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    result.push(val as f32);
                }
                Ok(result)
            }
            SafetensorsDtype::I64 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(8) {
                    let val = i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ]);
                    result.push(val as f32);
                }
                Ok(result)
            }
            SafetensorsDtype::U64 => {
                let mut result = Vec::with_capacity(numel);
                for chunk in bytes.chunks_exact(8) {
                    let val = u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7],
                    ]);
                    result.push(val as f32);
                }
                Ok(result)
            }
            SafetensorsDtype::Bool => {
                let mut result = Vec::with_capacity(numel);
                for &byte in bytes {
                    result.push(if byte != 0 { 1.0 } else { 0.0 });
                }
                Ok(result)
            }
        }
    }

    /// Convert to unified config
    ///
    /// Note: Safetensors files typically don't contain model configuration.
    /// Configuration is usually stored in a separate config.json file.
    /// This function attempts to infer some information from tensor shapes.
    pub fn to_unified_config(&self) -> UnifiedConfig {
        let mut config = UnifiedConfig::default();

        // Copy metadata
        config.metadata = self.metadata.clone();

        // Try to infer architecture from metadata
        if let Some(arch) = self.metadata.get("model_type") {
            config.architecture = Some(arch.clone());
        }

        // Try to infer model dimensions from tensor shapes
        // Look for common tensor patterns

        // Try to find embedding dimension from embed_tokens or similar
        for name in &self.tensor_names {
            if let Some(info) = self.tensors.get(name) {
                let name_lower = name.to_lowercase();
                
                // Embedding tensor (usually [vocab_size, hidden_size])
                if (name_lower.contains("embed") || name_lower.contains("wte")) && info.shape.len() == 2 {
                    config.vocab_size = Some(info.shape[0]);
                    config.hidden_size = Some(info.shape[1]);
                }
                
                // MLP tensors can reveal intermediate size
                if (name_lower.contains("mlp") || name_lower.contains("ffn")) && 
                   (name_lower.contains("up") || name_lower.contains("gate") || name_lower.contains("fc1")) &&
                   info.shape.len() == 2 {
                    // For up/gate projection: [intermediate_size, hidden_size]
                    config.intermediate_size = Some(info.shape[0]);
                }
                
                // Count layers from tensor names
                if name_lower.contains("layer") || name_lower.contains("block") {
                    // Try to extract layer number
                    for part in name.split('.') {
                        if let Ok(layer_num) = part.parse::<usize>() {
                            let current_max = config.n_layers.unwrap_or(0);
                            if layer_num + 1 > current_max {
                                config.n_layers = Some(layer_num + 1);
                            }
                        }
                    }
                }

                // Attention heads from q_proj shape
                if (name_lower.contains("q_proj") || name_lower.contains("query")) && info.shape.len() == 2 {
                    if let Some(hidden) = config.hidden_size {
                        // Assuming head_dim = 64 or 128 (common values)
                        let q_size = info.shape[0];
                        for head_dim in [64, 128, 80, 96] {
                            if q_size % head_dim == 0 {
                                config.n_heads = Some(q_size / head_dim);
                                break;
                            }
                        }
                        
                        // Check for GQA by comparing k_proj size
                        if config.n_heads.is_some() && hidden > 0 {
                            // Will be filled in by k_proj check below
                        }
                    }
                }

                // KV heads from k_proj shape (for GQA detection)
                if (name_lower.contains("k_proj") || name_lower.contains("key")) && 
                   !name_lower.contains("kv") && info.shape.len() == 2 {
                    if let Some(n_heads) = config.n_heads {
                        let k_size = info.shape[0];
                        if let Some(q_tensor) = self.tensor_names.iter()
                            .find(|n| n.to_lowercase().contains("q_proj") || n.to_lowercase().contains("query"))
                            .and_then(|n| self.tensors.get(n)) 
                        {
                            let q_size = q_tensor.shape[0];
                            if k_size < q_size {
                                // GQA: k_proj is smaller than q_proj
                                let head_dim = q_size / n_heads;
                                if k_size % head_dim == 0 {
                                    config.n_kv_heads = Some(k_size / head_dim);
                                }
                            } else {
                                config.n_kv_heads = config.n_heads;
                            }
                        }
                    }
                }
            }
        }

        // Default n_kv_heads to n_heads if not detected (MHA)
        if config.n_kv_heads.is_none() && config.n_heads.is_some() {
            config.n_kv_heads = config.n_heads;
        }

        // Try to get dtype info from first tensor
        if let Some(first_name) = self.tensor_names.first() {
            if let Some(info) = self.tensors.get(first_name) {
                config.quantization = Some(info.dtype.to_string());
            }
        }

        config
    }

    /// Get file size
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Get total tensor data size
    pub fn tensor_data_size(&self) -> usize {
        self.tensors.values().map(|t| t.size_bytes()).sum()
    }

    /// Print summary of the Safetensors file
    pub fn print_summary(&self) {
        println!("Safetensors File Summary");
        println!("========================");
        println!("File size: {} MB", self.file_size() / (1024 * 1024));
        println!("Header size: {} bytes", self.data_offset - 8);
        println!("Tensor count: {}", self.tensors.len());
        println!("Tensor data size: {} MB", self.tensor_data_size() / (1024 * 1024));
        println!();

        if !self.metadata.is_empty() {
            println!("Metadata:");
            for (key, value) in &self.metadata {
                let display_value = if value.len() > 50 {
                    format!("{}...", &value[..50])
                } else {
                    value.clone()
                };
                println!("  {}: {}", key, display_value);
            }
            println!();
        }

        // Count dtypes
        let mut dtype_counts: HashMap<SafetensorsDtype, usize> = HashMap::new();
        for info in self.tensors.values() {
            *dtype_counts.entry(info.dtype).or_insert(0) += 1;
        }

        println!("Data types:");
        for (dtype, count) in &dtype_counts {
            println!("  {}: {} tensors", dtype, count);
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

/// Helper function to load a Safetensors model along with its config
///
/// This loads both the model weights and the config.json file (if present)
/// from the same directory.
pub fn load_with_config<P: AsRef<Path>>(model_path: P) -> Result<(SafetensorsLoader, Option<Value>)> {
    let model_path = model_path.as_ref();
    let loader = SafetensorsLoader::load(model_path)?;

    // Try to load config.json from the same directory
    let config_path = model_path.parent()
        .map(|p| p.join("config.json"))
        .filter(|p| p.exists());

    let config = if let Some(config_path) = config_path {
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;
        let config: Value = serde_json::from_str(&config_str)
            .with_context(|| format!("Failed to parse config file: {:?}", config_path))?;
        Some(config)
    } else {
        None
    };

    Ok((loader, config))
}

/// Parse a HuggingFace config.json into UnifiedConfig
pub fn parse_hf_config(config: &Value) -> UnifiedConfig {
    let mut unified = UnifiedConfig::default();

    if let Some(obj) = config.as_object() {
        // Model type / architecture
        if let Some(v) = obj.get("model_type").and_then(|v| v.as_str()) {
            unified.architecture = Some(v.to_string());
        }

        // Hidden size
        if let Some(v) = obj.get("hidden_size").and_then(|v| v.as_u64()) {
            unified.hidden_size = Some(v as usize);
        }

        // Intermediate size
        if let Some(v) = obj.get("intermediate_size").and_then(|v| v.as_u64()) {
            unified.intermediate_size = Some(v as usize);
        }

        // Number of layers
        if let Some(v) = obj.get("num_hidden_layers").and_then(|v| v.as_u64()) {
            unified.n_layers = Some(v as usize);
        }

        // Number of attention heads
        if let Some(v) = obj.get("num_attention_heads").and_then(|v| v.as_u64()) {
            unified.n_heads = Some(v as usize);
        }

        // Number of KV heads (for GQA)
        if let Some(v) = obj.get("num_key_value_heads").and_then(|v| v.as_u64()) {
            unified.n_kv_heads = Some(v as usize);
        } else {
            unified.n_kv_heads = unified.n_heads;
        }

        // Vocab size
        if let Some(v) = obj.get("vocab_size").and_then(|v| v.as_u64()) {
            unified.vocab_size = Some(v as usize);
        }

        // Max position embeddings
        if let Some(v) = obj.get("max_position_embeddings").and_then(|v| v.as_u64()) {
            unified.max_position_embeddings = Some(v as usize);
        }

        // RoPE theta
        if let Some(v) = obj.get("rope_theta").and_then(|v| v.as_f64()) {
            unified.rope_theta = Some(v as f32);
        }

        // RMS norm eps
        if let Some(v) = obj.get("rms_norm_eps").and_then(|v| v.as_f64()) {
            unified.norm_eps = Some(v as f32);
        }

        // Layer norm eps (alternative)
        if unified.norm_eps.is_none() {
            if let Some(v) = obj.get("layer_norm_eps").and_then(|v| v.as_f64()) {
                unified.norm_eps = Some(v as f32);
            }
        }

        // Quantization config
        if let Some(quant_config) = obj.get("quantization_config") {
            if let Some(quant_method) = quant_config.get("quant_method").and_then(|v| v.as_str()) {
                unified.quantization = Some(quant_method.to_string());
            }
        }

        // Store original config keys as metadata
        for (key, value) in obj {
            if let Some(s) = value.as_str() {
                unified.metadata.insert(key.clone(), s.to_string());
            } else if let Some(n) = value.as_i64() {
                unified.metadata.insert(key.clone(), n.to_string());
            } else if let Some(n) = value.as_f64() {
                unified.metadata.insert(key.clone(), n.to_string());
            } else if let Some(b) = value.as_bool() {
                unified.metadata.insert(key.clone(), b.to_string());
            }
        }
    }

    unified
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(SafetensorsDtype::from_str("F32"), Some(SafetensorsDtype::F32));
        assert_eq!(SafetensorsDtype::from_str("f16"), Some(SafetensorsDtype::F16));
        assert_eq!(SafetensorsDtype::from_str("BF16"), Some(SafetensorsDtype::BF16));
        assert_eq!(SafetensorsDtype::from_str("I8"), Some(SafetensorsDtype::I8));
        assert_eq!(SafetensorsDtype::from_str("invalid"), None);
    }

    #[test]
    fn test_dtype_bytes() {
        assert_eq!(SafetensorsDtype::F32.bytes_per_element(), 4);
        assert_eq!(SafetensorsDtype::F16.bytes_per_element(), 2);
        assert_eq!(SafetensorsDtype::BF16.bytes_per_element(), 2);
        assert_eq!(SafetensorsDtype::I8.bytes_per_element(), 1);
        assert_eq!(SafetensorsDtype::F64.bytes_per_element(), 8);
    }

    #[test]
    fn test_parse_hf_config() {
        let config_json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-5
        }"#;

        let config: Value = serde_json::from_str(config_json).unwrap();
        let unified = parse_hf_config(&config);

        assert_eq!(unified.architecture, Some("llama".to_string()));
        assert_eq!(unified.hidden_size, Some(4096));
        assert_eq!(unified.intermediate_size, Some(14336));
        assert_eq!(unified.n_layers, Some(32));
        assert_eq!(unified.n_heads, Some(32));
        assert_eq!(unified.n_kv_heads, Some(8));
        assert_eq!(unified.vocab_size, Some(32000));
        assert_eq!(unified.max_position_embeddings, Some(4096));
        assert!((unified.rope_theta.unwrap() - 10000.0).abs() < 0.1);
        assert!((unified.norm_eps.unwrap() - 1e-5).abs() < 1e-7);
    }
}
