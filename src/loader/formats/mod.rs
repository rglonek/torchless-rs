//! Model Format Support (Phase 7)
//!
//! This module provides support for multiple model file formats:
//! - **GGUF**: The GGML Universal Format used by llama.cpp and related projects
//! - **Safetensors**: HuggingFace's safe tensor serialization format
//! - **Torchless Binary**: Our native format with JSON header + binary payload
//!
//! Format auto-detection is provided via magic byte inspection.

pub mod gguf;
pub mod safetensors;

use anyhow::{Context, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Supported model file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// Torchless native binary format (JSON header + binary tensors)
    TorchlessBinary,
    /// GGML Universal Format (used by llama.cpp)
    GGUF,
    /// HuggingFace Safetensors format
    Safetensors,
    /// Unknown format
    Unknown,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelFormat::TorchlessBinary => write!(f, "Torchless Binary"),
            ModelFormat::GGUF => write!(f, "GGUF"),
            ModelFormat::Safetensors => write!(f, "Safetensors"),
            ModelFormat::Unknown => write!(f, "Unknown"),
        }
    }
}

/// GGUF magic number: "GGUF" in little-endian (0x46554747)
const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Safetensors files start with a little-endian u64 header size,
/// followed by a JSON header. We check for reasonable header sizes
/// and valid JSON structure.
const SAFETENSORS_MAX_HEADER_CHECK: u64 = 100 * 1024 * 1024; // 100MB max header

/// Detect the format of a model file by inspecting magic bytes
///
/// # Arguments
/// * `path` - Path to the model file
///
/// # Returns
/// The detected `ModelFormat`, or `Unknown` if the format cannot be determined
pub fn detect_format<P: AsRef<Path>>(path: P) -> Result<ModelFormat> {
    let path = path.as_ref();
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open file for format detection: {:?}", path))?;

    let mut header = [0u8; 8];
    let bytes_read = file.read(&mut header)?;

    if bytes_read < 8 {
        return Ok(ModelFormat::Unknown);
    }

    // Check for GGUF magic: "GGUF" (0x47475546)
    if header[0..4] == GGUF_MAGIC {
        return Ok(ModelFormat::GGUF);
    }

    // Check for Safetensors: starts with little-endian u64 header size
    // followed by JSON header starting with '{'
    let header_size = u64::from_le_bytes(header);
    if header_size > 0 && header_size < SAFETENSORS_MAX_HEADER_CHECK {
        // Try to read the first byte of the JSON header
        let mut json_start = [0u8; 1];
        if file.read(&mut json_start).is_ok() && json_start[0] == b'{' {
            return Ok(ModelFormat::Safetensors);
        }
    }

    // Check for Torchless Binary: starts with little-endian u64 header size
    // followed by JSON header (but different structure than safetensors)
    // Torchless has a specific JSON structure with "metadata", "tensors", "tokenizer"
    if header_size > 0 && header_size < SAFETENSORS_MAX_HEADER_CHECK {
        // Reset file position and try to read JSON
        file = File::open(path)?;
        let mut full_header = vec![0u8; 8 + header_size as usize];
        if file.read_exact(&mut full_header).is_ok() {
            let json_bytes = &full_header[8..];
            // Trim null bytes (Torchless uses padding)
            let json_str = std::str::from_utf8(json_bytes)
                .map(|s| s.trim_end_matches('\0'))
                .unwrap_or("");
            
            // Check for Torchless-specific keys
            if json_str.contains("\"metadata\"") && json_str.contains("\"tensors\"") {
                return Ok(ModelFormat::TorchlessBinary);
            }
        }
    }

    Ok(ModelFormat::Unknown)
}

/// Load a model file with automatic format detection
///
/// This function detects the format of the model file and loads it
/// using the appropriate loader.
///
/// # Arguments
/// * `path` - Path to the model file
///
/// # Returns
/// A `UnifiedModelData` containing the loaded model parameters
pub fn load_model_auto<P: AsRef<Path>>(path: P) -> Result<UnifiedModelData> {
    let path = path.as_ref();
    let format = detect_format(path)?;

    match format {
        ModelFormat::TorchlessBinary => {
            let params = super::Parameters::load(path)?;
            Ok(UnifiedModelData::from_torchless(params))
        }
        ModelFormat::GGUF => {
            let loader = gguf::GGUFLoader::load(path)?;
            Ok(UnifiedModelData::from_gguf(loader))
        }
        ModelFormat::Safetensors => {
            let loader = safetensors::SafetensorsLoader::load(path)?;
            Ok(UnifiedModelData::from_safetensors(loader))
        }
        ModelFormat::Unknown => {
            anyhow::bail!(
                "Unknown model format for file: {:?}. \
                Supported formats: Torchless Binary, GGUF, Safetensors",
                path
            )
        }
    }
}

/// Unified model data that can be loaded from any supported format
///
/// This provides a common interface for working with model data
/// regardless of the source format.
#[derive(Debug)]
pub struct UnifiedModelData {
    /// The source format this data was loaded from
    pub format: ModelFormat,
    /// Model configuration/metadata
    pub config: UnifiedConfig,
    /// Tensor names available in the model
    pub tensor_names: Vec<String>,
    /// Internal data source
    inner: ModelDataInner,
}

enum ModelDataInner {
    Torchless(super::Parameters),
    GGUF(gguf::GGUFLoader),
    Safetensors(safetensors::SafetensorsLoader),
}

impl std::fmt::Debug for ModelDataInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelDataInner::Torchless(_) => write!(f, "Torchless(...)"),
            ModelDataInner::GGUF(loader) => write!(f, "GGUF({:?})", loader),
            ModelDataInner::Safetensors(loader) => write!(f, "Safetensors({:?})", loader),
        }
    }
}

/// Unified model configuration
#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    /// Model architecture (e.g., "llama", "mistral", "phi")
    pub architecture: Option<String>,
    /// Hidden size / embedding dimension
    pub hidden_size: Option<usize>,
    /// Intermediate size for MLP layers
    pub intermediate_size: Option<usize>,
    /// Number of layers
    pub n_layers: Option<usize>,
    /// Number of attention heads
    pub n_heads: Option<usize>,
    /// Number of key-value heads (for GQA)
    pub n_kv_heads: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// Maximum sequence length
    pub max_position_embeddings: Option<usize>,
    /// RoPE theta value
    pub rope_theta: Option<f32>,
    /// RMSNorm epsilon
    pub norm_eps: Option<f32>,
    /// Quantization type (if applicable)
    pub quantization: Option<String>,
    /// Additional metadata as key-value pairs
    pub metadata: std::collections::HashMap<String, String>,
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self {
            architecture: None,
            hidden_size: None,
            intermediate_size: None,
            n_layers: None,
            n_heads: None,
            n_kv_heads: None,
            vocab_size: None,
            max_position_embeddings: None,
            rope_theta: None,
            norm_eps: None,
            quantization: None,
            metadata: std::collections::HashMap::new(),
        }
    }
}

impl UnifiedModelData {
    /// Create from Torchless Parameters
    fn from_torchless(params: super::Parameters) -> Self {
        let tensor_names: Vec<String> = params.tensors.keys().cloned().collect();
        let config = UnifiedConfig {
            architecture: Some("mistral".to_string()),
            hidden_size: Some(params.config.hidden_size),
            intermediate_size: Some(params.config.intermediate_size),
            n_layers: Some(params.config.n_layers),
            n_heads: Some(params.config.n_heads),
            n_kv_heads: Some(params.config.n_kv_heads),
            vocab_size: Some(params.config.vocab_size),
            max_position_embeddings: Some(params.config.max_position_embeddings),
            rope_theta: Some(params.config.rope_theta),
            norm_eps: Some(params.config.norm_eps),
            quantization: Some(params.config.quant.clone()),
            metadata: std::collections::HashMap::new(),
        };

        Self {
            format: ModelFormat::TorchlessBinary,
            config,
            tensor_names,
            inner: ModelDataInner::Torchless(params),
        }
    }

    /// Create from GGUF loader
    fn from_gguf(loader: gguf::GGUFLoader) -> Self {
        let tensor_names: Vec<String> = loader.tensor_names().iter().cloned().collect();
        let config = loader.to_unified_config();

        Self {
            format: ModelFormat::GGUF,
            config,
            tensor_names,
            inner: ModelDataInner::GGUF(loader),
        }
    }

    /// Create from Safetensors loader
    fn from_safetensors(loader: safetensors::SafetensorsLoader) -> Self {
        let tensor_names: Vec<String> = loader.tensor_names().iter().cloned().collect();
        let config = loader.to_unified_config();

        Self {
            format: ModelFormat::Safetensors,
            config,
            tensor_names,
            inner: ModelDataInner::Safetensors(loader),
        }
    }

    /// Get tensor data as f32 by name
    pub fn get_tensor(&self, name: &str) -> Result<Vec<f32>> {
        match &self.inner {
            ModelDataInner::Torchless(params) => params.get_tensor(name),
            ModelDataInner::GGUF(loader) => loader.get_tensor_f32(name),
            ModelDataInner::Safetensors(loader) => loader.get_tensor_f32(name),
        }
    }

    /// Get tensor shape by name
    pub fn get_tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        match &self.inner {
            ModelDataInner::Torchless(params) => {
                params.get_tensor_shape(name).map(|s| s.to_vec())
            }
            ModelDataInner::GGUF(loader) => {
                loader.get_tensor_info(name).map(|info| info.shape.clone())
            }
            ModelDataInner::Safetensors(loader) => {
                loader.get_tensor_info(name).map(|info| info.shape.clone())
            }
        }
    }

    /// Check if a tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensor_names.contains(&name.to_string())
    }

    /// Get the underlying Torchless Parameters (if format is Torchless)
    pub fn as_torchless(&self) -> Option<&super::Parameters> {
        match &self.inner {
            ModelDataInner::Torchless(params) => Some(params),
            _ => None,
        }
    }

    /// Get the underlying GGUF loader (if format is GGUF)
    pub fn as_gguf(&self) -> Option<&gguf::GGUFLoader> {
        match &self.inner {
            ModelDataInner::GGUF(loader) => Some(loader),
            _ => None,
        }
    }

    /// Get the underlying Safetensors loader (if format is Safetensors)
    pub fn as_safetensors(&self) -> Option<&safetensors::SafetensorsLoader> {
        match &self.inner {
            ModelDataInner::Safetensors(loader) => Some(loader),
            _ => None,
        }
    }
}

// Re-export key types
pub use gguf::{GGUFLoader, GGUFMetadata, GGUFTensorInfo, GGMLType};
pub use safetensors::{SafetensorsLoader, SafetensorsTensorInfo};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", ModelFormat::GGUF), "GGUF");
        assert_eq!(format!("{}", ModelFormat::Safetensors), "Safetensors");
        assert_eq!(format!("{}", ModelFormat::TorchlessBinary), "Torchless Binary");
    }

    #[test]
    fn test_unified_config_default() {
        let config = UnifiedConfig::default();
        assert!(config.architecture.is_none());
        assert!(config.hidden_size.is_none());
        assert!(config.metadata.is_empty());
    }
}
