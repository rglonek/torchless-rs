use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod formats;
mod parameters;
pub mod quantization;
#[cfg(test)]
mod tests;

pub use parameters::{Parameters, TensorDtype, TensorView};
pub use quantization::{
    Q4KMBlock, Q4KSBlock, Q4_0Block, Q8_0Block, QuantFormat, QuantizedTensor, QK4_0, QK8_0, QK_K,
};

// Phase 7: Format Support
pub use formats::{
    detect_format,
    load_model_auto,
    GGMLType,
    // GGUF format
    GGUFLoader,
    GGUFMetadata,
    GGUFTensorInfo,
    // Format detection and auto-loading
    ModelFormat,
    // Safetensors format
    SafetensorsLoader,
    SafetensorsTensorInfo,
    UnifiedConfig,
    UnifiedModelData,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub act_type: String,
    pub quant: String,
}

#[derive(Debug, Deserialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    #[serde(default)]
    pub scale_offset: Option<usize>,
    #[serde(default)]
    pub scale_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerInfo {
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Header {
    metadata: HashMap<String, String>,
    tensors: HashMap<String, TensorInfo>,
    tokenizer: TokenizerInfo,
}
