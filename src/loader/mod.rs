use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod formats;
mod parameters;
pub mod quantization;
#[cfg(test)]
mod tests;
pub mod weight_matrix;

pub use parameters::{Parameters, TensorDtype, TensorView};
pub use quantization::{
    Q4KMBlock, Q4KSBlock, Q4_0Block, Q8_0Block, QuantFormat, QuantizedTensor, QK4_0, QK8_0, QK_K,
};
pub use weight_matrix::WeightMatrix;

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

    // MoE (Mixture-of-Experts) configuration
    // When n_routed_experts == 0, the model is a standard dense model.
    /// Number of routed experts per MoE layer (0 = dense model)
    #[serde(default)]
    pub n_routed_experts: usize,
    /// Number of experts activated per token (top-k routing)
    #[serde(default)]
    pub n_experts_per_token: usize,
    /// Number of shared experts that always activate (DeepSeek-style)
    #[serde(default)]
    pub n_shared_experts: usize,
    /// Intermediate size for MoE expert FFNs (may differ from dense intermediate_size)
    #[serde(default)]
    pub moe_intermediate_size: usize,
    /// Index of the first layer that uses MoE (layers before this are dense)
    #[serde(default)]
    pub first_moe_layer: usize,

    // GPT-OSS specific configuration
    /// Explicit head dimension (0 = compute from hidden_size / n_heads)
    #[serde(default)]
    pub head_dim: usize,
    /// SwiGLU clamping limit (0.0 = no clamping). GPT-OSS uses 7.0.
    #[serde(default)]
    pub swiglu_limit: f32,
    /// Sliding window size for alternating attention layers (0 = no sliding window)
    #[serde(default)]
    pub attention_sliding_window: usize,
    /// Whether attention projections have bias terms
    #[serde(default)]
    pub attention_bias: bool,
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
