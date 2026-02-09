//! Model Architecture Detection and Registry (Phase 8)
//!
//! This module provides:
//! - Architecture detection from tensor names and config
//! - A unified `Model` trait for polymorphic model handling
//! - Architecture-specific configurations and tensor name mappings
//!
//! # Supported Architectures
//! - **Mistral**: Grouped Query Attention, SwiGLU, sliding window
//! - **LLaMA**: Similar to Mistral, different RoPE scaling options
//! - **Phi-3**: Microsoft's efficient model with parallel attention
//! - **Gemma**: Google's model with GeGLU activation
//! - **Qwen**: Alibaba's multilingual model
//! - **GPT-OSS**: OpenAI's open-weight MoE model with attention sinks

use crate::loader::{Config, UnifiedConfig};
use crate::model::InferenceState;
use std::collections::HashSet;

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    /// Mistral AI models (7B, 8x7B MoE)
    Mistral,
    /// Meta LLaMA models (LLaMA 1/2/3)
    LLaMA,
    /// Microsoft Phi models (Phi-2, Phi-3)
    Phi,
    /// Google Gemma models
    Gemma,
    /// Alibaba Qwen models
    Qwen,
    /// DeepSeek MoE models (DeepSeek-V2, V3, R1)
    DeepSeek,
    /// OpenAI GPT-OSS open-weight MoE models (gpt-oss-120b, gpt-oss-20b)
    GptOss,
    /// Unknown/unsupported architecture
    Unknown,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArchitecture::Mistral => write!(f, "Mistral"),
            ModelArchitecture::LLaMA => write!(f, "LLaMA"),
            ModelArchitecture::Phi => write!(f, "Phi"),
            ModelArchitecture::Gemma => write!(f, "Gemma"),
            ModelArchitecture::Qwen => write!(f, "Qwen"),
            ModelArchitecture::DeepSeek => write!(f, "DeepSeek"),
            ModelArchitecture::GptOss => write!(f, "GPT-OSS"),
            ModelArchitecture::Unknown => write!(f, "Unknown"),
        }
    }
}

#[allow(clippy::should_implement_trait)]
impl ModelArchitecture {
    /// Parse architecture from string (case-insensitive)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "mistral" | "mistralai" => ModelArchitecture::Mistral,
            "llama" | "llama2" | "llama3" | "meta-llama" => ModelArchitecture::LLaMA,
            "phi" | "phi2" | "phi3" | "phi-2" | "phi-3" | "microsoft" => ModelArchitecture::Phi,
            "gemma" | "gemma2" | "google" => ModelArchitecture::Gemma,
            "qwen" | "qwen2" | "alibaba" => ModelArchitecture::Qwen,
            "deepseek" | "deepseek_v2" | "deepseek_v3" | "deepseek-ai" => {
                ModelArchitecture::DeepSeek
            }
            "gpt_oss" | "gptoss" | "gpt-oss" | "gptossforcausallm" => ModelArchitecture::GptOss,
            _ => ModelArchitecture::Unknown,
        }
    }
}

/// Tensor naming patterns for different architectures
#[derive(Debug, Clone)]
pub struct TensorNamePattern {
    /// Embedding weight name pattern
    pub embed_tokens: &'static str,
    /// Final normalization weight pattern
    pub final_norm: &'static str,
    /// LM head weight pattern
    pub lm_head: &'static str,
    /// Layer prefix pattern (use {} for layer index)
    pub layer_prefix: &'static str,
    /// Input layer norm within layer
    pub input_layernorm: &'static str,
    /// Post-attention layer norm within layer
    pub post_attention_layernorm: &'static str,
    /// Q projection
    pub q_proj: &'static str,
    /// K projection
    pub k_proj: &'static str,
    /// V projection
    pub v_proj: &'static str,
    /// O projection
    pub o_proj: &'static str,
    /// Gate projection (MLP)
    pub gate_proj: &'static str,
    /// Up projection (MLP)
    pub up_proj: &'static str,
    /// Down projection (MLP)
    pub down_proj: &'static str,
}

impl Default for TensorNamePattern {
    fn default() -> Self {
        Self::mistral()
    }
}

impl TensorNamePattern {
    /// Mistral/LLaMA naming pattern
    pub fn mistral() -> Self {
        Self {
            embed_tokens: "model.embed_tokens.weight",
            final_norm: "model.norm.weight",
            lm_head: "lm_head.weight",
            layer_prefix: "model.layers.{}",
            input_layernorm: "input_layernorm.weight",
            post_attention_layernorm: "post_attention_layernorm.weight",
            q_proj: "self_attn.q_proj.weight",
            k_proj: "self_attn.k_proj.weight",
            v_proj: "self_attn.v_proj.weight",
            o_proj: "self_attn.o_proj.weight",
            gate_proj: "mlp.gate_proj.weight",
            up_proj: "mlp.up_proj.weight",
            down_proj: "mlp.down_proj.weight",
        }
    }

    /// LLaMA naming pattern (identical to Mistral for transformer weights)
    pub fn llama() -> Self {
        Self::mistral() // Same naming convention
    }

    /// Phi model naming pattern
    pub fn phi() -> Self {
        Self {
            embed_tokens: "model.embed_tokens.weight",
            final_norm: "model.final_layernorm.weight",
            lm_head: "lm_head.weight",
            layer_prefix: "model.layers.{}",
            input_layernorm: "input_layernorm.weight",
            post_attention_layernorm: "post_attention_layernorm.weight",
            q_proj: "self_attn.q_proj.weight",
            k_proj: "self_attn.k_proj.weight",
            v_proj: "self_attn.v_proj.weight",
            o_proj: "self_attn.dense.weight",
            gate_proj: "mlp.gate_up_proj.weight", // Phi uses fused gate+up
            up_proj: "mlp.gate_up_proj.weight",   // Same as gate (fused)
            down_proj: "mlp.down_proj.weight",
        }
    }

    /// Gemma model naming pattern
    pub fn gemma() -> Self {
        Self {
            embed_tokens: "model.embed_tokens.weight",
            final_norm: "model.norm.weight",
            lm_head: "model.embed_tokens.weight", // Gemma ties embeddings
            layer_prefix: "model.layers.{}",
            input_layernorm: "input_layernorm.weight",
            post_attention_layernorm: "post_feedforward_layernorm.weight",
            q_proj: "self_attn.q_proj.weight",
            k_proj: "self_attn.k_proj.weight",
            v_proj: "self_attn.v_proj.weight",
            o_proj: "self_attn.o_proj.weight",
            gate_proj: "mlp.gate_proj.weight",
            up_proj: "mlp.up_proj.weight",
            down_proj: "mlp.down_proj.weight",
        }
    }

    /// Qwen model naming pattern
    pub fn qwen() -> Self {
        Self {
            embed_tokens: "transformer.wte.weight",
            final_norm: "transformer.ln_f.weight",
            lm_head: "lm_head.weight",
            layer_prefix: "transformer.h.{}",
            input_layernorm: "ln_1.weight",
            post_attention_layernorm: "ln_2.weight",
            q_proj: "attn.c_attn.weight", // Qwen fuses Q, K, V
            k_proj: "attn.c_attn.weight",
            v_proj: "attn.c_attn.weight",
            o_proj: "attn.c_proj.weight",
            gate_proj: "mlp.w1.weight",
            up_proj: "mlp.w2.weight",
            down_proj: "mlp.c_proj.weight",
        }
    }

    /// DeepSeek model naming pattern (same as Mistral/LLaMA for attention,
    /// MoE tensors are handled separately by MoeTensorNamePattern)
    pub fn deepseek() -> Self {
        Self {
            embed_tokens: "model.embed_tokens.weight",
            final_norm: "model.norm.weight",
            lm_head: "lm_head.weight",
            layer_prefix: "model.layers.{}",
            input_layernorm: "input_layernorm.weight",
            post_attention_layernorm: "post_attention_layernorm.weight",
            q_proj: "self_attn.q_proj.weight",
            k_proj: "self_attn.k_proj.weight",
            v_proj: "self_attn.v_proj.weight",
            o_proj: "self_attn.o_proj.weight",
            // Dense layers use standard MLP naming; MoE layers use MoeTensorNamePattern
            gate_proj: "mlp.gate_proj.weight",
            up_proj: "mlp.up_proj.weight",
            down_proj: "mlp.down_proj.weight",
        }
    }

    /// GPT-OSS model naming pattern (OpenAI MoE)
    /// Attention tensors follow Mistral/LLaMA convention.
    /// MoE expert tensors use batched format (handled in model file).
    pub fn gpt_oss() -> Self {
        Self {
            embed_tokens: "model.embed_tokens.weight",
            final_norm: "model.norm.weight",
            lm_head: "lm_head.weight",
            layer_prefix: "model.layers.{}",
            input_layernorm: "input_layernorm.weight",
            post_attention_layernorm: "post_attention_layernorm.weight",
            q_proj: "self_attn.q_proj.weight",
            k_proj: "self_attn.k_proj.weight",
            v_proj: "self_attn.v_proj.weight",
            o_proj: "self_attn.o_proj.weight",
            // MoE experts use batched format; these are not used for GPT-OSS MoE layers
            gate_proj: "mlp.gate_proj.weight",
            up_proj: "mlp.up_proj.weight",
            down_proj: "mlp.down_proj.weight",
        }
    }

    /// Get pattern for architecture
    pub fn for_architecture(arch: ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::Mistral => Self::mistral(),
            ModelArchitecture::LLaMA => Self::llama(),
            ModelArchitecture::Phi => Self::phi(),
            ModelArchitecture::Gemma => Self::gemma(),
            ModelArchitecture::Qwen => Self::qwen(),
            ModelArchitecture::DeepSeek => Self::deepseek(),
            ModelArchitecture::GptOss => Self::gpt_oss(),
            ModelArchitecture::Unknown => Self::mistral(), // Default
        }
    }

    /// Format layer tensor name
    pub fn layer_tensor(&self, layer_idx: usize, suffix: &str) -> String {
        format!(
            "{}.{}",
            self.layer_prefix.replace("{}", &layer_idx.to_string()),
            suffix
        )
    }

    /// Get Q projection name for layer
    pub fn q_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.q_proj)
    }

    /// Get K projection name for layer
    pub fn k_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.k_proj)
    }

    /// Get V projection name for layer
    pub fn v_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.v_proj)
    }

    /// Get O projection name for layer
    pub fn o_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.o_proj)
    }

    /// Get input layernorm name for layer
    pub fn input_layernorm_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.input_layernorm)
    }

    /// Get post-attention layernorm name for layer
    pub fn post_attention_layernorm_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.post_attention_layernorm)
    }

    /// Get gate projection name for layer
    pub fn gate_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.gate_proj)
    }

    /// Get up projection name for layer
    pub fn up_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.up_proj)
    }

    /// Get down projection name for layer
    pub fn down_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.down_proj)
    }
}

/// Tensor naming patterns for MoE (Mixture-of-Experts) layers.
/// These are used in addition to `TensorNamePattern` for MoE models.
#[derive(Debug, Clone)]
pub struct MoeTensorNamePattern {
    /// Layer prefix pattern (use {} for layer index) -- same as base
    pub layer_prefix: &'static str,
    /// Router gate weight pattern (within layer)
    pub router_gate: &'static str,
    /// Expert gate projection pattern (use {e} for expert index)
    pub expert_gate_proj: &'static str,
    /// Expert up projection pattern (use {e} for expert index)
    pub expert_up_proj: &'static str,
    /// Expert down projection pattern (use {e} for expert index)
    pub expert_down_proj: &'static str,
    /// Shared expert gate projection (if applicable)
    pub shared_expert_gate_proj: &'static str,
    /// Shared expert up projection (if applicable)
    pub shared_expert_up_proj: &'static str,
    /// Shared expert down projection (if applicable)
    pub shared_expert_down_proj: &'static str,
}

impl MoeTensorNamePattern {
    /// DeepSeek MoE naming pattern
    pub fn deepseek() -> Self {
        Self {
            layer_prefix: "model.layers.{}",
            router_gate: "mlp.gate.weight",
            expert_gate_proj: "mlp.experts.{e}.gate_proj.weight",
            expert_up_proj: "mlp.experts.{e}.up_proj.weight",
            expert_down_proj: "mlp.experts.{e}.down_proj.weight",
            shared_expert_gate_proj: "mlp.shared_experts.gate_proj.weight",
            shared_expert_up_proj: "mlp.shared_experts.up_proj.weight",
            shared_expert_down_proj: "mlp.shared_experts.down_proj.weight",
        }
    }

    /// Format a layer-level tensor name
    fn layer_tensor(&self, layer_idx: usize, suffix: &str) -> String {
        format!(
            "{}.{}",
            self.layer_prefix.replace("{}", &layer_idx.to_string()),
            suffix
        )
    }

    /// Format an expert-level tensor name
    fn expert_tensor(&self, layer_idx: usize, expert_idx: usize, pattern: &str) -> String {
        let suffix = pattern.replace("{e}", &expert_idx.to_string());
        self.layer_tensor(layer_idx, &suffix)
    }

    /// Get router gate weight name for layer
    pub fn router_gate_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.router_gate)
    }

    /// Get expert gate projection name
    pub fn expert_gate_proj_name(&self, layer_idx: usize, expert_idx: usize) -> String {
        self.expert_tensor(layer_idx, expert_idx, self.expert_gate_proj)
    }

    /// Get expert up projection name
    pub fn expert_up_proj_name(&self, layer_idx: usize, expert_idx: usize) -> String {
        self.expert_tensor(layer_idx, expert_idx, self.expert_up_proj)
    }

    /// Get expert down projection name
    pub fn expert_down_proj_name(&self, layer_idx: usize, expert_idx: usize) -> String {
        self.expert_tensor(layer_idx, expert_idx, self.expert_down_proj)
    }

    /// Get shared expert gate projection name
    pub fn shared_expert_gate_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.shared_expert_gate_proj)
    }

    /// Get shared expert up projection name
    pub fn shared_expert_up_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.shared_expert_up_proj)
    }

    /// Get shared expert down projection name
    pub fn shared_expert_down_proj_name(&self, layer_idx: usize) -> String {
        self.layer_tensor(layer_idx, self.shared_expert_down_proj)
    }
}

/// Architecture-specific configuration
#[derive(Debug, Clone)]
pub struct ArchitectureConfig {
    /// The detected architecture
    pub architecture: ModelArchitecture,
    /// Tensor naming pattern
    pub tensor_names: TensorNamePattern,
    /// Whether embeddings are tied to LM head
    pub tie_embeddings: bool,
    /// Whether Q/K/V projections are fused
    pub fused_qkv: bool,
    /// Whether gate/up projections are fused (Phi)
    pub fused_gate_up: bool,
    /// RoPE scaling type
    pub rope_scaling: RopeScaling,
    /// Activation function type
    pub activation: ActivationType,
    /// Normalization type
    pub norm_type: NormType,
    /// Whether to use parallel residuals (Phi)
    pub parallel_residual: bool,
    /// Whether this model uses MoE (Mixture-of-Experts) layers
    pub is_moe: bool,
    /// MoE tensor naming patterns (only used when is_moe is true)
    pub moe_tensor_names: Option<MoeTensorNamePattern>,
    /// Whether attention projections have bias terms (GPT-OSS)
    pub attention_bias: bool,
}

/// RoPE scaling configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeScaling {
    /// No scaling (standard RoPE)
    None,
    /// Linear scaling
    Linear { factor: f32 },
    /// Dynamic NTK scaling
    DynamicNTK {
        factor: f32,
        original_max_position: usize,
    },
    /// YaRN scaling (used in LLaMA 3)
    YaRN {
        factor: f32,
        original_max_position: usize,
    },
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// SiLU (Swish) activation - used in Mistral, LLaMA
    SiLU,
    /// GELU activation
    GELU,
    /// GELU with approximate tanh - used in some Phi models
    GELUTanh,
    /// SwiGLU (SiLU-gated GLU) - most common
    SwiGLU,
    /// GeGLU (GELU-gated GLU) - used in Gemma
    GeGLU,
}

/// Normalization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Root Mean Square Layer Normalization
    RMSNorm,
    /// Standard Layer Normalization
    LayerNorm,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Mistral,
            tensor_names: TensorNamePattern::mistral(),
            tie_embeddings: false,
            fused_qkv: false,
            fused_gate_up: false,
            rope_scaling: RopeScaling::None,
            activation: ActivationType::SwiGLU,
            norm_type: NormType::RMSNorm,
            parallel_residual: false,
            is_moe: false,
            moe_tensor_names: None,
            attention_bias: false,
        }
    }
}

impl ArchitectureConfig {
    /// Create config for Mistral architecture
    pub fn mistral() -> Self {
        Self {
            architecture: ModelArchitecture::Mistral,
            tensor_names: TensorNamePattern::mistral(),
            tie_embeddings: false,
            fused_qkv: false,
            fused_gate_up: false,
            rope_scaling: RopeScaling::None,
            activation: ActivationType::SwiGLU,
            norm_type: NormType::RMSNorm,
            parallel_residual: false,
            is_moe: false,
            moe_tensor_names: None,
            attention_bias: false,
        }
    }

    /// Create config for LLaMA architecture
    pub fn llama() -> Self {
        Self {
            architecture: ModelArchitecture::LLaMA,
            tensor_names: TensorNamePattern::llama(),
            tie_embeddings: false,
            fused_qkv: false,
            fused_gate_up: false,
            rope_scaling: RopeScaling::None, // Can be modified for LLaMA 3
            activation: ActivationType::SwiGLU,
            norm_type: NormType::RMSNorm,
            parallel_residual: false,
            is_moe: false,
            moe_tensor_names: None,
            attention_bias: false,
        }
    }

    /// Create config for LLaMA 3 architecture with RoPE scaling
    pub fn llama3(original_max_position: usize) -> Self {
        Self {
            architecture: ModelArchitecture::LLaMA,
            tensor_names: TensorNamePattern::llama(),
            tie_embeddings: false,
            fused_qkv: false,
            fused_gate_up: false,
            rope_scaling: RopeScaling::DynamicNTK {
                factor: 8.0,
                original_max_position,
            },
            activation: ActivationType::SwiGLU,
            norm_type: NormType::RMSNorm,
            parallel_residual: false,
            is_moe: false,
            moe_tensor_names: None,
            attention_bias: false,
        }
    }

    /// Create config for Phi architecture
    pub fn phi() -> Self {
        Self {
            architecture: ModelArchitecture::Phi,
            tensor_names: TensorNamePattern::phi(),
            tie_embeddings: false,
            fused_qkv: false,
            fused_gate_up: true, // Phi uses fused gate+up projection
            rope_scaling: RopeScaling::None,
            activation: ActivationType::GELUTanh,
            norm_type: NormType::LayerNorm,
            parallel_residual: true, // Phi uses parallel residuals
            is_moe: false,
            moe_tensor_names: None,
            attention_bias: false,
        }
    }

    /// Create config for Gemma architecture
    pub fn gemma() -> Self {
        Self {
            architecture: ModelArchitecture::Gemma,
            tensor_names: TensorNamePattern::gemma(),
            tie_embeddings: true, // Gemma ties embeddings
            fused_qkv: false,
            fused_gate_up: false,
            rope_scaling: RopeScaling::None,
            activation: ActivationType::GeGLU,
            norm_type: NormType::RMSNorm,
            parallel_residual: false,
            is_moe: false,
            moe_tensor_names: None,
            attention_bias: false,
        }
    }

    /// Create config for Qwen architecture
    pub fn qwen() -> Self {
        Self {
            architecture: ModelArchitecture::Qwen,
            tensor_names: TensorNamePattern::qwen(),
            tie_embeddings: false,
            fused_qkv: true, // Qwen fuses Q, K, V
            fused_gate_up: false,
            rope_scaling: RopeScaling::None,
            activation: ActivationType::SwiGLU,
            norm_type: NormType::RMSNorm,
            parallel_residual: false,
            is_moe: false,
            moe_tensor_names: None,
            attention_bias: false,
        }
    }

    /// Create config for DeepSeek MoE architecture
    pub fn deepseek() -> Self {
        Self {
            architecture: ModelArchitecture::DeepSeek,
            tensor_names: TensorNamePattern::deepseek(),
            tie_embeddings: false,
            fused_qkv: false,
            fused_gate_up: false,
            rope_scaling: RopeScaling::None,
            activation: ActivationType::SwiGLU,
            norm_type: NormType::RMSNorm,
            parallel_residual: false,
            is_moe: true,
            moe_tensor_names: Some(MoeTensorNamePattern::deepseek()),
            attention_bias: false,
        }
    }

    /// Create config for GPT-OSS MoE architecture (OpenAI)
    pub fn gpt_oss() -> Self {
        Self {
            architecture: ModelArchitecture::GptOss,
            tensor_names: TensorNamePattern::gpt_oss(),
            tie_embeddings: false,
            fused_qkv: false,
            fused_gate_up: false,
            rope_scaling: RopeScaling::YaRN {
                factor: 32.0,
                original_max_position: 4096,
            },
            activation: ActivationType::SwiGLU,
            norm_type: NormType::RMSNorm,
            parallel_residual: false,
            is_moe: true,
            moe_tensor_names: None, // GPT-OSS uses batched expert format, handled in model file
            attention_bias: true,
        }
    }

    /// Create config for detected architecture
    pub fn for_architecture(arch: ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::Mistral => Self::mistral(),
            ModelArchitecture::LLaMA => Self::llama(),
            ModelArchitecture::Phi => Self::phi(),
            ModelArchitecture::Gemma => Self::gemma(),
            ModelArchitecture::Qwen => Self::qwen(),
            ModelArchitecture::DeepSeek => Self::deepseek(),
            ModelArchitecture::GptOss => Self::gpt_oss(),
            ModelArchitecture::Unknown => Self::mistral(), // Default to Mistral
        }
    }
}

/// Detect model architecture from tensor names
pub fn detect_architecture_from_tensors(tensor_names: &[String]) -> ModelArchitecture {
    let names_set: HashSet<&str> = tensor_names.iter().map(|s| s.as_str()).collect();

    // Check for GPT-OSS patterns (batched experts with MXFP4 blocks + attention sinks)
    // GPT-OSS uses: model.layers.*.mlp.experts.gate_up_proj_blocks and *.self_attn.sinks
    if names_set
        .iter()
        .any(|n| n.contains("self_attn.sinks") || n.contains("gate_up_proj_blocks"))
    {
        return ModelArchitecture::GptOss;
    }

    // Check for DeepSeek/MoE patterns (per-expert tensors in MLP layers)
    // DeepSeek uses: model.layers.*.mlp.experts.*.gate_proj.weight
    if names_set
        .iter()
        .any(|n| n.contains("mlp.experts.") || n.contains("mlp.gate.weight"))
    {
        return ModelArchitecture::DeepSeek;
    }

    // Check for Qwen-specific patterns
    if names_set
        .iter()
        .any(|n| n.contains("transformer.h.") && n.contains("attn.c_attn"))
    {
        return ModelArchitecture::Qwen;
    }

    // Check for Phi-specific patterns (final_layernorm, parallel residual markers)
    if names_set.iter().any(|n| n.contains("final_layernorm")) {
        return ModelArchitecture::Phi;
    }

    // Check for Gemma-specific patterns (post_feedforward_layernorm)
    if names_set
        .iter()
        .any(|n| n.contains("post_feedforward_layernorm"))
    {
        return ModelArchitecture::Gemma;
    }

    // Check for standard Mistral/LLaMA patterns
    // These are very similar, so we default to Mistral unless we have explicit LLaMA markers
    if names_set
        .iter()
        .any(|n| n.contains("model.layers.") && n.contains("self_attn"))
    {
        // Could be either Mistral or LLaMA - need additional heuristics
        // Check for sliding_window in config or other Mistral-specific features
        return ModelArchitecture::Mistral; // Default to Mistral
    }

    ModelArchitecture::Unknown
}

/// Detect model architecture from UnifiedConfig
pub fn detect_architecture_from_config(config: &UnifiedConfig) -> ModelArchitecture {
    // First check if architecture is explicitly specified
    if let Some(ref arch) = config.architecture {
        let detected = ModelArchitecture::from_str(arch);
        if detected != ModelArchitecture::Unknown {
            return detected;
        }
        // Check for HuggingFace-style architecture class names
        let lower = arch.to_lowercase();
        if lower.contains("gptoss") || lower.contains("gpt_oss") {
            return ModelArchitecture::GptOss;
        }
    }

    // Check metadata for architecture hints
    if let Some(arch_name) = config.metadata.get("general.architecture") {
        return ModelArchitecture::from_str(arch_name);
    }

    // Fall back to Unknown
    ModelArchitecture::Unknown
}

/// Detect architecture from both config and tensor names
pub fn detect_architecture(config: &UnifiedConfig, tensor_names: &[String]) -> ModelArchitecture {
    // First try config-based detection (more reliable if available)
    let from_config = detect_architecture_from_config(config);
    if from_config != ModelArchitecture::Unknown {
        return from_config;
    }

    // Fall back to tensor-based detection
    detect_architecture_from_tensors(tensor_names)
}

/// Unified Model trait for polymorphic model handling
pub trait Model: Send + Sync {
    /// Get the model's architecture
    fn architecture(&self) -> ModelArchitecture;

    /// Get the model's configuration
    fn config(&self) -> &Config;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    /// Get hidden size
    fn hidden_size(&self) -> usize {
        self.config().hidden_size
    }

    /// Get number of layers
    fn n_layers(&self) -> usize {
        self.config().n_layers
    }

    /// Run forward pass for a single token
    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool);

    /// Run optimized forward pass (uses SIMD/parallel when available)
    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool);

    /// Get logits after forward pass
    fn get_logits<'a>(&self, state: &'a InferenceState) -> &'a [f32] {
        state.logits.as_slice().unwrap()
    }

    /// Encode text to tokens
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Decode tokens to text
    fn decode(&self, tokens: &[u32]) -> String;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_from_str() {
        assert_eq!(
            ModelArchitecture::from_str("mistral"),
            ModelArchitecture::Mistral
        );
        assert_eq!(
            ModelArchitecture::from_str("LLAMA"),
            ModelArchitecture::LLaMA
        );
        assert_eq!(ModelArchitecture::from_str("Phi-3"), ModelArchitecture::Phi);
        assert_eq!(
            ModelArchitecture::from_str("gemma"),
            ModelArchitecture::Gemma
        );
        assert_eq!(
            ModelArchitecture::from_str("qwen2"),
            ModelArchitecture::Qwen
        );
        assert_eq!(
            ModelArchitecture::from_str("unknown"),
            ModelArchitecture::Unknown
        );
    }

    #[test]
    fn test_tensor_name_pattern() {
        let pattern = TensorNamePattern::mistral();
        assert_eq!(
            pattern.q_proj_name(5),
            "model.layers.5.self_attn.q_proj.weight"
        );
        assert_eq!(
            pattern.gate_proj_name(0),
            "model.layers.0.mlp.gate_proj.weight"
        );
    }

    #[test]
    fn test_architecture_detection() {
        // Gemma-specific tensor names
        let gemma_tensors = vec![
            "model.layers.0.post_feedforward_layernorm.weight".to_string(),
            "model.embed_tokens.weight".to_string(),
        ];
        assert_eq!(
            detect_architecture_from_tensors(&gemma_tensors),
            ModelArchitecture::Gemma
        );

        // Phi-specific tensor names
        let phi_tensors = vec![
            "model.final_layernorm.weight".to_string(),
            "model.layers.0.input_layernorm.weight".to_string(),
        ];
        assert_eq!(
            detect_architecture_from_tensors(&phi_tensors),
            ModelArchitecture::Phi
        );

        // Qwen-specific tensor names
        let qwen_tensors = vec![
            "transformer.h.0.attn.c_attn.weight".to_string(),
            "transformer.wte.weight".to_string(),
        ];
        assert_eq!(
            detect_architecture_from_tensors(&qwen_tensors),
            ModelArchitecture::Qwen
        );
    }

    #[test]
    fn test_architecture_config() {
        let gemma_config = ArchitectureConfig::gemma();
        assert!(gemma_config.tie_embeddings);
        assert_eq!(gemma_config.activation, ActivationType::GeGLU);

        let phi_config = ArchitectureConfig::phi();
        assert!(phi_config.parallel_residual);
        assert!(phi_config.fused_gate_up);
    }

    #[test]
    fn test_deepseek_architecture_from_str() {
        assert_eq!(
            ModelArchitecture::from_str("deepseek"),
            ModelArchitecture::DeepSeek
        );
        assert_eq!(
            ModelArchitecture::from_str("deepseek_v3"),
            ModelArchitecture::DeepSeek
        );
        assert_eq!(
            ModelArchitecture::from_str("deepseek_v2"),
            ModelArchitecture::DeepSeek
        );
    }

    #[test]
    fn test_deepseek_detection_from_tensors() {
        let moe_tensors = vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.2.mlp.gate.weight".to_string(),
            "model.layers.2.mlp.experts.0.gate_proj.weight".to_string(),
            "model.layers.2.mlp.experts.0.up_proj.weight".to_string(),
            "model.layers.2.mlp.experts.0.down_proj.weight".to_string(),
        ];
        assert_eq!(
            detect_architecture_from_tensors(&moe_tensors),
            ModelArchitecture::DeepSeek
        );
    }

    #[test]
    fn test_deepseek_arch_config() {
        let config = ArchitectureConfig::deepseek();
        assert_eq!(config.architecture, ModelArchitecture::DeepSeek);
        assert!(config.is_moe);
        assert!(config.moe_tensor_names.is_some());
        assert!(!config.tie_embeddings);
        assert!(!config.fused_qkv);
    }

    #[test]
    fn test_gpt_oss_architecture_from_str() {
        assert_eq!(
            ModelArchitecture::from_str("gpt_oss"),
            ModelArchitecture::GptOss
        );
        assert_eq!(
            ModelArchitecture::from_str("gptoss"),
            ModelArchitecture::GptOss
        );
        assert_eq!(
            ModelArchitecture::from_str("gpt-oss"),
            ModelArchitecture::GptOss
        );
        assert_eq!(
            ModelArchitecture::from_str("GptOssForCausalLM"),
            ModelArchitecture::GptOss
        );
    }

    #[test]
    fn test_gpt_oss_detection_from_tensors() {
        let gpt_oss_tensors = vec![
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.self_attn.sinks".to_string(),
            "model.layers.0.mlp.router.weight".to_string(),
            "model.layers.0.mlp.experts.gate_up_proj_blocks".to_string(),
            "model.layers.0.mlp.experts.gate_up_proj_scales".to_string(),
            "model.layers.0.mlp.experts.down_proj_blocks".to_string(),
        ];
        assert_eq!(
            detect_architecture_from_tensors(&gpt_oss_tensors),
            ModelArchitecture::GptOss
        );
    }

    #[test]
    fn test_gpt_oss_arch_config() {
        let config = ArchitectureConfig::gpt_oss();
        assert_eq!(config.architecture, ModelArchitecture::GptOss);
        assert!(config.is_moe);
        assert!(config.attention_bias);
        assert!(config.moe_tensor_names.is_none()); // batched format, not per-expert
        assert!(!config.tie_embeddings);
        assert!(!config.fused_qkv);
    }

    #[test]
    fn test_moe_tensor_name_pattern() {
        let moe_names = MoeTensorNamePattern::deepseek();
        assert_eq!(
            moe_names.router_gate_name(5),
            "model.layers.5.mlp.gate.weight"
        );
        assert_eq!(
            moe_names.expert_gate_proj_name(3, 42),
            "model.layers.3.mlp.experts.42.gate_proj.weight"
        );
        assert_eq!(
            moe_names.shared_expert_gate_proj_name(7),
            "model.layers.7.mlp.shared_experts.gate_proj.weight"
        );
    }
}
