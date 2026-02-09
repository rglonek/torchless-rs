//! LLaMA Model Implementation (Phase 8)
//!
//! Meta's LLaMA (Large Language Model Meta AI) architecture.
//! Supports LLaMA 1, LLaMA 2, and LLaMA 3 models.
//!
//! # Key Features
//! - Grouped Query Attention (GQA)
//! - SwiGLU activation (SiLU-gated GLU)
//! - RMSNorm (Root Mean Square Layer Normalization)
//! - Rotary Position Embeddings (RoPE)
//! - Optional RoPE scaling for extended context (LLaMA 3)
//!
//! # Differences from Mistral
//! - No sliding window attention
//! - Different RoPE scaling options (linear, NTK, YaRN)
//! - Different vocabulary sizes
//!
//! # Usage
//! ```ignore
//! let params = Parameters::load("llama-7b.bin")?;
//! let model = LLaMA::load(params)?;
//! let mut state = InferenceState::new(model.config.clone());
//! model.forward(&mut state, token, false);
//! ```

use crate::kernels;
use crate::loader::{Config, Parameters, WeightMatrix};
use crate::model::architecture::{
    ArchitectureConfig, Model, ModelArchitecture, RopeScaling, TensorNamePattern,
};
use crate::model::{Attention, Embedding, InferenceState, Layer, RMSNorm, MLP};
use crate::model::{LazyAttention, LazyEmbedding, LazyLayer, LazyMLP};
use anyhow::Result;
use ndarray::Array1;

/// LLaMA model with eager weight loading
pub struct LLaMA {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub embedding: Embedding,
    pub layers: Vec<Layer>,
    pub norm: RMSNorm,
    pub lm_head: WeightMatrix,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl LLaMA {
    /// Load a LLaMA model from Parameters
    pub fn load(params: Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let tensor_names = TensorNamePattern::llama();

        // Determine LLaMA version for appropriate RoPE scaling
        let arch_config = Self::detect_llama_version(&config, &params);

        // Load embedding table
        eprintln!("Loading LLaMA embedding table...");
        let embedding = Embedding::new(params.get_weight_matrix(tensor_names.embed_tokens)?);

        // Load final norm
        eprintln!("Loading LLaMA final norm...");
        let norm_data = params.get_tensor(tensor_names.final_norm)?;
        let norm = RMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Load LM head
        eprintln!("Loading LLaMA LM head...");
        let lm_head = params.get_weight_matrix(tensor_names.lm_head)?;

        // Load layers
        eprintln!("Loading {} LLaMA layers...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            if i.is_multiple_of(4) {
                eprintln!("  Loading layer {}/{}...", i, config.n_layers);
            }
            layers.push(Self::load_layer(&params, i, &config, &tensor_names)?);
        }

        Ok(Self {
            config,
            arch_config,
            embedding,
            layers,
            norm,
            lm_head,
            tokenizer,
        })
    }

    /// Detect LLaMA version (1, 2, or 3) from config
    fn detect_llama_version(config: &Config, _params: &Parameters) -> ArchitectureConfig {
        // LLaMA 3 typically has:
        // - Vocabulary size around 128k
        // - RoPE theta of 500000 (vs 10000 for LLaMA 1/2)
        let is_llama3 = config.rope_theta > 100000.0 || config.vocab_size > 100000;

        if is_llama3 {
            eprintln!("Detected LLaMA 3 architecture");
            ArchitectureConfig::llama3(config.max_position_embeddings)
        } else {
            eprintln!("Detected LLaMA 1/2 architecture");
            ArchitectureConfig::llama()
        }
    }

    fn load_layer(
        params: &Parameters,
        layer_idx: usize,
        config: &Config,
        tensor_names: &TensorNamePattern,
    ) -> Result<Layer> {
        // Load norms
        let input_norm_data = params.get_tensor(&tensor_names.input_layernorm_name(layer_idx))?;
        let input_layernorm = RMSNorm::new(Array1::from_vec(input_norm_data), config.norm_eps);

        let post_attn_norm_data =
            params.get_tensor(&tensor_names.post_attention_layernorm_name(layer_idx))?;
        let post_attention_layernorm =
            RMSNorm::new(Array1::from_vec(post_attn_norm_data), config.norm_eps);

        // Load attention projections
        let q_proj = Self::load_weight(params, &tensor_names.q_proj_name(layer_idx))?;
        let k_proj = Self::load_weight(params, &tensor_names.k_proj_name(layer_idx))?;
        let v_proj = Self::load_weight(params, &tensor_names.v_proj_name(layer_idx))?;
        let o_proj = Self::load_weight(params, &tensor_names.o_proj_name(layer_idx))?;

        let self_attn = Attention::new(layer_idx, q_proj, k_proj, v_proj, o_proj);

        // Load MLP projections
        let gate_proj = Self::load_weight(params, &tensor_names.gate_proj_name(layer_idx))?;
        let up_proj = Self::load_weight(params, &tensor_names.up_proj_name(layer_idx))?;
        let down_proj = Self::load_weight(params, &tensor_names.down_proj_name(layer_idx))?;

        let mlp = MLP::new(gate_proj, up_proj, down_proj);

        Ok(Layer::new(
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        ))
    }

    fn load_weight(params: &Parameters, name: &str) -> Result<WeightMatrix> {
        params.get_weight_matrix(name)
    }

    /// Compute scaled RoPE frequencies for extended context
    #[allow(dead_code)]
    fn compute_scaled_rope_freqs(&self, head_dim: usize, pos: usize) -> (Array1<f32>, Array1<f32>) {
        let base_freq = kernels::init_rope_freqs(head_dim, self.config.rope_theta);

        match self.arch_config.rope_scaling {
            RopeScaling::None => kernels::rope_embeddings(&base_freq, pos),
            RopeScaling::Linear { factor } => {
                // Linear scaling: divide position by factor
                let scaled_pos = (pos as f32) / factor;
                let inv_freq = &base_freq;
                let half = head_dim / 2;
                let mut cos = Array1::zeros(half);
                let mut sin = Array1::zeros(half);
                for i in 0..half {
                    let angle = scaled_pos * inv_freq[i];
                    cos[i] = angle.cos();
                    sin[i] = angle.sin();
                }
                (cos, sin)
            }
            RopeScaling::DynamicNTK {
                factor,
                original_max_position,
            } => {
                // Dynamic NTK scaling (used in extended context models)
                let seq_len = pos + 1;
                if seq_len <= original_max_position {
                    kernels::rope_embeddings(&base_freq, pos)
                } else {
                    // Scale the base frequency
                    let scale = (factor * (seq_len as f32) / (original_max_position as f32))
                        .powf(head_dim as f32 / (head_dim as f32 - 2.0));
                    let scaled_theta = self.config.rope_theta * scale;
                    let scaled_freq = kernels::init_rope_freqs(head_dim, scaled_theta);
                    kernels::rope_embeddings(&scaled_freq, pos)
                }
            }
            RopeScaling::YaRN {
                factor,
                original_max_position,
            } => {
                // YaRN scaling (Yet another RoPE extensioN)
                // Simplified implementation - full YaRN has more complex interpolation
                let seq_len = pos + 1;
                if seq_len <= original_max_position {
                    kernels::rope_embeddings(&base_freq, pos)
                } else {
                    let scale = factor * (seq_len as f32) / (original_max_position as f32);
                    let scaled_pos = (pos as f32) / scale;
                    let inv_freq = &base_freq;
                    let half = head_dim / 2;
                    let mut cos = Array1::zeros(half);
                    let mut sin = Array1::zeros(half);
                    for i in 0..half {
                        let angle = scaled_pos * inv_freq[i];
                        cos[i] = angle.cos();
                        sin[i] = angle.sin();
                    }
                    (cos, sin)
                }
            }
        }
    }

    /// Forward pass
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Embedding lookup
        self.embedding.forward(state, token);

        // Pass through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(state, i, debug);
        }

        // Final norm
        self.norm.forward(state);

        // LM head projection
        self.lm_head.matmul_vec_into(
            state.hidden_state.as_slice().unwrap(),
            state.logits.as_slice_mut().unwrap(),
        );
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Embedding lookup
        self.embedding.forward(state, token);

        // Pass through all layers with optimized kernels
        for (i, layer) in self.layers.iter().enumerate() {
            layer.fast_forward(state, i, debug);
        }

        // Final norm (SIMD when available)
        self.norm.fast_forward(state);

        // LM head projection
        self.lm_head.matmul_vec_into(
            state.hidden_state.as_slice().unwrap(),
            state.logits.as_slice_mut().unwrap(),
        );
    }
}

impl Model for LLaMA {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::LLaMA
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LLaMA::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LLaMA::fast_forward(self, state, token, debug)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }
}

/// Lazy LLaMA model with memory-mapped weights
pub struct LazyLLaMA<'a> {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub params: &'a Parameters,
    pub layers: Vec<LazyLayer>,
    pub norm: RMSNorm,
    pub embedding: LazyEmbedding,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl<'a> LazyLLaMA<'a> {
    /// Load a lazy LLaMA model
    pub fn load(params: &'a Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let tensor_names = TensorNamePattern::llama();
        let arch_config = LLaMA::detect_llama_version(&config, params);

        // Load norms eagerly (small tensors)
        eprintln!("Loading LLaMA norms (lazy model)...");
        let norm_data = params.get_tensor(tensor_names.final_norm)?;
        let norm = RMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Create lazy embedding
        let embedding = LazyEmbedding::new(tensor_names.embed_tokens.to_string());

        // Create lazy layers
        eprintln!("Initializing {} lazy LLaMA layers...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            layers.push(Self::create_lazy_layer(params, i, &config, &tensor_names)?);
        }

        Ok(Self {
            config,
            arch_config,
            params,
            layers,
            norm,
            embedding,
            tokenizer,
        })
    }

    fn create_lazy_layer(
        params: &Parameters,
        layer_idx: usize,
        config: &Config,
        tensor_names: &TensorNamePattern,
    ) -> Result<LazyLayer> {
        // Load norms eagerly
        let input_norm_data = params.get_tensor(&tensor_names.input_layernorm_name(layer_idx))?;
        let input_layernorm = RMSNorm::new(Array1::from_vec(input_norm_data), config.norm_eps);

        let post_attn_norm_data =
            params.get_tensor(&tensor_names.post_attention_layernorm_name(layer_idx))?;
        let post_attention_layernorm =
            RMSNorm::new(Array1::from_vec(post_attn_norm_data), config.norm_eps);

        // Create lazy attention
        let self_attn = LazyAttention::new(
            layer_idx,
            tensor_names.q_proj_name(layer_idx),
            tensor_names.k_proj_name(layer_idx),
            tensor_names.v_proj_name(layer_idx),
            tensor_names.o_proj_name(layer_idx),
        );

        // Create lazy MLP
        let mlp = LazyMLP::new(
            tensor_names.gate_proj_name(layer_idx),
            tensor_names.up_proj_name(layer_idx),
            tensor_names.down_proj_name(layer_idx),
        );

        Ok(LazyLayer::new(
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        ))
    }

    /// Forward pass with lazy tensor loading
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Lazy embedding lookup
        self.embedding.forward(state, token, self.params);

        // Pass through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(state, i, debug, self.params);
        }

        // Final norm
        self.norm.forward(state);

        // LM head projection (lazy)
        let lm_head_view = self.params.get_tensor_view("lm_head.weight").unwrap();
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let logits = lm_head_view.matmul_vec(hidden_slice);
        for (i, &v) in logits.iter().enumerate() {
            state.logits[i] = v;
        }
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Lazy embedding lookup
        self.embedding.forward(state, token, self.params);

        // Pass through all layers with optimizations
        for (i, layer) in self.layers.iter().enumerate() {
            layer.fast_forward(state, i, debug, self.params);
        }

        // Final norm (SIMD when available)
        self.norm.fast_forward(state);

        // LM head projection (lazy)
        let lm_head_view = self.params.get_tensor_view("lm_head.weight").unwrap();
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let logits_slice = state.logits.as_slice_mut().unwrap();
        lm_head_view.matmul_vec_into(hidden_slice, logits_slice);
    }
}

impl Model for LazyLLaMA<'_> {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::LLaMA
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyLLaMA::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyLLaMA::fast_forward(self, state, token, debug)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_arch_config() {
        let arch = ArchitectureConfig::llama();
        assert_eq!(arch.architecture, ModelArchitecture::LLaMA);
        assert!(!arch.tie_embeddings);
        assert!(!arch.fused_qkv);
    }

    #[test]
    fn test_llama3_rope_scaling() {
        let arch = ArchitectureConfig::llama3(4096);
        match arch.rope_scaling {
            RopeScaling::DynamicNTK {
                factor,
                original_max_position,
            } => {
                assert_eq!(factor, 8.0);
                assert_eq!(original_max_position, 4096);
            }
            _ => panic!("Expected DynamicNTK scaling for LLaMA 3"),
        }
    }
}
