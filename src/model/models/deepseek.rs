//! DeepSeek MoE Model Implementation
//!
//! DeepSeek's Mixture-of-Experts architecture (DeepSeek-V2, V3, R1).
//!
//! # Key Features
//! - Mixture-of-Experts (MoE) FFN with top-k routing
//! - Shared experts that always activate (in addition to routed experts)
//! - Mixed dense + MoE layers (first N layers are dense, rest are MoE)
//! - Standard Grouped Query Attention (GQA)
//! - SwiGLU activation
//! - RMSNorm
//! - RoPE
//!
//! # Architecture
//! - Layers 0..first_moe_layer: standard dense transformer layers
//! - Layers first_moe_layer..n_layers: MoE transformer layers
//! - Each MoE layer has a router that selects top-k of N experts per token
//! - Optionally, shared experts always contribute to the output
//!
//! # Models
//! - DeepSeek-V3 (671B): 256 routed experts, 1 shared, top-8
//! - DeepSeek-R1 (671B): same architecture with reasoning/thinking
//!
//! # Usage
//! ```ignore
//! let params = Parameters::load("deepseek-v3/")?;
//! let model = DeepSeek::load(params)?;
//! let mut state = InferenceState::new(model.config.clone());
//! model.forward(&mut state, token, false);
//! ```

use crate::kernels;
use crate::loader::{Config, Parameters};
use crate::model::architecture::{
    ArchitectureConfig, Model, ModelArchitecture, MoeTensorNamePattern, TensorNamePattern,
};
use crate::model::modules::lazy_moe::{LazyExpert, LazyMoE, LazyMoERouter, LazySharedExpert};
use crate::model::modules::moe::{MoE, MoERouter};
use crate::model::LazyMoELayer;
use crate::model::{
    Attention, Embedding, InferenceState, Layer, LazyAttention, LazyEmbedding, LazyLayer, LazyMLP,
    MoELayer, RMSNorm, MLP,
};
use anyhow::Result;
use ndarray::{Array1, Array2};

/// Enum for layers that can be either dense or MoE
pub enum DeepSeekLayer {
    Dense(Layer),
    MoE(MoELayer),
}

/// Enum for lazy layers that can be either dense or MoE
pub enum LazyDeepSeekLayer {
    Dense(LazyLayer),
    MoE(LazyMoELayer),
}

/// DeepSeek MoE model with eager weight loading
pub struct DeepSeek {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub embedding: Embedding,
    pub layers: Vec<DeepSeekLayer>,
    pub norm: RMSNorm,
    pub lm_head: Array2<f32>,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl DeepSeek {
    /// Load a DeepSeek model from Parameters
    pub fn load(params: Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let tensor_names = TensorNamePattern::deepseek();
        let moe_names = MoeTensorNamePattern::deepseek();
        let arch_config = ArchitectureConfig::deepseek();

        // Load embedding table
        eprintln!("Loading DeepSeek embedding table...");
        let embed_data = params.get_tensor(tensor_names.embed_tokens)?;
        let embed_shape = params.get_tensor_shape(tensor_names.embed_tokens).unwrap();
        let embedding = Embedding::new(Array2::from_shape_vec(
            (embed_shape[0], embed_shape[1]),
            embed_data,
        )?);

        // Load final norm
        eprintln!("Loading DeepSeek final norm...");
        let norm_data = params.get_tensor(tensor_names.final_norm)?;
        let norm = RMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Load LM head
        eprintln!("Loading DeepSeek LM head...");
        let lm_head_data = params.get_tensor(tensor_names.lm_head)?;
        let lm_head_shape = params.get_tensor_shape(tensor_names.lm_head).unwrap();
        let lm_head = Array2::from_shape_vec((lm_head_shape[0], lm_head_shape[1]), lm_head_data)?;

        // Load layers (mixed dense + MoE)
        eprintln!(
            "Loading {} DeepSeek layers ({} dense, {} MoE)...",
            config.n_layers,
            config.first_moe_layer,
            config.n_layers - config.first_moe_layer
        );
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            if i.is_multiple_of(4) {
                eprintln!("  Loading layer {}/{}...", i, config.n_layers);
            }
            if i < config.first_moe_layer {
                // Dense layer
                layers.push(DeepSeekLayer::Dense(Self::load_dense_layer(
                    &params,
                    i,
                    &config,
                    &tensor_names,
                )?));
            } else {
                // MoE layer
                layers.push(DeepSeekLayer::MoE(Self::load_moe_layer(
                    &params,
                    i,
                    &config,
                    &tensor_names,
                    &moe_names,
                )?));
            }
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

    fn load_dense_layer(
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

        // Load dense MLP projections
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

    fn load_moe_layer(
        params: &Parameters,
        layer_idx: usize,
        config: &Config,
        tensor_names: &TensorNamePattern,
        moe_names: &MoeTensorNamePattern,
    ) -> Result<MoELayer> {
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

        // Load router gate
        let router_gate = Self::load_weight(params, &moe_names.router_gate_name(layer_idx))?;
        let router = MoERouter::new(
            router_gate,
            config.n_routed_experts,
            config.n_experts_per_token,
        );

        // Load all experts
        eprintln!(
            "    Loading {} experts for layer {}...",
            config.n_routed_experts, layer_idx
        );
        let mut experts = Vec::with_capacity(config.n_routed_experts);
        for e in 0..config.n_routed_experts {
            let gate_proj =
                Self::load_weight(params, &moe_names.expert_gate_proj_name(layer_idx, e))?;
            let up_proj = Self::load_weight(params, &moe_names.expert_up_proj_name(layer_idx, e))?;
            let down_proj =
                Self::load_weight(params, &moe_names.expert_down_proj_name(layer_idx, e))?;
            experts.push(MLP::new(gate_proj, up_proj, down_proj));
        }

        // Load shared expert(s) if present
        let shared_expert = if config.n_shared_experts > 0 {
            let gate_proj =
                Self::load_weight(params, &moe_names.shared_expert_gate_proj_name(layer_idx))?;
            let up_proj =
                Self::load_weight(params, &moe_names.shared_expert_up_proj_name(layer_idx))?;
            let down_proj =
                Self::load_weight(params, &moe_names.shared_expert_down_proj_name(layer_idx))?;
            Some(MLP::new(gate_proj, up_proj, down_proj))
        } else {
            None
        };

        let moe = MoE::new(router, experts, shared_expert);

        Ok(MoELayer::new(
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            moe,
        ))
    }

    fn load_weight(params: &Parameters, name: &str) -> Result<Array2<f32>> {
        let data = params.get_tensor(name)?;
        let shape = params.get_tensor_shape(name).unwrap();
        Ok(Array2::from_shape_vec((shape[0], shape[1]), data)?)
    }

    /// Forward pass
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Embedding lookup
        self.embedding.forward(state, token);

        // Pass through all layers (mixed dense + MoE)
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                DeepSeekLayer::Dense(l) => l.forward(state, i, debug),
                DeepSeekLayer::MoE(l) => l.forward(state, i, debug),
            }
        }

        // Final norm
        self.norm.forward(state);

        // LM head projection
        state
            .logits
            .assign(&kernels::matmul_vec(&self.lm_head, &state.hidden_state));
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        self.embedding.forward(state, token);

        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                DeepSeekLayer::Dense(l) => l.fast_forward(state, i, debug),
                DeepSeekLayer::MoE(l) => l.fast_forward(state, i, debug),
            }
        }

        self.norm.fast_forward(state);

        state.logits.assign(&kernels::fast_matmul_vec(
            &self.lm_head,
            &state.hidden_state,
        ));
    }
}

impl Model for DeepSeek {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::DeepSeek
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        DeepSeek::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        DeepSeek::fast_forward(self, state, token, debug)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }
}

// =============================================================================
// Lazy DeepSeek - Memory-efficient model with lazy tensor loading
// =============================================================================

/// Lazy DeepSeek model that keeps tensors memory-mapped and accesses them on-demand.
/// This is CRITICAL for MoE models -- a 671B model with 256 experts per layer
/// would require >1TB RAM if loaded eagerly. With lazy loading, only the router
/// gate weights and the selected top-k expert weights are accessed per token.
pub struct LazyDeepSeek<'a> {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub params: &'a Parameters,
    pub layers: Vec<LazyDeepSeekLayer>,
    pub norm: RMSNorm,
    pub embedding: LazyEmbedding,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl<'a> LazyDeepSeek<'a> {
    /// Load a lazy DeepSeek model from Parameters.
    /// Only small tensors (norms) are loaded eagerly; all projection matrices
    /// and expert weights remain memory-mapped.
    pub fn load(params: &'a Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let tensor_names = TensorNamePattern::deepseek();
        let moe_names = MoeTensorNamePattern::deepseek();
        let arch_config = ArchitectureConfig::deepseek();

        // Load final norm eagerly (small tensor)
        eprintln!("Loading DeepSeek norms (lazy model)...");
        let norm_data = params.get_tensor(tensor_names.final_norm)?;
        let norm = RMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Create lazy embedding
        let embedding = LazyEmbedding::new(tensor_names.embed_tokens.to_string());

        // Create lazy layers (mixed dense + MoE)
        eprintln!(
            "Initializing {} lazy DeepSeek layers ({} dense, {} MoE)...",
            config.n_layers,
            config.first_moe_layer,
            config.n_layers - config.first_moe_layer
        );
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            if i < config.first_moe_layer {
                layers.push(LazyDeepSeekLayer::Dense(Self::create_lazy_dense_layer(
                    params,
                    i,
                    &config,
                    &tensor_names,
                )?));
            } else {
                layers.push(LazyDeepSeekLayer::MoE(Self::create_lazy_moe_layer(
                    params,
                    i,
                    &config,
                    &tensor_names,
                    &moe_names,
                )?));
            }
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

    fn create_lazy_dense_layer(
        params: &Parameters,
        layer_idx: usize,
        config: &Config,
        tensor_names: &TensorNamePattern,
    ) -> Result<LazyLayer> {
        // Load norms eagerly (small tensors)
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

        // Create lazy dense MLP
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

    fn create_lazy_moe_layer(
        params: &Parameters,
        layer_idx: usize,
        config: &Config,
        tensor_names: &TensorNamePattern,
        moe_names: &MoeTensorNamePattern,
    ) -> Result<LazyMoELayer> {
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

        // Create lazy router
        let router = LazyMoERouter::new(
            moe_names.router_gate_name(layer_idx),
            config.n_routed_experts,
            config.n_experts_per_token,
        );

        // Create lazy experts (just tensor names, no data loaded)
        let mut experts = Vec::with_capacity(config.n_routed_experts);
        for e in 0..config.n_routed_experts {
            experts.push(LazyExpert::new(
                moe_names.expert_gate_proj_name(layer_idx, e),
                moe_names.expert_up_proj_name(layer_idx, e),
                moe_names.expert_down_proj_name(layer_idx, e),
            ));
        }

        // Create lazy shared expert if present
        let shared_expert = if config.n_shared_experts > 0 {
            Some(LazySharedExpert::new(
                moe_names.shared_expert_gate_proj_name(layer_idx),
                moe_names.shared_expert_up_proj_name(layer_idx),
                moe_names.shared_expert_down_proj_name(layer_idx),
            ))
        } else {
            None
        };

        let moe = LazyMoE::new(router, experts, shared_expert);

        Ok(LazyMoELayer::new(
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            moe,
        ))
    }

    /// Forward pass with lazy tensor loading
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Lazy embedding lookup
        self.embedding.forward(state, token, self.params);

        // Pass through all layers (mixed dense + MoE)
        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                LazyDeepSeekLayer::Dense(l) => l.forward(state, i, debug, self.params),
                LazyDeepSeekLayer::MoE(l) => l.forward(state, i, debug, self.params),
            }
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
        self.embedding.forward(state, token, self.params);

        for (i, layer) in self.layers.iter().enumerate() {
            match layer {
                LazyDeepSeekLayer::Dense(l) => l.fast_forward(state, i, debug, self.params),
                LazyDeepSeekLayer::MoE(l) => l.fast_forward(state, i, debug, self.params),
            }
        }

        self.norm.fast_forward(state);

        let lm_head_view = self.params.get_tensor_view("lm_head.weight").unwrap();
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let logits_slice = state.logits.as_slice_mut().unwrap();
        lm_head_view.matmul_vec_into(hidden_slice, logits_slice);
    }
}

impl Model for LazyDeepSeek<'_> {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::DeepSeek
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyDeepSeek::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyDeepSeek::fast_forward(self, state, token, debug)
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
    fn test_deepseek_arch_config() {
        let config = ArchitectureConfig::deepseek();
        assert_eq!(config.architecture, ModelArchitecture::DeepSeek);
        assert!(config.is_moe);
        assert!(config.moe_tensor_names.is_some());
    }
}
