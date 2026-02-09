//! Gemma Model Implementation (Phase 8)
//!
//! Google's Gemma family of open language models.
//!
//! # Key Features
//! - Tied embeddings (LM head shares weights with input embeddings)
//! - GeGLU activation (GELU-gated GLU)
//! - RMSNorm with special offset (+1 to weights)
//! - Grouped Query Attention (GQA)
//! - Rotary Position Embeddings (RoPE)
//!
//! # Architecture Differences from Mistral/LLaMA
//! - Tied embeddings: lm_head.weight = embed_tokens.weight
//! - Different normalization naming (post_feedforward_layernorm)
//! - GeGLU instead of SwiGLU
//! - RMSNorm weights are applied as (1 + weight) * x
//!
//! # Usage
//! ```ignore
//! let params = Parameters::load("gemma-2b.bin")?;
//! let model = Gemma::load(params)?;
//! let mut state = InferenceState::new(model.config.clone());
//! model.forward(&mut state, token, false);
//! ```

use crate::loader::{Config, Parameters, WeightMatrix};
use crate::model::architecture::{ArchitectureConfig, Model, ModelArchitecture, TensorNamePattern};
use crate::model::{Attention, Embedding, InferenceState};
use crate::model::{LazyAttention, LazyEmbedding, LazyMLP};
use anyhow::Result;
use ndarray::Array1;

/// Gemma-style RMSNorm with offset (applies as (1 + weight) * x)
pub struct GemmaRMSNorm {
    pub weight: Array1<f32>,
    pub eps: f32,
}

impl GemmaRMSNorm {
    pub fn new(weight: Array1<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass: Gemma-style RMSNorm with offset
    pub fn forward(&self, state: &mut InferenceState) {
        let x = &state.hidden_state;

        // Compute RMS (Root Mean Square)
        let sum_sq: f32 = x.iter().map(|&v| v * v).sum();
        let rms = (sum_sq / x.len() as f32 + self.eps).sqrt();
        let rms_inv = 1.0 / rms;

        // Normalize with offset: x = x * rms_inv * (1 + weight)
        for i in 0..x.len() {
            state.hidden_state[i] = state.hidden_state[i] * rms_inv * (1.0 + self.weight[i]);
        }
    }

    /// Fast forward (SIMD optimization could be added)
    pub fn fast_forward(&self, state: &mut InferenceState) {
        self.forward(state)
    }
}

/// Gemma-style MLP with GeGLU activation
pub struct GemmaMLP {
    pub gate_proj: WeightMatrix, // [intermediate_size, hidden_size]
    pub up_proj: WeightMatrix,   // [intermediate_size, hidden_size]
    pub down_proj: WeightMatrix, // [hidden_size, intermediate_size]
}

impl GemmaMLP {
    pub fn new(gate_proj: WeightMatrix, up_proj: WeightMatrix, down_proj: WeightMatrix) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// GELU activation (tanh approximation)
    /// Uses the formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn gelu(x: f32) -> f32 {
        let sqrt_2_over_pi = 0.797_884_6;
        let coeff = 0.044715;
        0.5 * x * (1.0 + (sqrt_2_over_pi * (x + coeff * x.powi(3))).tanh())
    }

    /// Forward pass: GeGLU (GELU-gated GLU)
    /// out = down_proj @ (gelu(gate_proj @ x) * (up_proj @ x))
    pub fn forward(&self, state: &mut InferenceState) {
        let hidden_slice = state.hidden_state.as_slice().unwrap();

        // gate = gate_proj @ hidden_state
        let gate = self.gate_proj.matmul_vec(hidden_slice);

        // up = up_proj @ hidden_state
        let up = self.up_proj.matmul_vec(hidden_slice);

        // Apply GELU to gate and multiply with up
        let mut intermediate = vec![0.0f32; gate.len()];
        for i in 0..gate.len() {
            intermediate[i] = Self::gelu(gate[i]) * up[i];
        }

        // down_proj @ intermediate
        self.down_proj
            .matmul_vec_into(&intermediate, state.hidden_state.as_slice_mut().unwrap());
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState) {
        self.forward(state)
    }
}

/// Gemma transformer layer
pub struct GemmaLayer {
    pub input_layernorm: GemmaRMSNorm,
    pub self_attn: Attention,
    pub post_feedforward_layernorm: GemmaRMSNorm,
    pub mlp: GemmaMLP,
}

impl GemmaLayer {
    pub fn new(
        input_layernorm: GemmaRMSNorm,
        self_attn: Attention,
        post_feedforward_layernorm: GemmaRMSNorm,
        mlp: GemmaMLP,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_feedforward_layernorm,
            mlp,
        }
    }

    /// Forward pass: norm -> attention -> residual -> norm -> mlp -> residual
    pub fn forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  Gemma Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm
        self.input_layernorm.forward(state);

        // Self-attention
        self.self_attn.forward(state);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual again
        state.residual.assign(&state.hidden_state);

        // Pre-MLP norm (post_feedforward_layernorm in Gemma naming)
        self.post_feedforward_layernorm.forward(state);

        // MLP with GeGLU
        self.mlp.forward(state);

        // Residual connection
        state.hidden_state += &state.residual;
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  Gemma Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm
        self.input_layernorm.fast_forward(state);

        // Self-attention (optimized)
        self.self_attn.fast_forward(state);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual again
        state.residual.assign(&state.hidden_state);

        // Pre-MLP norm
        self.post_feedforward_layernorm.fast_forward(state);

        // MLP with GeGLU (optimized)
        self.mlp.fast_forward(state);

        // Residual connection
        state.hidden_state += &state.residual;
    }
}

/// Gemma model with eager weight loading
pub struct Gemma {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub embedding: Embedding,
    pub layers: Vec<GemmaLayer>,
    pub norm: GemmaRMSNorm,
    /// LM head - for Gemma this is the same as embedding (tied weights)
    pub lm_head: WeightMatrix,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl Gemma {
    /// Load a Gemma model from Parameters
    pub fn load(params: Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let arch_config = ArchitectureConfig::gemma();
        let tensor_names = TensorNamePattern::gemma();

        // Load embedding table
        eprintln!("Loading Gemma embedding table...");
        let embedding = Embedding::new(params.get_weight_matrix(tensor_names.embed_tokens)?);

        // For Gemma, LM head uses tied embeddings - share with embedding if lm_head.weight doesn't exist
        eprintln!("Loading Gemma LM head...");
        let lm_head = if params.get_tensor_shape("lm_head.weight").is_some() {
            params.get_weight_matrix("lm_head.weight")?
        } else {
            // Use embedding weights (tied)
            embedding.table.clone()
        };

        // Load final norm
        eprintln!("Loading Gemma final norm...");
        let norm_data = params.get_tensor(tensor_names.final_norm)?;
        let norm = GemmaRMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Load layers
        eprintln!("Loading {} Gemma layers...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            if i.is_multiple_of(4) {
                eprintln!("  Loading layer {}/{}...", i, config.n_layers);
            }
            layers.push(Self::load_layer(&params, i, &config)?);
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

    fn load_layer(params: &Parameters, layer_idx: usize, config: &Config) -> Result<GemmaLayer> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Load norms (Gemma-style RMSNorm)
        let input_norm_data = params.get_tensor(&format!("{}.input_layernorm.weight", prefix))?;
        let input_layernorm = GemmaRMSNorm::new(Array1::from_vec(input_norm_data), config.norm_eps);

        // Gemma uses "post_feedforward_layernorm" instead of "post_attention_layernorm"
        let post_ff_norm_name = format!("{}.post_feedforward_layernorm.weight", prefix);
        let post_ff_norm_data = if params.get_tensor_shape(&post_ff_norm_name).is_some() {
            params.get_tensor(&post_ff_norm_name)?
        } else {
            // Fall back to standard naming
            params.get_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?
        };
        let post_feedforward_layernorm =
            GemmaRMSNorm::new(Array1::from_vec(post_ff_norm_data), config.norm_eps);

        // Load attention projections
        let q_proj = Self::load_weight(params, &format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_proj = Self::load_weight(params, &format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_proj = Self::load_weight(params, &format!("{}.self_attn.v_proj.weight", prefix))?;
        let o_proj = Self::load_weight(params, &format!("{}.self_attn.o_proj.weight", prefix))?;

        let self_attn = Attention::new(layer_idx, q_proj, k_proj, v_proj, o_proj);

        // Load MLP projections (GeGLU)
        let gate_proj = Self::load_weight(params, &format!("{}.mlp.gate_proj.weight", prefix))?;
        let up_proj = Self::load_weight(params, &format!("{}.mlp.up_proj.weight", prefix))?;
        let down_proj = Self::load_weight(params, &format!("{}.mlp.down_proj.weight", prefix))?;

        let mlp = GemmaMLP::new(gate_proj, up_proj, down_proj);

        Ok(GemmaLayer::new(
            input_layernorm,
            self_attn,
            post_feedforward_layernorm,
            mlp,
        ))
    }

    fn load_weight(params: &Parameters, name: &str) -> Result<WeightMatrix> {
        params.get_weight_matrix(name)
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

        // Pass through all layers with optimizations
        for (i, layer) in self.layers.iter().enumerate() {
            layer.fast_forward(state, i, debug);
        }

        // Final norm
        self.norm.fast_forward(state);

        // LM head projection
        self.lm_head.matmul_vec_into(
            state.hidden_state.as_slice().unwrap(),
            state.logits.as_slice_mut().unwrap(),
        );
    }
}

impl Model for Gemma {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Gemma
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        Gemma::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        Gemma::fast_forward(self, state, token, debug)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }
}

/// Lazy Gemma model with memory-mapped weights
pub struct LazyGemma<'a> {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub params: &'a Parameters,
    pub layers: Vec<LazyGemmaLayer>,
    pub norm: GemmaRMSNorm,
    pub embedding: LazyEmbedding,
    /// Whether to use tied embeddings for LM head
    pub tie_embeddings: bool,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

/// Lazy Gemma layer (stores tensor names, loads on demand)
pub struct LazyGemmaLayer {
    pub layer_idx: usize,
    pub input_norm: GemmaRMSNorm,
    pub post_ff_norm: GemmaRMSNorm,
    pub attn: LazyAttention,
    pub mlp: LazyMLP,
}

impl<'a> LazyGemma<'a> {
    /// Load a lazy Gemma model
    pub fn load(params: &'a Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let arch_config = ArchitectureConfig::gemma();

        // Check if we should use tied embeddings
        let tie_embeddings = params.get_tensor_shape("lm_head.weight").is_none();

        // Load norms eagerly (small tensors)
        eprintln!("Loading Gemma norms (lazy model)...");
        let norm_data = params.get_tensor("model.norm.weight")?;
        let norm = GemmaRMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Create lazy embedding
        let embedding = LazyEmbedding::new("model.embed_tokens.weight".to_string());

        // Create lazy layers
        eprintln!("Initializing {} lazy Gemma layers...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            layers.push(Self::create_lazy_layer(params, i, &config)?);
        }

        Ok(Self {
            config,
            arch_config,
            params,
            layers,
            norm,
            embedding,
            tie_embeddings,
            tokenizer,
        })
    }

    fn create_lazy_layer(
        params: &Parameters,
        layer_idx: usize,
        config: &Config,
    ) -> Result<LazyGemmaLayer> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Load norms eagerly
        let input_norm_data = params.get_tensor(&format!("{}.input_layernorm.weight", prefix))?;
        let input_norm = GemmaRMSNorm::new(Array1::from_vec(input_norm_data), config.norm_eps);

        let post_ff_norm_name = format!("{}.post_feedforward_layernorm.weight", prefix);
        let post_ff_norm_data = if params.get_tensor_shape(&post_ff_norm_name).is_some() {
            params.get_tensor(&post_ff_norm_name)?
        } else {
            params.get_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?
        };
        let post_ff_norm = GemmaRMSNorm::new(Array1::from_vec(post_ff_norm_data), config.norm_eps);

        // Create lazy attention
        let attn = LazyAttention::new(
            layer_idx,
            format!("{}.self_attn.q_proj.weight", prefix),
            format!("{}.self_attn.k_proj.weight", prefix),
            format!("{}.self_attn.v_proj.weight", prefix),
            format!("{}.self_attn.o_proj.weight", prefix),
        );

        // Create lazy MLP
        let mlp = LazyMLP::new(
            format!("{}.mlp.gate_proj.weight", prefix),
            format!("{}.mlp.up_proj.weight", prefix),
            format!("{}.mlp.down_proj.weight", prefix),
        );

        Ok(LazyGemmaLayer {
            layer_idx,
            input_norm,
            post_ff_norm,
            attn,
            mlp,
        })
    }

    /// Forward pass with lazy tensor loading
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Lazy embedding lookup
        self.embedding.forward(state, token, self.params);

        // Pass through all layers
        for layer in self.layers.iter() {
            self.forward_layer(state, layer, debug);
        }

        // Final norm
        self.norm.forward(state);

        // LM head projection (lazy, with tied embeddings support)
        let lm_head_name = if self.tie_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        let lm_head_view = self.params.get_tensor_view(lm_head_name).unwrap();
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let logits = lm_head_view.matmul_vec(hidden_slice);
        for (i, &v) in logits.iter().enumerate() {
            state.logits[i] = v;
        }
    }

    fn forward_layer(&self, state: &mut InferenceState, layer: &LazyGemmaLayer, debug: bool) {
        if debug && layer.layer_idx.is_multiple_of(8) {
            eprintln!(
                "  Lazy Gemma Layer {}/{}",
                layer.layer_idx, self.config.n_layers
            );
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm (Gemma-style)
        layer.input_norm.forward(state);

        // Lazy attention (uses layer_idx stored in LazyAttention)
        layer.attn.forward(state, self.params);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-MLP norm
        layer.post_ff_norm.forward(state);

        // Lazy MLP with GeGLU
        self.forward_geglu_mlp(state, &layer.mlp);

        // Residual connection
        state.hidden_state += &state.residual;
    }

    fn forward_geglu_mlp(&self, state: &mut InferenceState, mlp: &LazyMLP) {
        let gate_view = self.params.get_tensor_view(&mlp.gate_proj_name).unwrap();
        let up_view = self.params.get_tensor_view(&mlp.up_proj_name).unwrap();
        let down_view = self.params.get_tensor_view(&mlp.down_proj_name).unwrap();

        let hidden_slice = state.hidden_state.as_slice().unwrap();

        // Gate and up projections
        let gate = gate_view.matmul_vec(hidden_slice);
        let up = up_view.matmul_vec(hidden_slice);

        // Apply GELU to gate and multiply with up
        let mut intermediate = vec![0.0f32; gate.len()];
        for i in 0..gate.len() {
            intermediate[i] = GemmaMLP::gelu(gate[i]) * up[i];
        }

        // Down projection
        let output = down_view.matmul_vec(&intermediate);
        state.hidden_state.assign(&Array1::from_vec(output));
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Lazy embedding lookup
        self.embedding.forward(state, token, self.params);

        // Pass through all layers
        for layer in self.layers.iter() {
            self.fast_forward_layer(state, layer, debug);
        }

        // Final norm
        self.norm.fast_forward(state);

        // LM head projection (lazy)
        let lm_head_name = if self.tie_embeddings {
            "model.embed_tokens.weight"
        } else {
            "lm_head.weight"
        };
        let lm_head_view = self.params.get_tensor_view(lm_head_name).unwrap();
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let logits_slice = state.logits.as_slice_mut().unwrap();
        lm_head_view.matmul_vec_into(hidden_slice, logits_slice);
    }

    fn fast_forward_layer(&self, state: &mut InferenceState, layer: &LazyGemmaLayer, debug: bool) {
        // Same as forward for lazy mode (optimizations would need custom lazy kernels)
        self.forward_layer(state, layer, debug);
    }
}

impl Model for LazyGemma<'_> {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Gemma
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyGemma::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyGemma::fast_forward(self, state, token, debug)
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
    use crate::ActivationType;

    #[test]
    fn test_gelu() {
        // Test GELU activation
        assert!((GemmaMLP::gelu(0.0)).abs() < 0.01);
        assert!(GemmaMLP::gelu(1.0) > 0.8);
        assert!(GemmaMLP::gelu(-1.0) < -0.1);
    }

    #[test]
    fn test_gemma_arch_config() {
        let arch = ArchitectureConfig::gemma();
        assert_eq!(arch.architecture, ModelArchitecture::Gemma);
        assert!(arch.tie_embeddings);
        assert_eq!(arch.activation, ActivationType::GeGLU);
    }
}
