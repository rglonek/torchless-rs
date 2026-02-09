//! Phi Model Implementation (Phase 8)
//!
//! Microsoft's Phi family of efficient language models (Phi-2, Phi-3).
//!
//! # Key Features
//! - Parallel residual connections (attention and MLP run in parallel)
//! - Fused gate+up projection in MLP
//! - Layer Normalization (instead of RMSNorm)
//! - GELU (tanh approximation) activation
//! - Rotary Position Embeddings (RoPE)
//!
//! # Architecture Differences from Mistral/LLaMA
//! - Parallel blocks: attention and MLP outputs are summed, not sequential
//! - Different normalization (LayerNorm vs RMSNorm)
//! - Fused MLP projections
//! - Different tensor naming conventions
//!
//! # Usage
//! ```ignore
//! let params = Parameters::load("phi-3-mini.bin")?;
//! let model = Phi::load(params)?;
//! let mut state = InferenceState::new(model.config.clone());
//! model.forward(&mut state, token, false);
//! ```

use crate::kernels;
use crate::loader::{Config, Parameters};
use crate::model::architecture::{ArchitectureConfig, Model, ModelArchitecture};
use crate::model::LazyEmbedding;
use crate::model::{Attention, Embedding, InferenceState};
use anyhow::Result;
use ndarray::{Array1, Array2};

/// Layer Normalization (used by Phi instead of RMSNorm)
pub struct LayerNorm {
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// Forward pass: standard layer normalization
    pub fn forward(&self, state: &mut InferenceState) {
        let x = &state.hidden_state;
        let n = x.len() as f32;

        // Compute mean
        let mean: f32 = x.iter().sum::<f32>() / n;

        // Compute variance
        let variance: f32 = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;

        // Normalize
        let std_inv = 1.0 / (variance + self.eps).sqrt();
        for i in 0..x.len() {
            state.hidden_state[i] =
                (state.hidden_state[i] - mean) * std_inv * self.weight[i] + self.bias[i];
        }
    }

    /// Fast forward (same as forward for now, could add SIMD optimization)
    pub fn fast_forward(&self, state: &mut InferenceState) {
        self.forward(state)
    }
}

/// Phi-style MLP with fused gate+up projection and GELU activation
pub struct PhiMLP {
    /// Fused gate and up projection: [2 * intermediate_size, hidden_size]
    pub gate_up_proj: Array2<f32>,
    /// Down projection: [hidden_size, intermediate_size]
    pub down_proj: Array2<f32>,
    /// Intermediate size (half of gate_up_proj output)
    pub intermediate_size: usize,
}

impl PhiMLP {
    pub fn new(gate_up_proj: Array2<f32>, down_proj: Array2<f32>) -> Self {
        let intermediate_size = gate_up_proj.shape()[0] / 2;
        Self {
            gate_up_proj,
            down_proj,
            intermediate_size,
        }
    }

    /// GELU activation with tanh approximation
    fn gelu_tanh(x: f32) -> f32 {
        let sqrt_2_over_pi = 0.797_884_6;
        let coeff = 0.044715;
        0.5 * x * (1.0 + (sqrt_2_over_pi * (x + coeff * x.powi(3))).tanh())
    }

    /// Forward pass: fused gate+up with GELU activation
    pub fn forward(&self, state: &mut InferenceState) {
        // Project to fused gate+up
        let fused = kernels::matmul_vec(&self.gate_up_proj, &state.hidden_state);

        // Split and apply gated activation
        // First half is gate (apply GELU), second half is up
        let mut intermediate = Array1::zeros(self.intermediate_size);
        for i in 0..self.intermediate_size {
            let gate = Self::gelu_tanh(fused[i]);
            let up = fused[i + self.intermediate_size];
            intermediate[i] = gate * up;
        }

        // Down projection
        state
            .hidden_state
            .assign(&kernels::matmul_vec(&self.down_proj, &intermediate));
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState) {
        // Use parallel matmul when available
        let fused = kernels::fast_matmul_vec(&self.gate_up_proj, &state.hidden_state);

        // Split and apply gated activation
        let mut intermediate = Array1::zeros(self.intermediate_size);
        for i in 0..self.intermediate_size {
            let gate = Self::gelu_tanh(fused[i]);
            let up = fused[i + self.intermediate_size];
            intermediate[i] = gate * up;
        }

        // Down projection
        state
            .hidden_state
            .assign(&kernels::fast_matmul_vec(&self.down_proj, &intermediate));
    }
}

/// Phi-style transformer layer with parallel residuals
pub struct PhiLayer {
    pub input_layernorm: LayerNorm,
    pub self_attn: Attention,
    pub mlp: PhiMLP,
}

impl PhiLayer {
    pub fn new(input_layernorm: LayerNorm, self_attn: Attention, mlp: PhiMLP) -> Self {
        Self {
            input_layernorm,
            self_attn,
            mlp,
        }
    }

    /// Forward pass with parallel residuals
    /// Unlike Mistral/LLaMA which are sequential:
    ///   x = x + attention(norm(x))
    ///   x = x + mlp(norm(x))
    /// Phi uses parallel computation:
    ///   x = x + attention(norm(x)) + mlp(norm(x))
    pub fn forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx % 8 == 0 {
            eprintln!("  Phi Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save original residual
        state.residual.assign(&state.hidden_state);

        // Normalize
        self.input_layernorm.forward(state);

        // Save normalized state for MLP
        let normed = state.hidden_state.clone();

        // Attention branch
        self.self_attn.forward(state);
        let attn_output = state.hidden_state.clone();

        // MLP branch (uses normalized input)
        state.hidden_state.assign(&normed);
        self.mlp.forward(state);

        // Parallel residual: x = x + attn + mlp
        state.hidden_state += &state.residual;
        state.hidden_state += &attn_output;
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx % 8 == 0 {
            eprintln!("  Phi Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save original residual
        state.residual.assign(&state.hidden_state);

        // Normalize
        self.input_layernorm.fast_forward(state);

        // Save normalized state for MLP
        let normed = state.hidden_state.clone();

        // Attention branch (optimized)
        self.self_attn.fast_forward(state);
        let attn_output = state.hidden_state.clone();

        // MLP branch (uses normalized input)
        state.hidden_state.assign(&normed);
        self.mlp.fast_forward(state);

        // Parallel residual: x = x + attn + mlp
        state.hidden_state += &state.residual;
        state.hidden_state += &attn_output;
    }
}

/// Phi model with eager weight loading
pub struct Phi {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub embedding: Embedding,
    pub layers: Vec<PhiLayer>,
    pub final_norm: LayerNorm,
    pub lm_head: Array2<f32>,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl Phi {
    /// Load a Phi model from Parameters
    pub fn load(params: Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let arch_config = ArchitectureConfig::phi();

        // Phi uses different tensor names
        let embed_name = "model.embed_tokens.weight";
        let final_norm_weight = "model.final_layernorm.weight";
        let final_norm_bias = "model.final_layernorm.bias";
        let lm_head_name = "lm_head.weight";

        // Load embedding table
        eprintln!("Loading Phi embedding table...");
        let embed_data = params.get_tensor(embed_name)?;
        let embed_shape = params.get_tensor_shape(embed_name).unwrap();
        let embedding = Embedding::new(Array2::from_shape_vec(
            (embed_shape[0], embed_shape[1]),
            embed_data,
        )?);

        // Load final norm (LayerNorm with bias)
        eprintln!("Loading Phi final norm...");
        let norm_weight = params.get_tensor(final_norm_weight)?;
        let norm_bias = params
            .get_tensor(final_norm_bias)
            .unwrap_or_else(|_| vec![0.0; norm_weight.len()]); // Default to zeros if no bias
        let final_norm = LayerNorm::new(
            Array1::from_vec(norm_weight),
            Array1::from_vec(norm_bias),
            config.norm_eps,
        );

        // Load LM head
        eprintln!("Loading Phi LM head...");
        let lm_head_data = params.get_tensor(lm_head_name)?;
        let lm_head_shape = params.get_tensor_shape(lm_head_name).unwrap();
        let lm_head = Array2::from_shape_vec((lm_head_shape[0], lm_head_shape[1]), lm_head_data)?;

        // Load layers
        eprintln!("Loading {} Phi layers...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            if i % 4 == 0 {
                eprintln!("  Loading layer {}/{}...", i, config.n_layers);
            }
            layers.push(Self::load_layer(&params, i, &config)?);
        }

        Ok(Self {
            config,
            arch_config,
            embedding,
            layers,
            final_norm,
            lm_head,
            tokenizer,
        })
    }

    fn load_layer(params: &Parameters, layer_idx: usize, config: &Config) -> Result<PhiLayer> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Load input layernorm (with bias)
        let input_norm_weight = params.get_tensor(&format!("{}.input_layernorm.weight", prefix))?;
        let input_norm_bias = params
            .get_tensor(&format!("{}.input_layernorm.bias", prefix))
            .unwrap_or_else(|_| vec![0.0; input_norm_weight.len()]);
        let input_layernorm = LayerNorm::new(
            Array1::from_vec(input_norm_weight),
            Array1::from_vec(input_norm_bias),
            config.norm_eps,
        );

        // Load attention projections
        let q_proj = Self::load_weight(params, &format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_proj = Self::load_weight(params, &format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_proj = Self::load_weight(params, &format!("{}.self_attn.v_proj.weight", prefix))?;

        // Phi uses "dense" for output projection
        let o_proj_name = format!("{}.self_attn.dense.weight", prefix);
        let o_proj = if params.get_tensor_shape(&o_proj_name).is_some() {
            Self::load_weight(params, &o_proj_name)?
        } else {
            // Fall back to o_proj naming
            Self::load_weight(params, &format!("{}.self_attn.o_proj.weight", prefix))?
        };

        let self_attn = Attention::new(layer_idx, q_proj, k_proj, v_proj, o_proj);

        // Load MLP with fused gate+up projection
        let gate_up_name = format!("{}.mlp.gate_up_proj.weight", prefix);
        let gate_up_proj = if params.get_tensor_shape(&gate_up_name).is_some() {
            Self::load_weight(params, &gate_up_name)?
        } else {
            // Fall back to separate gate and up projections
            let gate = Self::load_weight(params, &format!("{}.mlp.gate_proj.weight", prefix))?;
            let up = Self::load_weight(params, &format!("{}.mlp.up_proj.weight", prefix))?;
            // Concatenate along dimension 0
            let mut fused = Array2::zeros((gate.shape()[0] + up.shape()[0], gate.shape()[1]));
            for i in 0..gate.shape()[0] {
                for j in 0..gate.shape()[1] {
                    fused[[i, j]] = gate[[i, j]];
                }
            }
            for i in 0..up.shape()[0] {
                for j in 0..up.shape()[1] {
                    fused[[i + gate.shape()[0], j]] = up[[i, j]];
                }
            }
            fused
        };
        let down_proj = Self::load_weight(params, &format!("{}.mlp.down_proj.weight", prefix))?;

        let mlp = PhiMLP::new(gate_up_proj, down_proj);

        Ok(PhiLayer::new(input_layernorm, self_attn, mlp))
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

        // Pass through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(state, i, debug);
        }

        // Final norm
        self.final_norm.forward(state);

        // LM head projection
        state
            .logits
            .assign(&kernels::matmul_vec(&self.lm_head, &state.hidden_state));
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
        self.final_norm.fast_forward(state);

        // LM head projection (parallel when available)
        state.logits.assign(&kernels::fast_matmul_vec(
            &self.lm_head,
            &state.hidden_state,
        ));
    }
}

impl Model for Phi {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Phi
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        Phi::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        Phi::fast_forward(self, state, token, debug)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }
}

/// Lazy Phi layer with eagerly loaded norms
pub struct LazyPhiLayer {
    pub input_layernorm: LayerNorm,
    pub layer_idx: usize,
}

/// Lazy Phi model with memory-mapped weights
pub struct LazyPhi<'a> {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub params: &'a Parameters,
    pub embedding: LazyEmbedding,
    pub layers: Vec<LazyPhiLayer>,
    pub final_norm: LayerNorm,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl<'a> LazyPhi<'a> {
    /// Load a lazy Phi model
    pub fn load(params: &'a Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let arch_config = ArchitectureConfig::phi();

        // Load final norm eagerly (small tensor)
        let final_norm_weight = "model.final_layernorm.weight";
        let final_norm_bias = "model.final_layernorm.bias";

        let norm_weight = params.get_tensor(final_norm_weight)?;
        let norm_bias = params
            .get_tensor(final_norm_bias)
            .unwrap_or_else(|_| vec![0.0; norm_weight.len()]);
        let final_norm = LayerNorm::new(
            Array1::from_vec(norm_weight),
            Array1::from_vec(norm_bias),
            config.norm_eps,
        );

        // Create lazy embedding
        let embedding = LazyEmbedding::new("model.embed_tokens.weight".to_string());

        // Create lazy layers with eagerly loaded norms
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            let prefix = format!("model.layers.{}", i);
            let norm_weight = params.get_tensor(&format!("{}.input_layernorm.weight", prefix))?;
            let norm_bias = params
                .get_tensor(&format!("{}.input_layernorm.bias", prefix))
                .unwrap_or_else(|_| vec![0.0; norm_weight.len()]);
            let input_layernorm = LayerNorm::new(
                Array1::from_vec(norm_weight),
                Array1::from_vec(norm_bias),
                config.norm_eps,
            );
            layers.push(LazyPhiLayer {
                input_layernorm,
                layer_idx: i,
            });
        }

        Ok(Self {
            config,
            arch_config,
            params,
            embedding,
            layers,
            final_norm,
            tokenizer,
        })
    }

    /// Forward pass with lazy loading
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Lazy embedding lookup
        self.embedding.forward(state, token, self.params);

        // Process each layer with lazy loading
        for layer in &self.layers {
            self.forward_layer(state, layer, debug);
        }

        // Final norm
        self.final_norm.forward(state);

        // LM head projection (lazy)
        let lm_head_view = self.params.get_tensor_view("lm_head.weight").unwrap();
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let logits = lm_head_view.matmul_vec(hidden_slice);
        for (i, &v) in logits.iter().enumerate() {
            state.logits[i] = v;
        }
    }

    fn forward_layer(&self, state: &mut InferenceState, layer: &LazyPhiLayer, debug: bool) {
        let layer_idx = layer.layer_idx;
        if debug && layer_idx % 8 == 0 {
            eprintln!("  Lazy Phi Layer {}/{}", layer_idx, self.config.n_layers);
        }

        let prefix = format!("model.layers.{}", layer_idx);

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Apply layer norm (using eagerly loaded weights)
        layer.input_layernorm.forward(state);

        // Save normalized for MLP
        let normed = state.hidden_state.clone();

        // Attention (lazy)
        let q_view = self
            .params
            .get_tensor_view(&format!("{}.self_attn.q_proj.weight", prefix))
            .unwrap();
        let k_view = self
            .params
            .get_tensor_view(&format!("{}.self_attn.k_proj.weight", prefix))
            .unwrap();
        let v_view = self
            .params
            .get_tensor_view(&format!("{}.self_attn.v_proj.weight", prefix))
            .unwrap();

        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let q = q_view.matmul_vec(hidden_slice);
        let k = k_view.matmul_vec(hidden_slice);
        let v = v_view.matmul_vec(hidden_slice);

        // Store Q/K/V (simplified attention for lazy mode)
        for (i, &val) in q.iter().enumerate() {
            if i < state.q_flat.len() {
                state.q_flat[i] = val;
            }
        }
        for (i, &val) in k.iter().enumerate() {
            if i < state.k_flat.len() {
                state.k_flat[i] = val;
            }
        }
        for (i, &val) in v.iter().enumerate() {
            if i < state.v_flat.len() {
                state.v_flat[i] = val;
            }
        }

        // Compute attention (simplified for demo)
        // Full implementation would use the attention module
        let o_name = format!("{}.self_attn.dense.weight", prefix);
        let o_view = self
            .params
            .get_tensor_view(&o_name)
            .or_else(|_| {
                self.params
                    .get_tensor_view(&format!("{}.self_attn.o_proj.weight", prefix))
            })
            .unwrap();

        // For now, use simplified attention output
        state
            .hidden_state
            .assign(&Array1::from_vec(o_view.matmul_vec(&q)));
        let attn_output = state.hidden_state.clone();

        // MLP branch (lazy)
        state.hidden_state.assign(&normed);

        let gate_up_name = format!("{}.mlp.gate_up_proj.weight", prefix);
        if let Ok(gate_up_view) = self.params.get_tensor_view(&gate_up_name) {
            let hidden_slice = state.hidden_state.as_slice().unwrap();
            let fused = gate_up_view.matmul_vec(hidden_slice);
            let intermediate_size = fused.len() / 2;

            let mut intermediate = vec![0.0f32; intermediate_size];
            for i in 0..intermediate_size {
                let gate = PhiMLP::gelu_tanh(fused[i]);
                let up = fused[i + intermediate_size];
                intermediate[i] = gate * up;
            }

            let down_view = self
                .params
                .get_tensor_view(&format!("{}.mlp.down_proj.weight", prefix))
                .unwrap();
            let mlp_out = down_view.matmul_vec(&intermediate);
            state.hidden_state.assign(&Array1::from_vec(mlp_out));
        }

        // Parallel residual
        state.hidden_state += &state.residual;
        state.hidden_state += &attn_output;
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Same as forward for lazy mode
        self.forward(state, token, debug)
    }
}

impl Model for LazyPhi<'_> {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Phi
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyPhi::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyPhi::fast_forward(self, state, token, debug)
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
    fn test_gelu_tanh() {
        // Test GELU approximation
        assert!((PhiMLP::gelu_tanh(0.0)).abs() < 0.01);
        assert!(PhiMLP::gelu_tanh(1.0) > 0.8);
        assert!(PhiMLP::gelu_tanh(-1.0) < -0.1);
    }

    #[test]
    fn test_phi_arch_config() {
        let arch = ArchitectureConfig::phi();
        assert_eq!(arch.architecture, ModelArchitecture::Phi);
        assert!(arch.parallel_residual);
        assert!(arch.fused_gate_up);
        assert_eq!(arch.activation, ActivationType::GELUTanh);
    }
}
