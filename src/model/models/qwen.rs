//! Qwen Model Implementation (Phase 8)
//!
//! Alibaba's Qwen family of multilingual language models.
//!
//! # Key Features
//! - Fused Q/K/V projection (single tensor for all three)
//! - SwiGLU activation
//! - RMSNorm
//! - Grouped Query Attention (GQA)
//! - Rotary Position Embeddings (RoPE) with NTK-aware scaling
//! - Different tensor naming convention (transformer.h.*)
//!
//! # Architecture Differences from Mistral/LLaMA
//! - Fused QKV: Q, K, V are computed from a single projection
//! - Different naming: "transformer.h.*" instead of "model.layers.*"
//! - Different embedding name: "transformer.wte" instead of "model.embed_tokens"
//!
//! # Usage
//! ```ignore
//! let params = Parameters::load("qwen-7b.bin")?;
//! let model = Qwen::load(params)?;
//! let mut state = InferenceState::new(model.config.clone());
//! model.forward(&mut state, token, false);
//! ```
#![allow(clippy::needless_range_loop)]

use crate::kernels;
use crate::loader::{Config, Parameters, WeightMatrix};
use crate::model::architecture::{ArchitectureConfig, Model, ModelArchitecture};
use crate::model::LazyEmbedding;
use crate::model::{Embedding, InferenceState, RMSNorm, MLP};
use anyhow::Result;
use ndarray::{s, Array1, Array2};

/// Qwen-style attention with fused QKV projection
pub struct QwenAttention {
    pub layer_idx: usize,
    /// Fused Q/K/V projection: [(n_heads + 2*n_kv_heads) * head_dim, hidden_size]
    pub c_attn: WeightMatrix,
    /// Output projection: [hidden_size, n_heads * head_dim]
    pub c_proj: WeightMatrix,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of KV heads (for GQA)
    pub n_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl QwenAttention {
    pub fn new(
        layer_idx: usize,
        c_attn: WeightMatrix,
        c_proj: WeightMatrix,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            layer_idx,
            c_attn,
            c_proj,
            n_heads,
            n_kv_heads,
            head_dim,
        }
    }

    /// Forward pass with fused QKV
    pub fn forward(&self, state: &mut InferenceState) {
        // Fused QKV projection
        let qkv = self
            .c_attn
            .matmul_vec(state.hidden_state.as_slice().unwrap());

        // Split into Q, K, V
        let q_size = self.n_heads * self.head_dim;
        let kv_size = self.n_kv_heads * self.head_dim;

        // Copy to state buffers
        for i in 0..q_size {
            state.q_flat[i] = qkv[i];
        }
        for i in 0..kv_size {
            state.k_flat[i] = qkv[q_size + i];
            state.v_flat[i] = qkv[q_size + kv_size + i];
        }

        // Reshape to heads
        for h in 0..self.n_heads {
            for d in 0..self.head_dim {
                state.q_state[[h, d]] = state.q_flat[h * self.head_dim + d];
            }
        }
        for h in 0..self.n_kv_heads {
            for d in 0..self.head_dim {
                state.k_state[[h, d]] = state.k_flat[h * self.head_dim + d];
                state.v_state[[h, d]] = state.v_flat[h * self.head_dim + d];
            }
        }

        // Generate RoPE embeddings
        let (cos, sin) = kernels::rope_embeddings(&state.inv_freq, state.pos);
        state.cos.assign(&cos);
        state.sin.assign(&sin);

        // Apply RoPE
        kernels::apply_rope(&mut state.q_state, &state.cos, &state.sin);
        kernels::apply_rope(&mut state.k_state, &state.cos, &state.sin);

        // Push to KV cache
        state.push_kv(self.layer_idx);

        // Attention computation
        let kv_groups = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let seq_len = state.pos + 1;

        state.context.fill(0.0);

        for h in 0..self.n_heads {
            let kv_head = h / kv_groups;

            let q_head = state.q_state.row(h);
            let k_cache_view = state
                .k_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);

            // Compute scores
            {
                let mut scores_slice = state.scores.slice_mut(s![h, ..seq_len]);
                kernels::compute_attention_scores(q_head, k_cache_view, &mut scores_slice, scale);
                kernels::softmax_view(&mut scores_slice);
            }

            // Weighted sum
            let v_cache_view = state
                .v_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);
            let scores_view = state.scores.slice(s![h, ..seq_len]);
            let mut context_row = state.context.slice_mut(s![h, ..]);
            kernels::weighted_sum_rows(scores_view, v_cache_view, &mut context_row);
        }

        // Flatten context
        for h in 0..self.n_heads {
            for d in 0..self.head_dim {
                state.context_flat[h * self.head_dim + d] = state.context[[h, d]];
            }
        }

        // Output projection
        self.c_proj.matmul_vec_into(
            state.context_flat.as_slice().unwrap(),
            state.hidden_state.as_slice_mut().unwrap(),
        );
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState) {
        // Fused QKV projection (parallel when available)
        let qkv = self
            .c_attn
            .matmul_vec(state.hidden_state.as_slice().unwrap());

        // Split into Q, K, V
        let q_size = self.n_heads * self.head_dim;
        let kv_size = self.n_kv_heads * self.head_dim;

        for i in 0..q_size {
            state.q_flat[i] = qkv[i];
        }
        for i in 0..kv_size {
            state.k_flat[i] = qkv[q_size + i];
            state.v_flat[i] = qkv[q_size + kv_size + i];
        }

        // Reshape to heads
        for h in 0..self.n_heads {
            for d in 0..self.head_dim {
                state.q_state[[h, d]] = state.q_flat[h * self.head_dim + d];
            }
        }
        for h in 0..self.n_kv_heads {
            for d in 0..self.head_dim {
                state.k_state[[h, d]] = state.k_flat[h * self.head_dim + d];
                state.v_state[[h, d]] = state.v_flat[h * self.head_dim + d];
            }
        }

        // Generate RoPE embeddings
        let (cos, sin) = kernels::rope_embeddings(&state.inv_freq, state.pos);
        state.cos.assign(&cos);
        state.sin.assign(&sin);

        // Apply RoPE
        kernels::apply_rope(&mut state.q_state, &state.cos, &state.sin);
        kernels::apply_rope(&mut state.k_state, &state.cos, &state.sin);

        // Push to KV cache (optimized)
        state.push_kv_optimized(self.layer_idx);

        // Attention computation
        let kv_groups = self.n_heads / self.n_kv_heads;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let seq_len = state.pos + 1;

        state.context.fill(0.0);

        for h in 0..self.n_heads {
            let kv_head = h / kv_groups;

            let q_head = state.q_state.row(h);
            let k_cache_view = state
                .k_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);

            {
                let mut scores_slice = state.scores.slice_mut(s![h, ..seq_len]);
                kernels::compute_attention_scores(q_head, k_cache_view, &mut scores_slice, scale);
                kernels::softmax_view(&mut scores_slice);
            }

            let v_cache_view = state
                .v_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);
            let scores_view = state.scores.slice(s![h, ..seq_len]);
            let mut context_row = state.context.slice_mut(s![h, ..]);
            kernels::weighted_sum_rows(scores_view, v_cache_view, &mut context_row);
        }

        // Flatten context
        for h in 0..self.n_heads {
            for d in 0..self.head_dim {
                state.context_flat[h * self.head_dim + d] = state.context[[h, d]];
            }
        }

        // Output projection (parallel when available)
        self.c_proj.matmul_vec_into(
            state.context_flat.as_slice().unwrap(),
            state.hidden_state.as_slice_mut().unwrap(),
        );
    }
}

/// Qwen transformer layer
pub struct QwenLayer {
    pub ln_1: RMSNorm,
    pub attn: QwenAttention,
    pub ln_2: RMSNorm,
    pub mlp: MLP,
}

impl QwenLayer {
    pub fn new(ln_1: RMSNorm, attn: QwenAttention, ln_2: RMSNorm, mlp: MLP) -> Self {
        Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        }
    }

    /// Forward pass
    pub fn forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  Qwen Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm
        self.ln_1.forward(state);

        // Attention
        self.attn.forward(state);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-MLP norm
        self.ln_2.forward(state);

        // MLP (SwiGLU)
        self.mlp.forward(state);

        // Residual connection
        state.hidden_state += &state.residual;
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  Qwen Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm (SIMD when available)
        self.ln_1.fast_forward(state);

        // Attention (optimized)
        self.attn.fast_forward(state);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-MLP norm (SIMD when available)
        self.ln_2.fast_forward(state);

        // MLP (parallel when available)
        self.mlp.fast_forward(state);

        // Residual connection
        state.hidden_state += &state.residual;
    }
}

/// Qwen model with eager weight loading
pub struct Qwen {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub embedding: Embedding,
    pub layers: Vec<QwenLayer>,
    pub ln_f: RMSNorm,
    pub lm_head: WeightMatrix,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl Qwen {
    /// Load a Qwen model from Parameters
    pub fn load(params: Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let arch_config = ArchitectureConfig::qwen();

        // Qwen tensor names
        let embed_name = Self::get_embed_name(&params);
        let ln_f_name = Self::get_ln_f_name(&params);
        let lm_head_name = "lm_head.weight";

        // Load embedding table
        eprintln!("Loading Qwen embedding table...");
        let embedding = Embedding::new(params.get_weight_matrix(&embed_name)?);

        // Load final norm
        eprintln!("Loading Qwen final norm...");
        let ln_f_data = params.get_tensor(&ln_f_name)?;
        let ln_f = RMSNorm::new(Array1::from_vec(ln_f_data), config.norm_eps);

        // Load LM head
        eprintln!("Loading Qwen LM head...");
        let lm_head = params.get_weight_matrix(lm_head_name)?;

        // Load layers
        eprintln!("Loading {} Qwen layers...", config.n_layers);
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
            ln_f,
            lm_head,
            tokenizer,
        })
    }

    fn get_embed_name(params: &Parameters) -> String {
        // Try Qwen naming first, fall back to standard
        if params.get_tensor_shape("transformer.wte.weight").is_some() {
            "transformer.wte.weight".to_string()
        } else {
            "model.embed_tokens.weight".to_string()
        }
    }

    fn get_ln_f_name(params: &Parameters) -> String {
        if params.get_tensor_shape("transformer.ln_f.weight").is_some() {
            "transformer.ln_f.weight".to_string()
        } else {
            "model.norm.weight".to_string()
        }
    }

    fn get_layer_prefix(params: &Parameters, layer_idx: usize) -> String {
        // Try Qwen naming first
        let qwen_prefix = format!("transformer.h.{}", layer_idx);
        let standard_prefix = format!("model.layers.{}", layer_idx);

        if params
            .get_tensor_shape(&format!("{}.ln_1.weight", qwen_prefix))
            .is_some()
        {
            qwen_prefix
        } else {
            standard_prefix
        }
    }

    fn load_layer(params: &Parameters, layer_idx: usize, config: &Config) -> Result<QwenLayer> {
        let prefix = Self::get_layer_prefix(params, layer_idx);
        let head_dim = config.hidden_size / config.n_heads;

        // Load norms
        let ln1_name = format!("{}.ln_1.weight", prefix);
        let ln1_fallback = format!("{}.input_layernorm.weight", prefix);
        let ln_1_data = params
            .get_tensor(&ln1_name)
            .or_else(|_| params.get_tensor(&ln1_fallback))?;
        let ln_1 = RMSNorm::new(Array1::from_vec(ln_1_data), config.norm_eps);

        let ln2_name = format!("{}.ln_2.weight", prefix);
        let ln2_fallback = format!("{}.post_attention_layernorm.weight", prefix);
        let ln_2_data = params
            .get_tensor(&ln2_name)
            .or_else(|_| params.get_tensor(&ln2_fallback))?;
        let ln_2 = RMSNorm::new(Array1::from_vec(ln_2_data), config.norm_eps);

        // Load attention (try fused QKV first, fall back to separate)
        let c_attn_name = format!("{}.attn.c_attn.weight", prefix);
        let c_proj_name = format!("{}.attn.c_proj.weight", prefix);

        let attn = if params.get_tensor_shape(&c_attn_name).is_some() {
            // Fused QKV
            let c_attn = Self::load_weight(params, &c_attn_name)?;
            let c_proj = Self::load_weight(params, &c_proj_name)?;
            QwenAttention::new(
                layer_idx,
                c_attn,
                c_proj,
                config.n_heads,
                config.n_kv_heads,
                head_dim,
            )
        } else {
            // Separate Q, K, V (fall back to standard naming) - stack into F32 WeightMatrix
            let q_data = params.get_tensor(&format!("{}.self_attn.q_proj.weight", prefix))?;
            let q_shape = params
                .get_tensor_shape(&format!("{}.self_attn.q_proj.weight", prefix))
                .unwrap();
            let k_data = params.get_tensor(&format!("{}.self_attn.k_proj.weight", prefix))?;
            let k_shape = params
                .get_tensor_shape(&format!("{}.self_attn.k_proj.weight", prefix))
                .unwrap();
            let v_data = params.get_tensor(&format!("{}.self_attn.v_proj.weight", prefix))?;
            let v_shape = params
                .get_tensor_shape(&format!("{}.self_attn.v_proj.weight", prefix))
                .unwrap();

            let q_size = q_shape[0];
            let k_size = k_shape[0];
            let v_size = v_shape[0];
            let hidden = q_shape[1];

            let mut c_attn = Array2::zeros((q_size + k_size + v_size, hidden));
            let q_arr = Array2::from_shape_vec((q_size, hidden), q_data)?;
            let k_arr = Array2::from_shape_vec((k_size, hidden), k_data)?;
            let v_arr = Array2::from_shape_vec((v_size, hidden), v_data)?;
            c_attn.slice_mut(s![..q_size, ..]).assign(&q_arr);
            c_attn
                .slice_mut(s![q_size..q_size + k_size, ..])
                .assign(&k_arr);
            c_attn.slice_mut(s![q_size + k_size.., ..]).assign(&v_arr);

            let c_attn = WeightMatrix::from_f32(c_attn);
            let c_proj =
                params.get_weight_matrix(&format!("{}.self_attn.o_proj.weight", prefix))?;

            QwenAttention::new(
                layer_idx,
                c_attn,
                c_proj,
                config.n_heads,
                config.n_kv_heads,
                head_dim,
            )
        };

        // Load MLP (try Qwen naming first)
        let gate_name = format!("{}.mlp.w1.weight", prefix);
        let up_name = format!("{}.mlp.w2.weight", prefix);
        let down_name = format!("{}.mlp.c_proj.weight", prefix);

        let mlp = if params.get_tensor_shape(&gate_name).is_some() {
            let gate = Self::load_weight(params, &gate_name)?;
            let up = Self::load_weight(params, &up_name)?;
            let down = Self::load_weight(params, &down_name)?;
            MLP::new(gate, up, down)
        } else {
            // Fall back to standard naming
            let gate = Self::load_weight(params, &format!("{}.mlp.gate_proj.weight", prefix))?;
            let up = Self::load_weight(params, &format!("{}.mlp.up_proj.weight", prefix))?;
            let down = Self::load_weight(params, &format!("{}.mlp.down_proj.weight", prefix))?;
            MLP::new(gate, up, down)
        };

        Ok(QwenLayer::new(ln_1, attn, ln_2, mlp))
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
        self.ln_f.forward(state);

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

        // Final norm (SIMD when available)
        self.ln_f.fast_forward(state);

        // LM head projection (parallel when available)
        self.lm_head.matmul_vec_into(
            state.hidden_state.as_slice().unwrap(),
            state.logits.as_slice_mut().unwrap(),
        );
    }
}

impl Model for Qwen {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Qwen
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        Qwen::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        Qwen::fast_forward(self, state, token, debug)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }
}

/// Lazy Qwen layer with eagerly loaded norms
pub struct LazyQwenLayer {
    pub ln_1: RMSNorm,
    pub ln_2: RMSNorm,
    pub layer_idx: usize,
}

/// Lazy Qwen model with memory-mapped weights
pub struct LazyQwen<'a> {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub params: &'a Parameters,
    pub embedding: LazyEmbedding,
    pub layers: Vec<LazyQwenLayer>,
    pub ln_f: RMSNorm,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl<'a> LazyQwen<'a> {
    /// Load a lazy Qwen model
    pub fn load(params: &'a Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let arch_config = ArchitectureConfig::qwen();

        let embed_name = Qwen::get_embed_name(params);
        let ln_f_name = Qwen::get_ln_f_name(params);

        // Load final norm eagerly (small tensor)
        let ln_f_data = params.get_tensor(&ln_f_name)?;
        let ln_f = RMSNorm::new(Array1::from_vec(ln_f_data), config.norm_eps);

        // Create lazy embedding
        let embedding = LazyEmbedding::new(embed_name);

        // Create lazy layers with eagerly loaded norms
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            let prefix = Qwen::get_layer_prefix(params, i);

            let ln1_name = format!("{}.ln_1.weight", prefix);
            let ln1_fallback = format!("{}.input_layernorm.weight", prefix);
            let ln1_data = params
                .get_tensor(&ln1_name)
                .or_else(|_| params.get_tensor(&ln1_fallback))?;
            let ln_1 = RMSNorm::new(Array1::from_vec(ln1_data), config.norm_eps);

            let ln2_name = format!("{}.ln_2.weight", prefix);
            let ln2_fallback = format!("{}.post_attention_layernorm.weight", prefix);
            let ln2_data = params
                .get_tensor(&ln2_name)
                .or_else(|_| params.get_tensor(&ln2_fallback))?;
            let ln_2 = RMSNorm::new(Array1::from_vec(ln2_data), config.norm_eps);

            layers.push(LazyQwenLayer {
                ln_1,
                ln_2,
                layer_idx: i,
            });
        }

        Ok(Self {
            config,
            arch_config,
            params,
            embedding,
            layers,
            ln_f,
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
        self.ln_f.forward(state);

        // LM head projection (lazy)
        let lm_head_view = self.params.get_tensor_view("lm_head.weight").unwrap();
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let logits = lm_head_view.matmul_vec(hidden_slice);
        for (i, &v) in logits.iter().enumerate() {
            state.logits[i] = v;
        }
    }

    fn forward_layer(&self, state: &mut InferenceState, layer: &LazyQwenLayer, debug: bool) {
        let layer_idx = layer.layer_idx;
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  Lazy Qwen Layer {}/{}", layer_idx, self.config.n_layers);
        }

        let prefix = Qwen::get_layer_prefix(self.params, layer_idx);
        let head_dim = self.config.hidden_size / self.config.n_heads;

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Apply ln_1 (using eagerly loaded weights)
        layer.ln_1.forward(state);

        // Lazy attention with fused QKV
        let c_attn_name = format!("{}.attn.c_attn.weight", prefix);
        let c_proj_name = format!("{}.attn.c_proj.weight", prefix);

        if let Ok(c_attn_view) = self.params.get_tensor_view(&c_attn_name) {
            let hidden_slice = state.hidden_state.as_slice().unwrap();
            let qkv = c_attn_view.matmul_vec(hidden_slice);

            let q_size = self.config.n_heads * head_dim;
            let kv_size = self.config.n_kv_heads * head_dim;

            // Split QKV
            for i in 0..q_size.min(state.q_flat.len()) {
                state.q_flat[i] = qkv[i];
            }
            for i in 0..kv_size.min(state.k_flat.len()) {
                state.k_flat[i] = qkv[q_size + i];
                state.v_flat[i] = qkv[q_size + kv_size + i];
            }

            // Simplified attention output (full implementation would use attention module)
            let c_proj_view = self.params.get_tensor_view(&c_proj_name).unwrap();
            let attn_out = c_proj_view.matmul_vec(&qkv[..q_size]);
            state.hidden_state.assign(&Array1::from_vec(attn_out));
        } else {
            // Fall back to separate Q, K, V
            let q_view = self
                .params
                .get_tensor_view(&format!("{}.self_attn.q_proj.weight", prefix))
                .unwrap();
            let hidden_slice = state.hidden_state.as_slice().unwrap();
            let q = q_view.matmul_vec(hidden_slice);

            let o_view = self
                .params
                .get_tensor_view(&format!("{}.self_attn.o_proj.weight", prefix))
                .unwrap();
            let attn_out = o_view.matmul_vec(&q);
            state.hidden_state.assign(&Array1::from_vec(attn_out));
        }

        // Residual
        state.hidden_state += &state.residual;

        // Save residual for MLP
        state.residual.assign(&state.hidden_state);

        // Apply ln_2 (using eagerly loaded weights)
        layer.ln_2.forward(state);

        // Lazy MLP
        let gate_name = format!("{}.mlp.w1.weight", prefix);
        let up_name = format!("{}.mlp.w2.weight", prefix);
        let down_name = format!("{}.mlp.c_proj.weight", prefix);

        if let Ok(gate_view) = self.params.get_tensor_view(&gate_name) {
            let hidden_slice = state.hidden_state.as_slice().unwrap();
            let gate = gate_view.matmul_vec(hidden_slice);

            let up_view = self.params.get_tensor_view(&up_name).unwrap();
            let up = up_view.matmul_vec(hidden_slice);

            // SwiGLU
            let intermediate: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(&g, &u)| kernels::silu_scalar(g) * u)
                .collect();

            let down_view = self.params.get_tensor_view(&down_name).unwrap();
            let mlp_out = down_view.matmul_vec(&intermediate);
            state.hidden_state.assign(&Array1::from_vec(mlp_out));
        } else {
            // Fall back to standard naming
            let gate_view = self
                .params
                .get_tensor_view(&format!("{}.mlp.gate_proj.weight", prefix))
                .unwrap();
            let hidden_slice = state.hidden_state.as_slice().unwrap();
            let gate = gate_view.matmul_vec(hidden_slice);

            let up_view = self
                .params
                .get_tensor_view(&format!("{}.mlp.up_proj.weight", prefix))
                .unwrap();
            let up = up_view.matmul_vec(hidden_slice);

            let intermediate: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(&g, &u)| kernels::silu_scalar(g) * u)
                .collect();

            let down_view = self
                .params
                .get_tensor_view(&format!("{}.mlp.down_proj.weight", prefix))
                .unwrap();
            let mlp_out = down_view.matmul_vec(&intermediate);
            state.hidden_state.assign(&Array1::from_vec(mlp_out));
        }

        // Final residual
        state.hidden_state += &state.residual;
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        self.forward(state, token, debug)
    }
}

impl Model for LazyQwen<'_> {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Qwen
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyQwen::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyQwen::fast_forward(self, state, token, debug)
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
    fn test_qwen_arch_config() {
        let arch = ArchitectureConfig::qwen();
        assert_eq!(arch.architecture, ModelArchitecture::Qwen);
        assert!(arch.fused_qkv);
        assert!(!arch.tie_embeddings);
    }
}
