//! GPT-OSS Model Implementation (OpenAI)
//!
//! OpenAI's open-weight Mixture-of-Experts models (gpt-oss-120b, gpt-oss-20b).
//!
//! # Key Features
//! - Sparse Mixture-of-Experts (MoE) FFN on every layer (no dense layers)
//! - Attention sinks: per-head learnable scalar added to softmax (learned no-op target)
//! - Bias on all attention projections (Q, K, V, O) and MoE router
//! - Fused gate+up expert projections (interleaved in single tensor)
//! - MXFP4 quantization for expert weights (dequantized at load time)
//! - Alternating sliding window / full attention layers
//! - Clamped SwiGLU activation (alpha=1.702, limit=7.0)
//! - YaRN RoPE scaling
//! - RMSNorm
//!
//! # Architecture (gpt-oss-120b)
//! - 36 layers, all MoE
//! - 128 routed experts, top-4 per token
//! - hidden_size=2880, intermediate_size=2880
//! - 64 attention heads, 8 KV heads (GQA), head_dim=64
//! - vocab_size=201088
//!
//! # Usage
//! ```ignore
//! let params = Parameters::load("gpt-oss-20b/")?;
//! let model = GptOss::load(params)?;
//! let mut state = InferenceState::new(model.config.clone());
//! model.forward(&mut state, token, false);
//! ```

use crate::kernels;
use crate::loader::{Config, Parameters, WeightMatrix};
use crate::model::architecture::{ArchitectureConfig, Model, ModelArchitecture, TensorNamePattern};
use crate::model::modules::RMSNorm;
use crate::model::{Embedding, InferenceState};
use anyhow::Result;
use ndarray::{s, Array1, Array2};

// =============================================================================
// GPT-OSS Clamped SwiGLU Activation
// =============================================================================

/// Clamped SwiGLU activation as used in GPT-OSS.
///
/// The fused gate+up tensor interleaves gate (even indices) and linear (odd indices).
/// - gate: x_glu = clamp(x[::2], max=limit)
/// - linear: x_linear = clamp(x[1::2], min=-limit, max=limit)
/// - output = (x_glu * sigmoid(alpha * x_glu)) * (x_linear + 1)
///
/// alpha = 1.702, limit = swiglu_limit (default 7.0)
fn clamped_swiglu(fused: &[f32], limit: f32) -> Vec<f32> {
    let half = fused.len() / 2;
    let alpha: f32 = 1.702;
    let mut output = vec![0.0f32; half];

    for i in 0..half {
        let x_glu = fused[i * 2].min(limit); // clamp max=limit only
        let x_linear = fused[i * 2 + 1].clamp(-limit, limit);

        // SiLU-like with alpha: x * sigmoid(alpha * x)
        let gate = x_glu * (1.0 / (1.0 + (-alpha * x_glu).exp()));
        output[i] = gate * (x_linear + 1.0);
    }

    output
}

// =============================================================================
// GPT-OSS Attention (with biases and attention sinks)
// =============================================================================

/// Attention module for GPT-OSS with bias terms and attention sinks.
///
/// Attention sinks are per-head learnable scalars appended to the attention
/// score vector before softmax. After softmax, the sink probability is discarded.
/// This gives the model a learned "no-op" attention target that absorbs
/// irrelevant attention mass.
pub struct GptOssAttention {
    pub layer_idx: usize,
    pub q_proj: WeightMatrix,
    pub k_proj: WeightMatrix,
    pub v_proj: WeightMatrix,
    pub o_proj: WeightMatrix,
    pub q_bias: Array1<f32>,
    pub k_bias: Array1<f32>,
    pub v_bias: Array1<f32>,
    pub o_bias: Array1<f32>,
    /// Per-head attention sink values: [n_heads]
    pub sinks: Array1<f32>,
    /// Sliding window size (0 = full attention)
    pub sliding_window: usize,
}

impl GptOssAttention {
    /// Forward pass: multi-head attention with GQA, biases, and attention sinks
    pub fn forward(&self, state: &mut InferenceState) {
        let head_dim = state.config.hidden_size / state.config.n_heads;

        // Project to Q, K, V with bias
        self.q_proj.matmul_vec_into(
            state.hidden_state.as_slice().unwrap(),
            state.q_flat.as_slice_mut().unwrap(),
        );
        self.k_proj.matmul_vec_into(
            state.hidden_state.as_slice().unwrap(),
            state.k_flat.as_slice_mut().unwrap(),
        );
        self.v_proj.matmul_vec_into(
            state.hidden_state.as_slice().unwrap(),
            state.v_flat.as_slice_mut().unwrap(),
        );

        // Add biases
        for i in 0..state.q_flat.len() {
            state.q_flat[i] += self.q_bias[i];
        }
        for i in 0..state.k_flat.len() {
            state.k_flat[i] += self.k_bias[i];
        }
        for i in 0..state.v_flat.len() {
            state.v_flat[i] += self.v_bias[i];
        }

        // Copy flat buffers into shaped state arrays
        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.q_state[[h, d]] = state.q_flat[h * head_dim + d];
            }
        }
        for h in 0..state.config.n_kv_heads {
            for d in 0..head_dim {
                state.k_state[[h, d]] = state.k_flat[h * head_dim + d];
                state.v_state[[h, d]] = state.v_flat[h * head_dim + d];
            }
        }

        // Apply RoPE
        let (cos, sin) = kernels::rope_embeddings(&state.inv_freq, state.pos);
        state.cos.assign(&cos);
        state.sin.assign(&sin);
        kernels::apply_rope(&mut state.q_state, &state.cos, &state.sin);
        kernels::apply_rope(&mut state.k_state, &state.cos, &state.sin);

        // Push K, V to cache
        state.push_kv(self.layer_idx);

        // Attention computation with sinks
        let kv_groups = state.config.n_heads / state.config.n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        state.context.fill(0.0);

        for h in 0..state.config.n_heads {
            let kv_head = h / kv_groups;
            let seq_len = state.pos + 1;

            let q_head = state.q_state.row(h);
            let k_cache_view = state
                .k_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);

            // Compute attention scores
            {
                let mut scores_slice = state.scores.slice_mut(s![h, ..seq_len]);
                kernels::compute_attention_scores(q_head, k_cache_view, &mut scores_slice, scale);

                // Apply causal mask with optional sliding window
                if self.sliding_window > 0 {
                    let pos = state.pos;
                    for i in 0..seq_len {
                        if pos >= self.sliding_window && i < pos - self.sliding_window + 1 {
                            scores_slice[i] = f32::NEG_INFINITY;
                        }
                    }
                }

                // Softmax with attention sink appended
                // We append the sink value, compute softmax over [scores..., sink],
                // then discard the sink probability.
                let sink_val = self.sinks[h];
                let n = scores_slice.len();

                // Find max including sink
                let max_score = scores_slice
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
                    .max(sink_val);

                // Compute exp values and sum (including sink)
                let mut sum_exp = (sink_val - max_score).exp(); // sink exp
                for i in 0..n {
                    let e = (scores_slice[i] - max_score).exp();
                    scores_slice[i] = e;
                    sum_exp += e;
                }

                // Normalize (sink probability is implicitly discarded)
                let inv_sum = 1.0 / sum_exp;
                for i in 0..n {
                    scores_slice[i] *= inv_sum;
                }
            }

            // Weighted sum of V
            let v_cache_view = state
                .v_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);
            let scores_view = state.scores.slice(s![h, ..seq_len]);
            let mut context_row = state.context.slice_mut(s![h, ..]);
            kernels::weighted_sum_rows(scores_view, v_cache_view, &mut context_row);
        }

        // Flatten context
        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.context_flat[h * head_dim + d] = state.context[[h, d]];
            }
        }

        // Output projection with bias
        self.o_proj.matmul_vec_into(
            state.context_flat.as_slice().unwrap(),
            state.hidden_state.as_slice_mut().unwrap(),
        );
        for i in 0..state.hidden_state.len() {
            state.hidden_state[i] += self.o_bias[i];
        }
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState) {
        // For now, reuse the standard forward. Parallel optimization can be added later.
        self.forward(state);
    }
}

// =============================================================================
// GPT-OSS MoE (Batched Experts with Fused Gate+Up)
// =============================================================================

/// Router for GPT-OSS MoE with bias term.
pub struct GptOssMoERouter {
    pub gate_weight: WeightMatrix, // [n_experts, hidden_size]
    pub gate_bias: Array1<f32>,    // [n_experts]
    pub n_experts: usize,
    pub top_k: usize,
}

impl GptOssMoERouter {
    /// Compute top-k expert routing with bias.
    pub fn route(&self, state: &InferenceState) -> (Vec<usize>, Vec<f32>) {
        let logits_vec = self
            .gate_weight
            .matmul_vec(state.hidden_state.as_slice().unwrap());
        let mut router_logits = Array1::from_vec(logits_vec);
        // Add bias
        for i in 0..router_logits.len() {
            router_logits[i] += self.gate_bias[i];
        }
        self.top_k_softmax(&router_logits)
    }

    fn top_k_softmax(&self, logits: &Array1<f32>) -> (Vec<usize>, Vec<f32>) {
        let n = logits.len();
        assert!(self.top_k <= n, "top_k must be <= n_experts");

        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_indices: Vec<usize> = indexed[..self.top_k].iter().map(|(i, _)| *i).collect();
        let top_logits: Vec<f32> = indexed[..self.top_k].iter().map(|(_, v)| *v).collect();

        let max_logit = top_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = top_logits.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();
        let weights: Vec<f32> = exp_values.iter().map(|&v| v / sum_exp).collect();

        (top_indices, weights)
    }
}

/// A single GPT-OSS expert with fused gate+up projection and biases.
///
/// gate_up_proj: [intermediate_size*2, hidden_size] (interleaved gate and up)
/// down_proj: [hidden_size, intermediate_size]
/// Both have bias terms.
pub struct GptOssExpert {
    pub gate_up_proj: WeightMatrix,
    pub gate_up_bias: Array1<f32>,
    pub down_proj: WeightMatrix,
    pub down_bias: Array1<f32>,
}

/// GPT-OSS MoE layer: router + experts with batched weights.
pub struct GptOssMoE {
    pub router: GptOssMoERouter,
    pub experts: Vec<GptOssExpert>,
    pub swiglu_limit: f32,
}

impl GptOssMoE {
    /// Forward pass: route to top-k experts, run selected FFNs with clamped SwiGLU.
    pub fn forward(&self, state: &mut InferenceState) {
        let (expert_indices, expert_weights) = self.router.route(state);

        let input = state.hidden_state.clone();
        let hidden_size = state.hidden_state.len();
        let mut accumulated = Array1::<f32>::zeros(hidden_size);

        for (idx, &expert_idx) in expert_indices.iter().enumerate() {
            let weight = expert_weights[idx];
            let expert = &self.experts[expert_idx];

            // gate_up = gate_up_proj @ input + gate_up_bias
            let mut gate_up_vec = expert.gate_up_proj.matmul_vec(input.as_slice().unwrap());
            for (i, val) in gate_up_vec.iter_mut().enumerate() {
                *val += expert.gate_up_bias[i];
            }

            // Apply clamped SwiGLU
            let activated = clamped_swiglu(&gate_up_vec, self.swiglu_limit);

            // down = down_proj @ activated + down_bias
            let down = expert.down_proj.matmul_vec(&activated);

            for j in 0..hidden_size {
                accumulated[j] += weight * (down[j] + expert.down_bias[j]);
            }
        }

        state.hidden_state.assign(&accumulated);
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState) {
        // Reuse standard forward for now
        self.forward(state);
    }
}

// =============================================================================
// GPT-OSS Layer
// =============================================================================

/// A single GPT-OSS transformer layer: attention + MoE.
pub struct GptOssLayer {
    pub input_layernorm: RMSNorm,
    pub self_attn: GptOssAttention,
    pub post_attention_layernorm: RMSNorm,
    pub moe: GptOssMoE,
}

impl GptOssLayer {
    /// Forward pass: norm -> attention -> residual -> norm -> MoE -> residual
    pub fn forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  GPT-OSS Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm
        self.input_layernorm.forward(state);

        // Self-attention (with sinks and biases)
        self.self_attn.forward(state);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual again
        state.residual.assign(&state.hidden_state);

        // Pre-MoE norm
        self.post_attention_layernorm.forward(state);

        // MoE FFN
        self.moe.forward(state);

        // Residual connection
        state.hidden_state += &state.residual;
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  GPT-OSS Layer {}/{}", layer_idx, state.config.n_layers);
        }

        state.residual.assign(&state.hidden_state);
        self.input_layernorm.fast_forward(state);
        self.self_attn.fast_forward(state);
        state.hidden_state += &state.residual;

        state.residual.assign(&state.hidden_state);
        self.post_attention_layernorm.fast_forward(state);
        self.moe.fast_forward(state);
        state.hidden_state += &state.residual;
    }
}

// =============================================================================
// GPT-OSS Model (Eager Loading)
// =============================================================================

/// GPT-OSS model with eager weight loading.
///
/// All weights are dequantized from MXFP4 to f32 at load time.
/// Suitable for gpt-oss-20b on systems with sufficient RAM.
pub struct GptOss {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub embedding: Embedding,
    pub layers: Vec<GptOssLayer>,
    pub norm: RMSNorm,
    pub lm_head: WeightMatrix,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl GptOss {
    /// Load a GPT-OSS model from Parameters.
    pub fn load(params: Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let tensor_names = TensorNamePattern::gpt_oss();
        let arch_config = ArchitectureConfig::gpt_oss();

        let swiglu_limit = if config.swiglu_limit > 0.0 {
            config.swiglu_limit
        } else {
            7.0 // GPT-OSS default
        };

        // Load embedding table
        eprintln!("Loading GPT-OSS embedding table...");
        let embedding = Embedding::new(params.get_weight_matrix(tensor_names.embed_tokens)?);

        // Load final norm
        eprintln!("Loading GPT-OSS final norm...");
        let norm_data = params.get_tensor(tensor_names.final_norm)?;
        let norm = RMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Load LM head
        eprintln!("Loading GPT-OSS LM head...");
        let lm_head = params.get_weight_matrix(tensor_names.lm_head)?;

        // Determine sliding window from config
        let sliding_window = if config.attention_sliding_window > 0 {
            config.attention_sliding_window
        } else if config.sliding_window > 0 {
            config.sliding_window
        } else {
            128 // GPT-OSS default
        };

        // Load layers (all MoE)
        eprintln!("Loading {} GPT-OSS layers (all MoE)...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            if i % 4 == 0 {
                eprintln!("  Loading layer {}/{}...", i, config.n_layers);
            }
            // Alternating sliding window: even layers = sliding, odd layers = full
            let layer_sliding_window = if i % 2 == 0 { sliding_window } else { 0 };
            layers.push(Self::load_layer(
                &params,
                i,
                &config,
                &tensor_names,
                layer_sliding_window,
                swiglu_limit,
            )?);
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

    fn load_layer(
        params: &Parameters,
        layer_idx: usize,
        config: &Config,
        tensor_names: &TensorNamePattern,
        sliding_window: usize,
        swiglu_limit: f32,
    ) -> Result<GptOssLayer> {
        let prefix = tensor_names
            .layer_prefix
            .replace("{}", &layer_idx.to_string());

        // Load norms
        let input_norm_data = params.get_tensor(&tensor_names.input_layernorm_name(layer_idx))?;
        let input_layernorm = RMSNorm::new(Array1::from_vec(input_norm_data), config.norm_eps);

        let post_attn_norm_data =
            params.get_tensor(&tensor_names.post_attention_layernorm_name(layer_idx))?;
        let post_attention_layernorm =
            RMSNorm::new(Array1::from_vec(post_attn_norm_data), config.norm_eps);

        // Load attention (with biases and sinks)
        let self_attn = Self::load_attention(params, layer_idx, &prefix, sliding_window)?;

        // Load MoE
        let moe = Self::load_moe(params, layer_idx, config, &prefix, swiglu_limit)?;

        Ok(GptOssLayer {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            moe,
        })
    }

    fn load_attention(
        params: &Parameters,
        layer_idx: usize,
        prefix: &str,
        sliding_window: usize,
    ) -> Result<GptOssAttention> {
        let q_proj = Self::load_weight(params, &format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_proj = Self::load_weight(params, &format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_proj = Self::load_weight(params, &format!("{}.self_attn.v_proj.weight", prefix))?;
        let o_proj = Self::load_weight(params, &format!("{}.self_attn.o_proj.weight", prefix))?;

        let q_bias = Self::load_bias(params, &format!("{}.self_attn.q_proj.bias", prefix))?;
        let k_bias = Self::load_bias(params, &format!("{}.self_attn.k_proj.bias", prefix))?;
        let v_bias = Self::load_bias(params, &format!("{}.self_attn.v_proj.bias", prefix))?;
        let o_bias = Self::load_bias(params, &format!("{}.self_attn.o_proj.bias", prefix))?;

        let sinks_data = params.get_tensor(&format!("{}.self_attn.sinks", prefix))?;
        let sinks = Array1::from_vec(sinks_data);

        Ok(GptOssAttention {
            layer_idx,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_bias,
            k_bias,
            v_bias,
            o_bias,
            sinks,
            sliding_window,
        })
    }

    fn load_moe(
        params: &Parameters,
        layer_idx: usize,
        config: &Config,
        prefix: &str,
        swiglu_limit: f32,
    ) -> Result<GptOssMoE> {
        // Load router
        let router_weight = Self::load_weight(params, &format!("{}.mlp.router.weight", prefix))?;
        let router_bias = Self::load_bias(params, &format!("{}.mlp.router.bias", prefix))?;

        let n_experts = config.n_routed_experts;
        let top_k = config.n_experts_per_token;
        let router = GptOssMoERouter {
            gate_weight: router_weight,
            gate_bias: router_bias,
            n_experts,
            top_k,
        };

        // Load batched expert weights and split into individual experts.
        // Since we can't easily slice raw quantized bytes, use the F32 fallback:
        // load via get_tensor (dequantized), slice per expert, wrap with WeightMatrix::from_f32.
        //
        // Tensor shapes (from HuggingFace safetensors):
        // gate_up_proj: [n_experts, intermediate_size*2, hidden_size]
        // down_proj: [n_experts, hidden_size, intermediate_size]
        // gate_up_proj_bias: [n_experts, intermediate_size*2]
        // down_proj_bias: [n_experts, hidden_size]

        eprintln!(
            "    Loading {} experts for layer {}...",
            n_experts, layer_idx
        );

        let gate_up_name = format!("{}.mlp.experts.gate_up_proj", prefix);
        let down_name = format!("{}.mlp.experts.down_proj", prefix);
        let gate_up_bias_name = format!("{}.mlp.experts.gate_up_proj_bias", prefix);
        let down_bias_name = format!("{}.mlp.experts.down_proj_bias", prefix);

        let gate_up_data = params.get_tensor(&gate_up_name)?;
        let gate_up_shape = params.get_tensor_shape(&gate_up_name).unwrap();

        let down_data = params.get_tensor(&down_name)?;
        let down_shape = params.get_tensor_shape(&down_name).unwrap();

        let gate_up_bias_data = params.get_tensor(&gate_up_bias_name)?;
        let down_bias_data = params.get_tensor(&down_bias_name)?;

        let inter2 = gate_up_shape[1];
        let hidden = gate_up_shape[2];
        let down_hidden = down_shape[1];
        let inter = down_shape[2];

        let mut experts = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let gu_offset = e * inter2 * hidden;
            let gu_slice = &gate_up_data[gu_offset..gu_offset + inter2 * hidden];
            let gate_up_arr = Array2::from_shape_vec((inter2, hidden), gu_slice.to_vec())?;
            let gate_up_proj = WeightMatrix::from_f32(gate_up_arr);

            let gu_bias_offset = e * inter2;
            let gu_bias_slice = &gate_up_bias_data[gu_bias_offset..gu_bias_offset + inter2];
            let gate_up_bias = Array1::from_vec(gu_bias_slice.to_vec());

            let d_offset = e * down_hidden * inter;
            let d_slice = &down_data[d_offset..d_offset + down_hidden * inter];
            let down_arr = Array2::from_shape_vec((down_hidden, inter), d_slice.to_vec())?;
            let down_proj = WeightMatrix::from_f32(down_arr);

            let d_bias_offset = e * down_hidden;
            let d_bias_slice = &down_bias_data[d_bias_offset..d_bias_offset + down_hidden];
            let down_bias = Array1::from_vec(d_bias_slice.to_vec());

            experts.push(GptOssExpert {
                gate_up_proj,
                gate_up_bias,
                down_proj,
                down_bias,
            });
        }

        Ok(GptOssMoE {
            router,
            experts,
            swiglu_limit,
        })
    }

    fn load_weight(params: &Parameters, name: &str) -> Result<WeightMatrix> {
        params.get_weight_matrix(name)
    }

    fn load_bias(params: &Parameters, name: &str) -> Result<Array1<f32>> {
        let data = params.get_tensor(name)?;
        Ok(Array1::from_vec(data))
    }

    /// Forward pass
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        self.embedding.forward(state, token);

        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(state, i, debug);
        }

        self.norm.forward(state);

        self.lm_head.matmul_vec_into(
            state.hidden_state.as_slice().unwrap(),
            state.logits.as_slice_mut().unwrap(),
        );
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        self.embedding.forward(state, token);

        for (i, layer) in self.layers.iter().enumerate() {
            layer.fast_forward(state, i, debug);
        }

        self.norm.fast_forward(state);

        self.lm_head.matmul_vec_into(
            state.hidden_state.as_slice().unwrap(),
            state.logits.as_slice_mut().unwrap(),
        );
    }
}

impl Model for GptOss {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::GptOss
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        GptOss::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        GptOss::fast_forward(self, state, token, debug)
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }
}

// =============================================================================
// Lazy GPT-OSS - Memory-efficient model with lazy tensor loading
// =============================================================================

/// Lazy GPT-OSS model that keeps tensors memory-mapped and accesses them on-demand.
///
/// For GPT-OSS 120B (117B total parameters), eager loading would require enormous
/// RAM. The lazy variant only loads the router weights eagerly and accesses expert
/// weights on demand for the top-k selected experts per token.
pub struct LazyGptOss<'a> {
    pub config: Config,
    pub arch_config: ArchitectureConfig,
    pub params: &'a Parameters,
    pub layers: Vec<LazyGptOssLayer>,
    pub norm: RMSNorm,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

/// Lazy GPT-OSS layer: norms loaded eagerly, attention and MoE loaded on demand.
pub struct LazyGptOssLayer {
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub layer_prefix: String,
    pub sliding_window: usize,
    pub swiglu_limit: f32,
    pub n_experts: usize,
    pub top_k: usize,
    /// Router gate weight loaded eagerly (small tensor, needed every token)
    pub router_weight: Array2<f32>,
    pub router_bias: Array1<f32>,
}

impl<'a> LazyGptOss<'a> {
    /// Load a lazy GPT-OSS model. Only norms and router weights are loaded eagerly.
    pub fn load(params: &'a Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let tensor_names = TensorNamePattern::gpt_oss();
        let arch_config = ArchitectureConfig::gpt_oss();

        let swiglu_limit = if config.swiglu_limit > 0.0 {
            config.swiglu_limit
        } else {
            7.0
        };

        let sliding_window = if config.attention_sliding_window > 0 {
            config.attention_sliding_window
        } else if config.sliding_window > 0 {
            config.sliding_window
        } else {
            128
        };

        // Load final norm eagerly
        eprintln!("Loading GPT-OSS norms (lazy model)...");
        let norm_data = params.get_tensor(tensor_names.final_norm)?;
        let norm = RMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Create lazy layers
        eprintln!("Initializing {} lazy GPT-OSS layers...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            let prefix = tensor_names.layer_prefix.replace("{}", &i.to_string());

            // Load norms eagerly
            let input_norm_data = params.get_tensor(&tensor_names.input_layernorm_name(i))?;
            let input_layernorm = RMSNorm::new(Array1::from_vec(input_norm_data), config.norm_eps);

            let post_attn_norm_data =
                params.get_tensor(&tensor_names.post_attention_layernorm_name(i))?;
            let post_attention_layernorm =
                RMSNorm::new(Array1::from_vec(post_attn_norm_data), config.norm_eps);

            // Load router eagerly (small tensor, needed every forward pass)
            let router_weight_data = params.get_tensor(&format!("{}.mlp.router.weight", prefix))?;
            let router_weight_shape = params
                .get_tensor_shape(&format!("{}.mlp.router.weight", prefix))
                .unwrap();
            let router_weight = Array2::from_shape_vec(
                (router_weight_shape[0], router_weight_shape[1]),
                router_weight_data,
            )?;
            let router_bias_data = params.get_tensor(&format!("{}.mlp.router.bias", prefix))?;
            let router_bias = Array1::from_vec(router_bias_data);

            let layer_sliding_window = if i % 2 == 0 { sliding_window } else { 0 };

            layers.push(LazyGptOssLayer {
                input_layernorm,
                post_attention_layernorm,
                layer_prefix: prefix,
                sliding_window: layer_sliding_window,
                swiglu_limit,
                n_experts: config.n_routed_experts,
                top_k: config.n_experts_per_token,
                router_weight,
                router_bias,
            });
        }

        Ok(Self {
            config,
            arch_config,
            params,
            layers,
            norm,
            tokenizer,
        })
    }

    /// Forward pass with lazy tensor loading
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Lazy embedding lookup
        let embed_view = self
            .params
            .get_tensor_view("model.embed_tokens.weight")
            .unwrap();
        let embed_row = embed_view.get_row(token as usize);
        for (i, &v) in embed_row.iter().enumerate() {
            state.hidden_state[i] = v;
        }
        state.pos += 1;

        for (i, layer) in self.layers.iter().enumerate() {
            self.forward_layer(state, layer, i, debug);
        }

        self.norm.forward(state);

        // LM head (lazy)
        let lm_head_view = self.params.get_tensor_view("lm_head.weight").unwrap();
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let logits = lm_head_view.matmul_vec(hidden_slice);
        for (i, &v) in logits.iter().enumerate() {
            state.logits[i] = v;
        }
    }

    fn forward_layer(
        &self,
        state: &mut InferenceState,
        layer: &LazyGptOssLayer,
        layer_idx: usize,
        debug: bool,
    ) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!(
                "  Lazy GPT-OSS Layer {}/{}",
                layer_idx, state.config.n_layers
            );
        }

        let prefix = &layer.layer_prefix;
        let head_dim = state.config.hidden_size / state.config.n_heads;

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm
        layer.input_layernorm.forward(state);

        // === Lazy Attention ===
        // Load attention weights on demand
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

        // Q, K, V projections
        let q_flat = q_view.matmul_vec(hidden_slice);
        let k_flat = k_view.matmul_vec(hidden_slice);
        let v_flat = v_view.matmul_vec(hidden_slice);

        // Add biases
        let q_bias_data = self
            .params
            .get_tensor(&format!("{}.self_attn.q_proj.bias", prefix))
            .unwrap();
        let k_bias_data = self
            .params
            .get_tensor(&format!("{}.self_attn.k_proj.bias", prefix))
            .unwrap();
        let v_bias_data = self
            .params
            .get_tensor(&format!("{}.self_attn.v_proj.bias", prefix))
            .unwrap();

        for i in 0..q_flat.len() {
            state.q_flat[i] = q_flat[i] + q_bias_data[i];
        }
        for i in 0..k_flat.len() {
            state.k_flat[i] = k_flat[i] + k_bias_data[i];
        }
        for i in 0..v_flat.len() {
            state.v_flat[i] = v_flat[i] + v_bias_data[i];
        }

        // Copy to shaped arrays
        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.q_state[[h, d]] = state.q_flat[h * head_dim + d];
            }
        }
        for h in 0..state.config.n_kv_heads {
            for d in 0..head_dim {
                state.k_state[[h, d]] = state.k_flat[h * head_dim + d];
                state.v_state[[h, d]] = state.v_flat[h * head_dim + d];
            }
        }

        // RoPE
        let (cos, sin) = kernels::rope_embeddings(&state.inv_freq, state.pos);
        state.cos.assign(&cos);
        state.sin.assign(&sin);
        kernels::apply_rope(&mut state.q_state, &state.cos, &state.sin);
        kernels::apply_rope(&mut state.k_state, &state.cos, &state.sin);

        state.push_kv(layer_idx);

        // Attention with sinks
        let kv_groups = state.config.n_heads / state.config.n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let sinks_data = self
            .params
            .get_tensor(&format!("{}.self_attn.sinks", prefix))
            .unwrap();

        state.context.fill(0.0);

        for (h, &sink_val) in sinks_data.iter().enumerate().take(state.config.n_heads) {
            let kv_head = h / kv_groups;
            let seq_len = state.pos + 1;
            let q_head = state.q_state.row(h);
            let k_cache_view = state.k_cache.slice(s![layer_idx, kv_head, ..seq_len, ..]);

            {
                let mut scores_slice = state.scores.slice_mut(s![h, ..seq_len]);
                kernels::compute_attention_scores(q_head, k_cache_view, &mut scores_slice, scale);

                if layer.sliding_window > 0 {
                    let pos = state.pos;
                    for ii in 0..seq_len {
                        if pos >= layer.sliding_window && ii < pos - layer.sliding_window + 1 {
                            scores_slice[ii] = f32::NEG_INFINITY;
                        }
                    }
                }

                // Softmax with sink
                let n = scores_slice.len();
                let max_score = scores_slice
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)
                    .max(sink_val);
                let mut sum_exp = (sink_val - max_score).exp();
                for ii in 0..n {
                    let e = (scores_slice[ii] - max_score).exp();
                    scores_slice[ii] = e;
                    sum_exp += e;
                }
                let inv_sum = 1.0 / sum_exp;
                for ii in 0..n {
                    scores_slice[ii] *= inv_sum;
                }
            }

            let v_cache_view = state.v_cache.slice(s![layer_idx, kv_head, ..seq_len, ..]);
            let scores_view = state.scores.slice(s![h, ..seq_len]);
            let mut context_row = state.context.slice_mut(s![h, ..]);
            kernels::weighted_sum_rows(scores_view, v_cache_view, &mut context_row);
        }

        // Flatten context and output projection with bias
        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.context_flat[h * head_dim + d] = state.context[[h, d]];
            }
        }

        let o_view = self
            .params
            .get_tensor_view(&format!("{}.self_attn.o_proj.weight", prefix))
            .unwrap();
        let context_slice = state.context_flat.as_slice().unwrap();
        let o_result = o_view.matmul_vec(context_slice);
        let o_bias_data = self
            .params
            .get_tensor(&format!("{}.self_attn.o_proj.bias", prefix))
            .unwrap();
        for i in 0..state.hidden_state.len() {
            state.hidden_state[i] = o_result[i] + o_bias_data[i];
        }

        // Residual
        state.hidden_state += &state.residual;
        state.residual.assign(&state.hidden_state);

        // Pre-MoE norm
        layer.post_attention_layernorm.forward(state);

        // === Lazy MoE ===
        // Route using pre-loaded router
        let mut router_logits = kernels::matmul_vec(&layer.router_weight, &state.hidden_state);
        for i in 0..router_logits.len() {
            router_logits[i] += layer.router_bias[i];
        }

        // Top-k selection
        let mut indexed: Vec<(usize, f32)> = router_logits.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_indices: Vec<usize> = indexed[..layer.top_k].iter().map(|(i, _)| *i).collect();
        let top_logits: Vec<f32> = indexed[..layer.top_k].iter().map(|(_, v)| *v).collect();
        let max_logit = top_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = top_logits.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();
        let weights: Vec<f32> = exp_values.iter().map(|&v| v / sum_exp).collect();

        let input = state.hidden_state.clone();
        let hidden_size = state.hidden_state.len();
        let mut accumulated = Array1::<f32>::zeros(hidden_size);

        // Load and run selected experts lazily
        let gate_up_name = format!("{}.mlp.experts.gate_up_proj", prefix);
        let down_name = format!("{}.mlp.experts.down_proj", prefix);
        let gate_up_bias_name = format!("{}.mlp.experts.gate_up_proj_bias", prefix);
        let down_bias_name = format!("{}.mlp.experts.down_proj_bias", prefix);

        let gate_up_data = self.params.get_tensor(&gate_up_name).unwrap();
        let gate_up_shape = self.params.get_tensor_shape(&gate_up_name).unwrap();
        let down_data = self.params.get_tensor(&down_name).unwrap();
        let down_shape = self.params.get_tensor_shape(&down_name).unwrap();
        let gate_up_bias_data = self.params.get_tensor(&gate_up_bias_name).unwrap();
        let down_bias_data = self.params.get_tensor(&down_bias_name).unwrap();

        let inter2 = gate_up_shape[1];
        let hidden = gate_up_shape[2];
        let down_hidden = down_shape[1];
        let inter = down_shape[2];

        for (idx, &expert_idx) in top_indices.iter().enumerate() {
            let weight = weights[idx];

            // Slice expert gate_up_proj
            let gu_offset = expert_idx * inter2 * hidden;
            let gu_slice = &gate_up_data[gu_offset..gu_offset + inter2 * hidden];
            let gu_proj = Array2::from_shape_vec((inter2, hidden), gu_slice.to_vec()).unwrap();
            let gu_bias_offset = expert_idx * inter2;
            let gu_bias = &gate_up_bias_data[gu_bias_offset..gu_bias_offset + inter2];

            let gate_up = kernels::matmul_vec(&gu_proj, &input);
            let mut gate_up_vec: Vec<f32> = gate_up.to_vec();
            for i in 0..gate_up_vec.len() {
                gate_up_vec[i] += gu_bias[i];
            }

            let activated = clamped_swiglu(&gate_up_vec, layer.swiglu_limit);

            // Slice expert down_proj
            let d_offset = expert_idx * down_hidden * inter;
            let d_slice = &down_data[d_offset..d_offset + down_hidden * inter];
            let d_proj = Array2::from_shape_vec((down_hidden, inter), d_slice.to_vec()).unwrap();
            let d_bias_offset = expert_idx * down_hidden;
            let d_bias = &down_bias_data[d_bias_offset..d_bias_offset + down_hidden];

            let activated_arr = Array1::from_vec(activated);
            let down = kernels::matmul_vec(&d_proj, &activated_arr);

            for j in 0..hidden_size {
                accumulated[j] += weight * (down[j] + d_bias[j]);
            }
        }

        state.hidden_state.assign(&accumulated);

        // Residual
        state.hidden_state += &state.residual;
    }

    /// Optimized forward pass
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Reuse standard forward for now
        self.forward(state, token, debug);
    }
}

impl Model for LazyGptOss<'_> {
    fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::GptOss
    }

    fn config(&self) -> &Config {
        &self.config
    }

    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyGptOss::forward(self, state, token, debug)
    }

    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        LazyGptOss::fast_forward(self, state, token, debug)
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
    fn test_clamped_swiglu() {
        // Test with simple input: [gate0, linear0, gate1, linear1]
        let input = vec![1.0, 0.5, -1.0, 0.0];
        let result = clamped_swiglu(&input, 7.0);
        assert_eq!(result.len(), 2);
        // gate0=1.0, linear0=0.5: swish(1.0) * (0.5 + 1.0)
        let alpha = 1.702;
        let expected0 = (1.0 * (1.0 / (1.0 + (-alpha * 1.0_f32).exp()))) * 1.5;
        assert!(
            (result[0] - expected0).abs() < 1e-5,
            "got {}, expected {}",
            result[0],
            expected0
        );
    }

    #[test]
    fn test_clamped_swiglu_clamps() {
        // Values exceeding limit should be clamped
        let input = vec![10.0, 10.0, -10.0, -10.0];
        let result = clamped_swiglu(&input, 7.0);
        // gate0=10.0 clamped to 7.0, linear0=10.0 clamped to 7.0
        let alpha = 1.702;
        let expected0 = (7.0 * (1.0 / (1.0 + (-alpha * 7.0_f32).exp()))) * (7.0 + 1.0);
        assert!((result[0] - expected0).abs() < 1e-3);
    }

    #[test]
    fn test_gpt_oss_arch_config() {
        let config = ArchitectureConfig::gpt_oss();
        assert_eq!(config.architecture, ModelArchitecture::GptOss);
        assert!(config.is_moe);
        assert!(config.attention_bias);
    }
}
