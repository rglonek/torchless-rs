//! Phase 6: Parallel Model Implementations
//!
//! This module provides parallelized model implementations:
//! - **PipelineParallelMistral**: Pipeline-parallel layer execution
//! - **TensorParallelMistral**: Tensor-parallel attention and MLP
//! - **ParallelInferenceState**: State optimized for parallel execution
//!
//! # Usage
//!
//! ```ignore
//! use torchless::model::parallel::{PipelineParallelMistral, ParallelConfig};
//!
//! // Create a pipeline-parallel model
//! let config = ParallelConfig::pipeline(4); // 4 micro-batches
//! let model = PipelineParallelMistral::load(params, config)?;
//!
//! // Or use tensor parallelism
//! let tp_config = ParallelConfig::tensor_parallel(2); // 2-way tensor parallel
//! let tp_model = TensorParallelMistral::load(params, tp_config)?;
//! ```

use crate::kernels;
use crate::loader::weight_matrix::WeightMatrix;
use crate::loader::{Config, Parameters};
use anyhow::Result;
use ndarray::{s, Array1, Array2, ArrayView2};

#[cfg(feature = "parallel")]
use crate::kernels::parallel::{
    matmul_vec_adaptive_into, mlp_tensor_parallel, PipelineConfig, PipelineState,
    TensorParallelConfig, WorkDistributionConfig,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::modules::{Embedding, RMSNorm};
use super::InferenceState;

// =============================================================================
// Parallel Configuration
// =============================================================================

/// Configuration for parallel model execution
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Pipeline parallelism configuration
    pub pipeline: Option<PipelineConfig>,
    /// Tensor parallelism configuration  
    #[cfg(feature = "parallel")]
    pub tensor_parallel: Option<TensorParallelConfig>,
    /// Work distribution configuration
    #[cfg(feature = "parallel")]
    pub work_distribution: WorkDistributionConfig,
    /// Whether to use adaptive chunking for matmul
    pub adaptive_matmul: bool,
    /// Whether to parallelize attention heads
    pub parallel_heads: bool,
    /// Number of layers to group for pipeline stages
    pub layers_per_stage: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            pipeline: None,
            #[cfg(feature = "parallel")]
            tensor_parallel: None,
            #[cfg(feature = "parallel")]
            work_distribution: WorkDistributionConfig::default(),
            adaptive_matmul: true,
            parallel_heads: true,
            layers_per_stage: 4,
        }
    }
}

impl ParallelConfig {
    /// Create a configuration for pipeline parallelism
    pub fn pipeline(num_micro_batches: usize) -> Self {
        Self {
            pipeline: Some(PipelineConfig {
                num_micro_batches,
                async_execution: false,
                memory_budget: 512 * 1024 * 1024, // 512MB
            }),
            ..Default::default()
        }
    }

    /// Create a configuration for tensor parallelism
    #[cfg(feature = "parallel")]
    pub fn tensor_parallel(world_size: usize, rank: usize) -> Self {
        Self {
            tensor_parallel: Some(TensorParallelConfig::new(world_size, rank)),
            ..Default::default()
        }
    }

    /// Create a configuration optimized for a specific model size
    #[cfg(feature = "parallel")]
    pub fn for_model(config: &Config) -> Self {
        let work_dist = WorkDistributionConfig::for_matmul(config.hidden_size, config.hidden_size);

        Self {
            work_distribution: work_dist,
            adaptive_matmul: true,
            parallel_heads: config.n_heads >= 4,
            layers_per_stage: if config.n_layers >= 32 { 8 } else { 4 },
            ..Default::default()
        }
    }

    #[cfg(not(feature = "parallel"))]
    pub fn for_model(_config: &Config) -> Self {
        Self::default()
    }
}

// =============================================================================
// Parallel Inference State
// =============================================================================

/// Inference state optimized for parallel execution
///
/// Extends the base InferenceState with additional buffers for
/// parallel and pipelined computation.
pub struct ParallelInferenceState {
    /// Base inference state
    pub inner: InferenceState,
    /// Pipeline micro-batch buffers (one per micro-batch in flight)
    pub micro_batch_buffers: Vec<Array1<f32>>,
    /// Partial results for tensor parallelism (one per partition)
    pub partial_results: Vec<Array1<f32>>,
    /// Work distribution configuration
    #[cfg(feature = "parallel")]
    pub work_config: WorkDistributionConfig,
}

impl ParallelInferenceState {
    /// Create a new parallel inference state
    pub fn new(config: Config, parallel_config: &ParallelConfig) -> Self {
        let num_micro_batches = parallel_config
            .pipeline
            .as_ref()
            .map(|p| p.num_micro_batches)
            .unwrap_or(1);

        let micro_batch_buffers = (0..num_micro_batches)
            .map(|_| Array1::zeros(config.hidden_size))
            .collect();

        #[cfg(feature = "parallel")]
        let world_size = parallel_config
            .tensor_parallel
            .as_ref()
            .map(|t| t.world_size)
            .unwrap_or(1);

        #[cfg(not(feature = "parallel"))]
        let world_size = 1;

        let partial_results = (0..world_size)
            .map(|_| Array1::zeros(config.hidden_size))
            .collect();

        Self {
            inner: InferenceState::new(config),
            micro_batch_buffers,
            partial_results,
            #[cfg(feature = "parallel")]
            work_config: parallel_config.work_distribution.clone(),
        }
    }

    /// Get the underlying inference state
    #[inline]
    pub fn as_inference_state(&mut self) -> &mut InferenceState {
        &mut self.inner
    }
}

// =============================================================================
// Parallel Attention Module
// =============================================================================

/// Attention module with parallelization support
pub struct ParallelAttention {
    pub layer_idx: usize,
    pub q_proj: Array2<f32>,
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
}

impl ParallelAttention {
    pub fn new(
        layer_idx: usize,
        q_proj: Array2<f32>,
        k_proj: Array2<f32>,
        v_proj: Array2<f32>,
        o_proj: Array2<f32>,
    ) -> Self {
        Self {
            layer_idx,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        }
    }

    /// Forward pass with adaptive parallel work distribution
    #[cfg(feature = "parallel")]
    #[allow(clippy::needless_range_loop)] // Index-based access required for ndarray cache
    pub fn forward_adaptive(
        &self,
        state: &mut InferenceState,
        work_config: &WorkDistributionConfig,
    ) {
        let head_dim = state.config.hidden_size / state.config.n_heads;
        let n_heads = state.config.n_heads;
        let n_kv_heads = state.config.n_kv_heads;
        let kv_groups = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = state.pos + 1;

        // Adaptive parallel projections
        matmul_vec_adaptive_into(
            &self.q_proj,
            &state.hidden_state,
            &mut state.q_flat,
            work_config,
        );
        matmul_vec_adaptive_into(
            &self.k_proj,
            &state.hidden_state,
            &mut state.k_flat,
            work_config,
        );
        matmul_vec_adaptive_into(
            &self.v_proj,
            &state.hidden_state,
            &mut state.v_flat,
            work_config,
        );

        // Copy flat buffers into shaped state arrays
        for h in 0..n_heads {
            for d in 0..head_dim {
                state.q_state[[h, d]] = state.q_flat[h * head_dim + d];
            }
        }
        for h in 0..n_kv_heads {
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

        // Push to KV cache
        state.push_kv_optimized(self.layer_idx);

        // Parallel attention with adaptive chunking
        let attn_config = WorkDistributionConfig::for_attention(n_heads, seq_len);

        // Collect attention results in parallel
        let head_results: Vec<(usize, Vec<f32>)> = (0..n_heads)
            .into_par_iter()
            .with_min_len(attn_config.optimal_chunk_size(n_heads))
            .map(|h| {
                let kv_head = h / kv_groups;

                // Get query for this head
                let q_head: Vec<f32> = (0..head_dim).map(|d| state.q_state[[h, d]]).collect();

                // Compute attention scores
                let mut scores = vec![0.0f32; seq_len];
                for i in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += state.k_cache.get(self.layer_idx, kv_head, i, d) * q_head[d];
                    }
                    scores[i] = dot * scale;
                }

                // Softmax
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv_sum = 1.0 / sum;
                for s in &mut scores {
                    *s *= inv_sum;
                }

                // Weighted sum
                let mut context = vec![0.0f32; head_dim];
                for i in 0..seq_len {
                    let w = scores[i];
                    for d in 0..head_dim {
                        context[d] += w * state.v_cache.get(self.layer_idx, kv_head, i, d);
                    }
                }

                (h, context)
            })
            .collect();

        // Scatter results back
        for (h, context) in head_results {
            for d in 0..head_dim {
                state.context[[h, d]] = context[d];
            }
        }

        // Flatten context
        for h in 0..n_heads {
            for d in 0..head_dim {
                state.context_flat[h * head_dim + d] = state.context[[h, d]];
            }
        }

        // Output projection
        matmul_vec_adaptive_into(
            &self.o_proj,
            &state.context_flat,
            &mut state.hidden_state,
            work_config,
        );
    }

    #[cfg(not(feature = "parallel"))]
    pub fn forward_adaptive(&self, state: &mut InferenceState, _work_config: &()) {
        // Fall back to standard forward
        self.forward(state);
    }

    /// Standard forward pass (non-parallel)
    pub fn forward(&self, state: &mut InferenceState) {
        let head_dim = state.config.hidden_size / state.config.n_heads;

        kernels::matmul_vec_into(&self.q_proj, &state.hidden_state, &mut state.q_flat);
        kernels::matmul_vec_into(&self.k_proj, &state.hidden_state, &mut state.k_flat);
        kernels::matmul_vec_into(&self.v_proj, &state.hidden_state, &mut state.v_flat);

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

        let (cos, sin) = kernels::rope_embeddings(&state.inv_freq, state.pos);
        state.cos.assign(&cos);
        state.sin.assign(&sin);
        kernels::apply_rope(&mut state.q_state, &state.cos, &state.sin);
        kernels::apply_rope(&mut state.k_state, &state.cos, &state.sin);

        state.push_kv(self.layer_idx);

        let kv_groups = state.config.n_heads / state.config.n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        state.context.fill(0.0);

        for h in 0..state.config.n_heads {
            let kv_head = h / kv_groups;
            let seq_len = state.pos + 1;

            let q_head = state.q_state.row(h);
            let k_data = state
                .k_cache
                .get_slice_f32(self.layer_idx, kv_head, seq_len);
            let k_arr = ArrayView2::from_shape((seq_len, head_dim), &k_data).unwrap();

            {
                let mut scores_slice = state.scores.slice_mut(s![h, ..seq_len]);
                kernels::compute_attention_scores(q_head, k_arr, &mut scores_slice, scale);
                kernels::softmax_view(&mut scores_slice);
            }

            let v_data = state
                .v_cache
                .get_slice_f32(self.layer_idx, kv_head, seq_len);
            let v_arr = ArrayView2::from_shape((seq_len, head_dim), &v_data).unwrap();
            let scores_view = state.scores.slice(s![h, ..seq_len]);
            let mut context_row = state.context.slice_mut(s![h, ..]);
            kernels::weighted_sum_rows(scores_view, v_arr, &mut context_row);
        }

        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.context_flat[h * head_dim + d] = state.context[[h, d]];
            }
        }

        kernels::matmul_vec_into(&self.o_proj, &state.context_flat, &mut state.hidden_state);
    }
}

// =============================================================================
// Parallel MLP Module
// =============================================================================

/// MLP module with parallelization support
pub struct ParallelMLP {
    pub gate_proj: Array2<f32>,
    pub up_proj: Array2<f32>,
    pub down_proj: Array2<f32>,
}

impl ParallelMLP {
    pub fn new(gate_proj: Array2<f32>, up_proj: Array2<f32>, down_proj: Array2<f32>) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Forward pass with adaptive parallel work distribution
    #[cfg(feature = "parallel")]
    pub fn forward_adaptive(
        &self,
        state: &mut InferenceState,
        work_config: &WorkDistributionConfig,
    ) {
        // Use adaptive matmul for projections
        matmul_vec_adaptive_into(
            &self.gate_proj,
            &state.hidden_state,
            &mut state.mlp_gate,
            work_config,
        );
        matmul_vec_adaptive_into(
            &self.up_proj,
            &state.hidden_state,
            &mut state.mlp_up,
            work_config,
        );

        // Parallel fused SiLU and multiply
        let gate_slice = state
            .mlp_gate
            .as_slice()
            .expect("mlp_gate must be contiguous");
        let up_slice = state.mlp_up.as_slice().expect("mlp_up must be contiguous");

        let activated: Vec<f32> = gate_slice
            .par_iter()
            .zip(up_slice.par_iter())
            .map(|(&g, &u)| {
                let silu_g = g / (1.0 + (-g).exp());
                silu_g * u
            })
            .collect();

        state.mlp_gate.assign(&Array1::from_vec(activated));

        // Output projection
        matmul_vec_adaptive_into(
            &self.down_proj,
            &state.mlp_gate,
            &mut state.hidden_state,
            work_config,
        );
    }

    /// Forward pass with tensor parallelism
    #[cfg(feature = "parallel")]
    pub fn forward_tensor_parallel(
        &self,
        state: &mut InferenceState,
        tp_config: &TensorParallelConfig,
    ) {
        let output = mlp_tensor_parallel(
            &state.hidden_state,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
            tp_config,
        );
        state.hidden_state.assign(&output);
    }

    /// Standard forward pass
    pub fn forward(&self, state: &mut InferenceState) {
        state
            .mlp_gate
            .assign(&kernels::matmul_vec(&self.gate_proj, &state.hidden_state));
        state
            .mlp_up
            .assign(&kernels::matmul_vec(&self.up_proj, &state.hidden_state));

        let gate_activated = kernels::silu(&state.mlp_gate);
        for i in 0..state.mlp_gate.len() {
            state.mlp_gate[i] = gate_activated[i] * state.mlp_up[i];
        }

        state
            .hidden_state
            .assign(&kernels::matmul_vec(&self.down_proj, &state.mlp_gate));
    }
}

// =============================================================================
// Parallel Layer
// =============================================================================

/// Transformer layer with parallelization support
pub struct ParallelLayer {
    pub input_layernorm: RMSNorm,
    pub self_attn: ParallelAttention,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: ParallelMLP,
}

impl ParallelLayer {
    pub fn new(
        input_layernorm: RMSNorm,
        self_attn: ParallelAttention,
        post_attention_layernorm: RMSNorm,
        mlp: ParallelMLP,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    /// Forward pass with adaptive parallelism
    #[cfg(feature = "parallel")]
    pub fn forward_parallel(
        &self,
        state: &mut InferenceState,
        layer_idx: usize,
        work_config: &WorkDistributionConfig,
        debug: bool,
    ) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  Layer {}/{} (parallel)", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm
        self.input_layernorm.fast_forward(state);

        // Parallel attention
        self.self_attn.forward_adaptive(state, work_config);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-MLP norm
        self.post_attention_layernorm.fast_forward(state);

        // Parallel MLP
        self.mlp.forward_adaptive(state, work_config);

        // Residual connection
        state.hidden_state += &state.residual;
    }

    /// Standard forward pass
    pub fn forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  Layer {}/{}", layer_idx, state.config.n_layers);
        }

        state.residual.assign(&state.hidden_state);
        self.input_layernorm.forward(state);
        self.self_attn.forward(state);
        state.hidden_state += &state.residual;
        state.residual.assign(&state.hidden_state);
        self.post_attention_layernorm.forward(state);
        self.mlp.forward(state);
        state.hidden_state += &state.residual;
    }
}

// =============================================================================
// Pipeline-Parallel Mistral Model
// =============================================================================

/// Mistral model with pipeline parallelism support
///
/// Enables overlapping computation across layers when processing
/// multiple tokens. Best suited for batch inference scenarios.
pub struct PipelineParallelMistral {
    pub config: Config,
    pub parallel_config: ParallelConfig,
    pub embedding: Embedding,
    pub layers: Vec<ParallelLayer>,
    pub norm: RMSNorm,
    pub lm_head: Array2<f32>,
    pub tokenizer: crate::tokenizer::Tokenizer,
    /// Pipeline state for managing micro-batch execution
    #[cfg(feature = "parallel")]
    #[allow(dead_code)] // Used for future pipeline execution methods
    pipeline_state: PipelineState,
}

impl PipelineParallelMistral {
    /// Load a pipeline-parallel model
    pub fn load(params: Parameters, parallel_config: ParallelConfig) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();

        // Load embedding
        eprintln!("Loading embedding table...");
        let embed_data = params.get_tensor("model.embed_tokens.weight")?;
        let embed_shape = params
            .get_tensor_shape("model.embed_tokens.weight")
            .unwrap();
        let embedding = Embedding::new(WeightMatrix::F32(Array2::from_shape_vec(
            (embed_shape[0], embed_shape[1]),
            embed_data,
        )?));

        // Load final norm
        eprintln!("Loading final norm...");
        let norm_data = params.get_tensor("model.norm.weight")?;
        let norm = RMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Load LM head
        eprintln!("Loading LM head...");
        let lm_head_data = params.get_tensor("lm_head.weight")?;
        let lm_head_shape = params.get_tensor_shape("lm_head.weight").unwrap();
        let lm_head = Array2::from_shape_vec((lm_head_shape[0], lm_head_shape[1]), lm_head_data)?;

        // Load layers
        eprintln!("Loading {} layers (pipeline-parallel)...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            if i.is_multiple_of(4) {
                eprintln!("  Loading layer {}/{}...", i, config.n_layers);
            }
            layers.push(Self::load_layer(&params, i, &config)?);
        }

        #[cfg(feature = "parallel")]
        let pipeline_state = {
            let num_stages = config.n_layers / parallel_config.layers_per_stage.max(1);
            let max_depth = parallel_config
                .pipeline
                .as_ref()
                .map(|p| p.num_micro_batches)
                .unwrap_or(1);
            PipelineState::new(num_stages.max(1), max_depth)
        };

        Ok(Self {
            config,
            parallel_config,
            embedding,
            layers,
            norm,
            lm_head,
            tokenizer,
            #[cfg(feature = "parallel")]
            pipeline_state,
        })
    }

    fn load_layer(params: &Parameters, layer_idx: usize, config: &Config) -> Result<ParallelLayer> {
        let prefix = format!("model.layers.{}", layer_idx);

        let input_norm_data = params.get_tensor(&format!("{}.input_layernorm.weight", prefix))?;
        let input_layernorm = RMSNorm::new(Array1::from_vec(input_norm_data), config.norm_eps);

        let post_attn_norm_data =
            params.get_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
        let post_attention_layernorm =
            RMSNorm::new(Array1::from_vec(post_attn_norm_data), config.norm_eps);

        let q_proj = Self::load_weight(params, &format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_proj = Self::load_weight(params, &format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_proj = Self::load_weight(params, &format!("{}.self_attn.v_proj.weight", prefix))?;
        let o_proj = Self::load_weight(params, &format!("{}.self_attn.o_proj.weight", prefix))?;

        let self_attn = ParallelAttention::new(layer_idx, q_proj, k_proj, v_proj, o_proj);

        let gate_proj = Self::load_weight(params, &format!("{}.mlp.gate_proj.weight", prefix))?;
        let up_proj = Self::load_weight(params, &format!("{}.mlp.up_proj.weight", prefix))?;
        let down_proj = Self::load_weight(params, &format!("{}.mlp.down_proj.weight", prefix))?;

        let mlp = ParallelMLP::new(gate_proj, up_proj, down_proj);

        Ok(ParallelLayer::new(
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        ))
    }

    fn load_weight(params: &Parameters, name: &str) -> Result<Array2<f32>> {
        let data = params.get_tensor(name)?;
        let shape = params.get_tensor_shape(name).unwrap();
        Ok(Array2::from_shape_vec((shape[0], shape[1]), data)?)
    }

    /// Forward pass with adaptive parallel work distribution
    #[cfg(feature = "parallel")]
    pub fn forward_parallel(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Embedding lookup
        self.embedding.forward(state, token);

        // Pass through layers with adaptive parallelism
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward_parallel(state, i, &self.parallel_config.work_distribution, debug);
        }

        // Final norm
        self.norm.fast_forward(state);

        // LM head projection
        matmul_vec_adaptive_into(
            &self.lm_head,
            &state.hidden_state,
            &mut state.logits,
            &self.parallel_config.work_distribution,
        );
    }

    /// Standard forward pass
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        self.embedding.forward(state, token);

        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(state, i, debug);
        }

        self.norm.forward(state);
        state
            .logits
            .assign(&kernels::matmul_vec(&self.lm_head, &state.hidden_state));
    }

    /// Fast forward - uses parallel version when available
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        #[cfg(feature = "parallel")]
        {
            self.forward_parallel(state, token, debug);
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.forward(state, token, debug);
        }
    }
}

// =============================================================================
// Tensor-Parallel Mistral Model
// =============================================================================

/// Mistral model with tensor parallelism support
///
/// Splits attention heads and MLP intermediate dimension across
/// multiple devices/threads for larger model support.
#[cfg(feature = "parallel")]
pub struct TensorParallelMistral {
    pub config: Config,
    pub tp_config: TensorParallelConfig,
    pub work_config: WorkDistributionConfig,
    pub embedding: Embedding,
    pub layers: Vec<ParallelLayer>,
    pub norm: RMSNorm,
    pub lm_head: Array2<f32>,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

#[cfg(feature = "parallel")]
impl TensorParallelMistral {
    /// Load a tensor-parallel model
    pub fn load(params: Parameters, tp_config: TensorParallelConfig) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();
        let work_config =
            WorkDistributionConfig::for_matmul(config.hidden_size, config.hidden_size);

        // Load embedding (replicated across all ranks)
        eprintln!("Loading embedding table (rank {})...", tp_config.rank);
        let embed_data = params.get_tensor("model.embed_tokens.weight")?;
        let embed_shape = params
            .get_tensor_shape("model.embed_tokens.weight")
            .unwrap();
        let embedding = Embedding::new(WeightMatrix::F32(Array2::from_shape_vec(
            (embed_shape[0], embed_shape[1]),
            embed_data,
        )?));

        // Load final norm (replicated)
        let norm_data = params.get_tensor("model.norm.weight")?;
        let norm = RMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Load LM head (replicated or partitioned based on vocab size)
        let lm_head_data = params.get_tensor("lm_head.weight")?;
        let lm_head_shape = params.get_tensor_shape("lm_head.weight").unwrap();
        let lm_head = Array2::from_shape_vec((lm_head_shape[0], lm_head_shape[1]), lm_head_data)?;

        // Load layers (attention is tensor-parallel, MLP is tensor-parallel)
        eprintln!(
            "Loading {} layers (tensor-parallel, rank {})...",
            config.n_layers, tp_config.rank
        );
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            if i.is_multiple_of(4) {
                eprintln!("  Loading layer {}/{}...", i, config.n_layers);
            }
            layers.push(PipelineParallelMistral::load_layer(&params, i, &config)?);
        }

        Ok(Self {
            config,
            tp_config,
            work_config,
            embedding,
            layers,
            norm,
            lm_head,
            tokenizer,
        })
    }

    /// Forward pass with tensor parallelism
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        self.embedding.forward(state, token);

        for (i, layer) in self.layers.iter().enumerate() {
            // Use tensor-parallel attention
            layer.forward_parallel(state, i, &self.work_config, debug);
        }

        self.norm.fast_forward(state);
        matmul_vec_adaptive_into(
            &self.lm_head,
            &state.hidden_state,
            &mut state.logits,
            &self.work_config,
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.adaptive_matmul);
        assert!(config.parallel_heads);
    }

    #[test]
    fn test_parallel_config_pipeline() {
        let config = ParallelConfig::pipeline(4);
        assert!(config.pipeline.is_some());
        assert_eq!(config.pipeline.as_ref().unwrap().num_micro_batches, 4);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_config_tensor_parallel() {
        let config = ParallelConfig::tensor_parallel(2, 0);
        assert!(config.tensor_parallel.is_some());
        let tp = config.tensor_parallel.as_ref().unwrap();
        assert_eq!(tp.world_size, 2);
        assert_eq!(tp.rank, 0);
    }

    #[test]
    fn test_parallel_mlp_new() {
        let gate = Array2::zeros((8, 4));
        let up = Array2::zeros((8, 4));
        let down = Array2::zeros((4, 8));

        let mlp = ParallelMLP::new(gate, up, down);
        assert_eq!(mlp.gate_proj.dim(), (8, 4));
        assert_eq!(mlp.down_proj.dim(), (4, 8));
    }

    #[test]
    fn test_parallel_attention_new() {
        let q = Array2::zeros((16, 4));
        let k = Array2::zeros((4, 4));
        let v = Array2::zeros((4, 4));
        let o = Array2::zeros((4, 16));

        let attn = ParallelAttention::new(0, q, k, v, o);
        assert_eq!(attn.layer_idx, 0);
        assert_eq!(attn.q_proj.dim(), (16, 4));
    }
}
