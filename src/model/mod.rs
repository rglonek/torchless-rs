use crate::kernels;
use crate::loader::{Config, Parameters};
use anyhow::Result;
use ndarray::{Array1, Array2, Array4};

pub mod architecture;
pub mod batching;
pub mod models;
pub mod modules;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod speculative;

pub use modules::{
    Attention, Embedding, Layer, LazyAttention, LazyEmbedding, LazyLayer, LazyMLP, RMSNorm, MLP,
};

// =============================================================================
// Phase 8: Multi-Architecture Support
// =============================================================================

// Architecture detection and configuration
pub use architecture::{
    detect_architecture, detect_architecture_from_config, detect_architecture_from_tensors,
    ActivationType, ArchitectureConfig, Model, ModelArchitecture, NormType, RopeScaling,
    TensorNamePattern,
};

// Model implementations for different architectures
pub use models::{
    // Dynamic model enum for runtime polymorphism
    DynamicModel,
    // Gemma (Google)
    Gemma,
    // LLaMA (Meta)
    LLaMA,
    LazyGemma,
    LazyLLaMA,
    LazyPhi,
    LazyQwen,
    ModelLoader,
    // Phi (Microsoft)
    Phi,
    // Qwen (Alibaba)
    Qwen,
};

// Flash Attention exports (Phase 4 Algorithmic Optimization)
pub use modules::{
    flash_attention_estimate_memory, flash_attention_into, flash_attention_multi_head,
    flash_attention_single_head, FlashAttentionConfig, FLASH_ATTENTION_THRESHOLD, FLASH_TILE_SIZE,
};

#[cfg(feature = "parallel")]
pub use modules::flash_attention_parallel;

// Speculative Decoding exports (Phase 4 Algorithmic Optimization)
pub use speculative::{
    CacheState, LookaheadDecoder, SelfSpeculativeDecoder, SpeculativeConfig, SpeculativeDecoder,
    SpeculativeModel, SpeculativeStats, TokenBuffer,
};

// Continuous Batching exports (Phase 4 Algorithmic Optimization)
pub use batching::{
    BatchScheduler, BatchStepResult, BatchedInferenceState, BatchingConfig, BatchingStats,
    ContinuousBatchingEngine, KVCachePool, Sequence, SequenceId, SequenceStatus,
};

// Phase 6: Parallelization exports
#[cfg(feature = "parallel")]
pub use parallel::{
    ParallelAttention, ParallelConfig, ParallelInferenceState, ParallelLayer, ParallelMLP,
    PipelineParallelMistral, TensorParallelMistral,
};

/// Default maximum sequence length if not specified.
const DEFAULT_MAX_SEQ_LEN: usize = 2048;

pub struct InferenceState {
    pub config: Config,
    pub pos: usize,
    pub max_seq_len: usize,

    // Hidden state and residual
    pub hidden_state: Array1<f32>,
    pub residual: Array1<f32>,

    // RoPE embeddings
    pub inv_freq: Array1<f32>,
    pub cos: Array1<f32>,
    pub sin: Array1<f32>,

    // Attention projection buffers (flat, for matmul output)
    pub q_flat: Array1<f32>, // [n_heads * head_dim]
    pub k_flat: Array1<f32>, // [n_kv_heads * head_dim]
    pub v_flat: Array1<f32>, // [n_kv_heads * head_dim]

    // Attention states (reshaped views into flat buffers)
    pub q_state: Array2<f32>, // [n_heads, head_dim]
    pub k_state: Array2<f32>, // [n_kv_heads, head_dim]
    pub v_state: Array2<f32>, // [n_kv_heads, head_dim]

    // KV cache
    pub k_cache: Array4<f32>, // [n_layers, n_kv_heads, max_seq_len, head_dim]
    pub v_cache: Array4<f32>, // [n_layers, n_kv_heads, max_seq_len, head_dim]

    // Attention outputs
    pub scores: Array2<f32>,  // [n_heads, max_seq_len]
    pub context: Array2<f32>, // [n_heads, head_dim]

    // Pre-allocated buffer for flattened context (avoids clone in output projection)
    pub context_flat: Array1<f32>, // [hidden_size]

    // MLP intermediate states
    pub mlp_gate: Array1<f32>, // [intermediate_size]
    pub mlp_up: Array1<f32>,   // [intermediate_size]

    // Output logits and probs
    pub logits: Array1<f32>, // [vocab_size]
    pub probs: Array1<f32>,  // [vocab_size]
}

impl InferenceState {
    /// Create a new inference state with the default max sequence length.
    pub fn new(config: Config) -> Self {
        Self::with_seq_len(config, DEFAULT_MAX_SEQ_LEN)
    }

    /// Create a new inference state with a custom max sequence length.
    pub fn with_seq_len(config: Config, max_seq_len: usize) -> Self {
        let head_dim = config.hidden_size / config.n_heads;

        // Initialize RoPE inverse frequencies
        let inv_freq = kernels::init_rope_freqs(head_dim, config.rope_theta);

        Self {
            pos: 0,
            max_seq_len,
            hidden_state: Array1::zeros(config.hidden_size),
            residual: Array1::zeros(config.hidden_size),

            inv_freq,
            cos: Array1::zeros(head_dim / 2),
            sin: Array1::zeros(head_dim / 2),

            // Pre-allocated flat buffers for projections (avoids per-forward allocation)
            q_flat: Array1::zeros(config.n_heads * head_dim),
            k_flat: Array1::zeros(config.n_kv_heads * head_dim),
            v_flat: Array1::zeros(config.n_kv_heads * head_dim),

            q_state: Array2::zeros((config.n_heads, head_dim)),
            k_state: Array2::zeros((config.n_kv_heads, head_dim)),
            v_state: Array2::zeros((config.n_kv_heads, head_dim)),

            k_cache: Array4::zeros((config.n_layers, config.n_kv_heads, max_seq_len, head_dim)),
            v_cache: Array4::zeros((config.n_layers, config.n_kv_heads, max_seq_len, head_dim)),

            scores: Array2::zeros((config.n_heads, max_seq_len)),
            context: Array2::zeros((config.n_heads, head_dim)),

            // Pre-allocated buffer for flattened context (avoids clone in output projection)
            context_flat: Array1::zeros(config.hidden_size),

            mlp_gate: Array1::zeros(config.intermediate_size),
            mlp_up: Array1::zeros(config.intermediate_size),

            logits: Array1::zeros(config.vocab_size),
            probs: Array1::zeros(config.vocab_size),

            config,
        }
    }

    /// Push current K and V states to the cache at position `pos` for layer `layer_idx`
    pub fn push_kv(&mut self, layer_idx: usize) {
        for h in 0..self.config.n_kv_heads {
            // Copy k_state[h] to k_cache[layer_idx, h, pos, :]
            for d in 0..self.k_state.shape()[1] {
                self.k_cache[[layer_idx, h, self.pos, d]] = self.k_state[[h, d]];
                self.v_cache[[layer_idx, h, self.pos, d]] = self.v_state[[h, d]];
            }
        }
    }

    /// Push current K and V states to cache using optimized unchecked access.
    ///
    /// # Safety
    /// This uses unchecked array access for performance. The caller must ensure
    /// that layer_idx, pos, and all head indices are within bounds.
    pub fn push_kv_optimized(&mut self, layer_idx: usize) {
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.k_state.shape()[1];
        let pos = self.pos;

        // Get raw pointers for unchecked access
        let k_state_slice = self.k_state.as_slice().expect("k_state must be contiguous");
        let v_state_slice = self.v_state.as_slice().expect("v_state must be contiguous");

        // Calculate strides for 4D cache access
        // Shape: [n_layers, n_kv_heads, max_seq_len, head_dim]
        let cache_shape = self.k_cache.shape();
        let stride_layer = cache_shape[1] * cache_shape[2] * cache_shape[3];
        let stride_head = cache_shape[2] * cache_shape[3];
        let stride_pos = cache_shape[3];

        let k_cache_slice = self
            .k_cache
            .as_slice_mut()
            .expect("k_cache must be contiguous");
        let v_cache_slice = self
            .v_cache
            .as_slice_mut()
            .expect("v_cache must be contiguous");

        unsafe {
            let base_offset = layer_idx * stride_layer + pos * stride_pos;

            for h in 0..n_kv_heads {
                let cache_offset = base_offset + h * stride_head;
                let state_offset = h * head_dim;

                // Use pointer arithmetic for bounds-check-free copy
                let k_dst = k_cache_slice.as_mut_ptr().add(cache_offset);
                let v_dst = v_cache_slice.as_mut_ptr().add(cache_offset);
                let k_src = k_state_slice.as_ptr().add(state_offset);
                let v_src = v_state_slice.as_ptr().add(state_offset);

                // Copy head_dim elements
                std::ptr::copy_nonoverlapping(k_src, k_dst, head_dim);
                std::ptr::copy_nonoverlapping(v_src, v_dst, head_dim);
            }
        }
    }

    /// Reset the inference state for a new sequence.
    pub fn reset(&mut self) {
        self.pos = 0;
        // Hidden state and residual will be overwritten
    }
}

// =============================================================================
// Arena-based Inference State (Phase 3 Memory Optimization)
// =============================================================================

/// Arena-backed inference state for reduced allocation overhead.
///
/// This variant uses the bumpalo arena allocator for temporary buffers,
/// providing:
/// - O(1) allocation (just bump a pointer)
/// - Zero per-token deallocation overhead
/// - Better cache locality (contiguous allocations)
/// - Arena reset between tokens for memory reuse
///
/// # Performance
/// Reduces allocation overhead in forward passes by 10-50x compared to
/// standard InferenceState when processing many tokens.
///
/// # Usage
/// ```ignore
/// let mut state = ArenaInferenceState::new(config);
/// for token in tokens {
///     model.forward(&mut state.as_inference_state(), token, false);
///     // Arena temporaries are automatically reset
/// }
/// ```
pub struct ArenaInferenceState {
    /// The underlying inference state
    pub inner: InferenceState,
    /// Arena for temporary allocations during forward pass
    pub arena: crate::memory::InferenceArena,
    /// Aligned scratch buffer for MLP intermediate (reused across layers)
    pub mlp_scratch: crate::memory::AlignedBuffer<f32>,
    /// Aligned scratch buffer for attention scores
    pub attention_scratch: crate::memory::AlignedBuffer<f32>,
}

impl ArenaInferenceState {
    /// Create a new arena-backed inference state with the default max sequence length.
    pub fn new(config: Config) -> Self {
        Self::with_seq_len(config, DEFAULT_MAX_SEQ_LEN)
    }

    /// Create a new arena-backed inference state with a custom max sequence length.
    pub fn with_seq_len(config: Config, max_seq_len: usize) -> Self {
        // Create arena sized for typical forward pass
        let arena = crate::memory::InferenceArena::for_inference(
            config.hidden_size,
            config.intermediate_size,
            config.n_heads,
            max_seq_len,
        );

        // Pre-allocate aligned scratch buffers
        let mlp_scratch = crate::memory::AlignedBuffer::zeros(config.intermediate_size);
        let attention_scratch = crate::memory::AlignedBuffer::zeros(config.n_heads * max_seq_len);

        Self {
            inner: InferenceState::with_seq_len(config, max_seq_len),
            arena,
            mlp_scratch,
            attention_scratch,
        }
    }

    /// Get a reference to the underlying InferenceState for compatibility.
    #[inline]
    pub fn as_inference_state(&mut self) -> &mut InferenceState {
        &mut self.inner
    }

    /// Reset the arena after a forward pass to reuse memory.
    #[inline]
    pub fn reset_arena(&mut self) {
        self.arena.reset();
    }

    /// Get a temporary aligned slice from the arena.
    ///
    /// This is very fast (O(1)) and the memory is automatically reclaimed
    /// when `reset_arena()` is called.
    #[inline]
    pub fn temp_slice(&mut self, len: usize) -> &mut [f32] {
        self.arena.alloc_aligned::<f32>(len)
    }

    /// Get the MLP scratch buffer.
    #[inline]
    pub fn mlp_buffer(&mut self) -> &mut [f32] {
        self.mlp_scratch.as_mut_slice()
    }

    /// Get a slice of the attention scratch buffer.
    #[inline]
    pub fn attention_buffer(&mut self, len: usize) -> &mut [f32] {
        &mut self.attention_scratch.as_mut_slice()[..len]
    }

    /// Get total arena bytes allocated.
    pub fn arena_bytes(&self) -> usize {
        self.arena.allocated_bytes()
    }
}

pub struct Mistral {
    pub config: Config,
    pub embedding: Embedding,
    pub layers: Vec<Layer>,
    pub norm: RMSNorm,
    pub lm_head: Array2<f32>,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl Mistral {
    pub fn load(params: Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();

        // Load embedding table
        eprintln!("Loading embedding table...");
        let embed_data = params.get_tensor("model.embed_tokens.weight")?;
        let embed_shape = params
            .get_tensor_shape("model.embed_tokens.weight")
            .unwrap();
        let embedding = Embedding::new(Array2::from_shape_vec(
            (embed_shape[0], embed_shape[1]),
            embed_data,
        )?);

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
        eprintln!("Loading {} layers...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            if i.is_multiple_of(4) {
                eprintln!("  Loading layer {}/{}...", i, config.n_layers);
            }
            layers.push(Self::load_layer(&params, i, &config)?);
        }

        Ok(Self {
            config,
            embedding,
            layers,
            norm,
            lm_head,
            tokenizer,
        })
    }

    fn load_layer(params: &Parameters, layer_idx: usize, config: &Config) -> Result<Layer> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Load norms
        let input_norm_data = params.get_tensor(&format!("{}.input_layernorm.weight", prefix))?;
        let input_layernorm = RMSNorm::new(Array1::from_vec(input_norm_data), config.norm_eps);

        let post_attn_norm_data =
            params.get_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
        let post_attention_layernorm =
            RMSNorm::new(Array1::from_vec(post_attn_norm_data), config.norm_eps);

        // Load attention projections
        let q_proj = Self::load_weight(params, &format!("{}.self_attn.q_proj.weight", prefix))?;
        let k_proj = Self::load_weight(params, &format!("{}.self_attn.k_proj.weight", prefix))?;
        let v_proj = Self::load_weight(params, &format!("{}.self_attn.v_proj.weight", prefix))?;
        let o_proj = Self::load_weight(params, &format!("{}.self_attn.o_proj.weight", prefix))?;

        let self_attn = Attention::new(layer_idx, q_proj, k_proj, v_proj, o_proj);

        // Load MLP projections
        let gate_proj = Self::load_weight(params, &format!("{}.mlp.gate_proj.weight", prefix))?;
        let up_proj = Self::load_weight(params, &format!("{}.mlp.up_proj.weight", prefix))?;
        let down_proj = Self::load_weight(params, &format!("{}.mlp.down_proj.weight", prefix))?;

        let mlp = MLP::new(gate_proj, up_proj, down_proj);

        Ok(Layer::new(
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

    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Embedding lookup
        self.embedding.forward(state, token);

        // Pass through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(state, i, debug);
        }

        // Final norm
        self.norm.forward(state);

        // LM head projection to get logits
        state
            .logits
            .assign(&kernels::matmul_vec(&self.lm_head, &state.hidden_state));
    }

    /// Optimized forward pass: uses parallel attention and matmul when available
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Embedding lookup
        self.embedding.forward(state, token);

        // Pass through all layers with optimized attention
        for (i, layer) in self.layers.iter().enumerate() {
            layer.fast_forward(state, i, debug);
        }

        // Final norm (uses SIMD when that feature is enabled)
        self.norm.fast_forward(state);

        // LM head projection to get logits (parallel when feature enabled)
        state.logits.assign(&kernels::fast_matmul_vec(
            &self.lm_head,
            &state.hidden_state,
        ));
    }
}

// =============================================================================
// Lazy Mistral - Memory-efficient model with lazy tensor loading
// =============================================================================

/// Lazy Mistral model that keeps tensors memory-mapped and accesses them on-demand.
/// This dramatically reduces memory usage compared to eager loading.
///
/// Memory savings:
/// - Eager Mistral: ~25GB RAM (all tensors in memory as f32)
/// - LazyMistral: <2GB RAM (only small buffers + mmap overhead)
pub struct LazyMistral<'a> {
    pub config: Config,
    pub params: &'a Parameters,
    pub layers: Vec<LazyLayer>,
    pub norm: RMSNorm,
    pub embedding: LazyEmbedding,
    pub tokenizer: crate::tokenizer::Tokenizer,
}

impl<'a> LazyMistral<'a> {
    /// Load a lazy model from Parameters.
    /// Only small tensors (norms) are loaded eagerly; large projection matrices
    /// remain memory-mapped and are accessed on-demand during forward pass.
    pub fn load(params: &'a Parameters) -> Result<Self> {
        let config = params.config.clone();
        let tokenizer = params.tokenizer.clone();

        // Only load the norm weights eagerly (these are small: hidden_size floats each)
        eprintln!("Loading norms (lazy model)...");
        let norm_data = params.get_tensor("model.norm.weight")?;
        let norm = RMSNorm::new(Array1::from_vec(norm_data), config.norm_eps);

        // Create lazy embedding (stores tensor name, not data)
        let embedding = LazyEmbedding::new("model.embed_tokens.weight".to_string());

        // Create lazy layers
        eprintln!("Initializing {} lazy layers...", config.n_layers);
        let mut layers = Vec::new();
        for i in 0..config.n_layers {
            layers.push(Self::create_lazy_layer(params, i, &config)?);
        }

        Ok(Self {
            config,
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
    ) -> Result<LazyLayer> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Load norms eagerly (small tensors)
        let input_norm_data = params.get_tensor(&format!("{}.input_layernorm.weight", prefix))?;
        let input_layernorm = RMSNorm::new(Array1::from_vec(input_norm_data), config.norm_eps);

        let post_attn_norm_data =
            params.get_tensor(&format!("{}.post_attention_layernorm.weight", prefix))?;
        let post_attention_layernorm =
            RMSNorm::new(Array1::from_vec(post_attn_norm_data), config.norm_eps);

        // Create lazy attention (stores tensor names, not data)
        let self_attn = LazyAttention::new(
            layer_idx,
            format!("{}.self_attn.q_proj.weight", prefix),
            format!("{}.self_attn.k_proj.weight", prefix),
            format!("{}.self_attn.v_proj.weight", prefix),
            format!("{}.self_attn.o_proj.weight", prefix),
        );

        // Create lazy MLP (stores tensor names, not data)
        let mlp = LazyMLP::new(
            format!("{}.mlp.gate_proj.weight", prefix),
            format!("{}.mlp.up_proj.weight", prefix),
            format!("{}.mlp.down_proj.weight", prefix),
        );

        Ok(LazyLayer::new(
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        ))
    }

    /// Forward pass with lazy tensor loading.
    /// Tensors are accessed from the memory map on-demand during computation.
    pub fn forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Lazy embedding lookup
        self.embedding.forward(state, token, self.params);

        // Pass through all layers with lazy tensor access
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

    /// Optimized forward pass: uses SIMD and parallel kernels when available.
    pub fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool) {
        // Lazy embedding lookup
        self.embedding.forward(state, token, self.params);

        // Pass through all layers with lazy tensor access + optimizations
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
