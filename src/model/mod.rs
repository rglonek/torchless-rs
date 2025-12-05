use crate::kernels;
use crate::loader::{Config, Parameters};
use anyhow::Result;
use ndarray::{Array1, Array2, Array4};

pub mod modules;
pub use modules::{Attention, Embedding, Layer, RMSNorm, MLP};

const MAX_SEQ_LEN: usize = 500;

pub struct InferenceState {
    pub config: Config,
    pub pos: usize,

    // Hidden state and residual
    pub hidden_state: Array1<f32>,
    pub residual: Array1<f32>,

    // RoPE embeddings
    pub inv_freq: Array1<f32>,
    pub cos: Array1<f32>,
    pub sin: Array1<f32>,

    // Attention states
    pub q_state: Array2<f32>, // [n_heads, head_dim]
    pub k_state: Array2<f32>, // [n_kv_heads, head_dim]
    pub v_state: Array2<f32>, // [n_kv_heads, head_dim]

    // KV cache
    pub k_cache: Array4<f32>, // [n_layers, n_kv_heads, max_seq_len, head_dim]
    pub v_cache: Array4<f32>, // [n_layers, n_kv_heads, max_seq_len, head_dim]

    // Attention outputs
    pub scores: Array2<f32>,  // [n_heads, max_seq_len]
    pub context: Array2<f32>, // [n_heads, head_dim]

    // MLP intermediate states
    pub mlp_gate: Array1<f32>, // [intermediate_size]
    pub mlp_up: Array1<f32>,   // [intermediate_size]

    // Output logits and probs
    pub logits: Array1<f32>, // [vocab_size]
    pub probs: Array1<f32>,  // [vocab_size]
}

impl InferenceState {
    pub fn new(config: Config) -> Self {
        let head_dim = config.hidden_size / config.n_heads;

        // Initialize RoPE inverse frequencies
        let inv_freq = kernels::init_rope_freqs(head_dim, config.rope_theta);

        Self {
            pos: 0,
            hidden_state: Array1::zeros(config.hidden_size),
            residual: Array1::zeros(config.hidden_size),

            inv_freq,
            cos: Array1::zeros(head_dim / 2),
            sin: Array1::zeros(head_dim / 2),

            q_state: Array2::zeros((config.n_heads, head_dim)),
            k_state: Array2::zeros((config.n_kv_heads, head_dim)),
            v_state: Array2::zeros((config.n_kv_heads, head_dim)),

            k_cache: Array4::zeros((config.n_layers, config.n_kv_heads, MAX_SEQ_LEN, head_dim)),
            v_cache: Array4::zeros((config.n_layers, config.n_kv_heads, MAX_SEQ_LEN, head_dim)),

            scores: Array2::zeros((config.n_heads, MAX_SEQ_LEN)),
            context: Array2::zeros((config.n_heads, head_dim)),

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
            if i % 4 == 0 {
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
}
