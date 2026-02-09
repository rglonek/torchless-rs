use super::super::InferenceState;
use super::{LazyAttention, LazyMLP, RMSNorm};
use crate::loader::Parameters;

/// Lazy transformer decoder layer.
/// Norm weights are stored eagerly (they're small), but attention and MLP
/// projections are accessed lazily from the memory-mapped file.
pub struct LazyLayer {
    pub input_layernorm: RMSNorm,
    pub self_attn: LazyAttention,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: LazyMLP,
}

impl LazyLayer {
    pub fn new(
        input_layernorm: RMSNorm,
        self_attn: LazyAttention,
        post_attention_layernorm: RMSNorm,
        mlp: LazyMLP,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    /// Forward pass: norm -> attention -> residual -> norm -> mlp -> residual
    /// Uses lazy tensor loading for attention and MLP projections.
    pub fn forward(
        &self,
        state: &mut InferenceState,
        layer_idx: usize,
        debug: bool,
        params: &Parameters,
    ) {
        if debug && layer_idx % 8 == 0 {
            eprintln!("  Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm
        self.input_layernorm.forward(state);

        // Self-attention (lazy tensor access)
        self.self_attn.forward(state, params);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual again
        state.residual.assign(&state.hidden_state);

        // Pre-MLP norm
        self.post_attention_layernorm.forward(state);

        // MLP (lazy tensor access)
        self.mlp.forward(state, params);

        // Residual connection
        state.hidden_state += &state.residual;
    }

    /// Optimized forward pass: uses SIMD and parallel kernels where available.
    pub fn fast_forward(
        &self,
        state: &mut InferenceState,
        layer_idx: usize,
        debug: bool,
        params: &Parameters,
    ) {
        if debug && layer_idx % 8 == 0 {
            eprintln!("  Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm (uses SIMD when available)
        self.input_layernorm.fast_forward(state);

        // Self-attention (lazy + optimized)
        self.self_attn.fast_forward(state, params);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual again
        state.residual.assign(&state.hidden_state);

        // Pre-MLP norm (uses SIMD when available)
        self.post_attention_layernorm.fast_forward(state);

        // MLP (lazy + optimized)
        self.mlp.fast_forward(state, params);

        // Residual connection
        state.hidden_state += &state.residual;
    }
}
