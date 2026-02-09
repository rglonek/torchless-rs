use super::super::InferenceState;
use super::lazy_moe::LazyMoE;
use super::{LazyAttention, RMSNorm};
use crate::loader::Parameters;

/// Lazy transformer decoder layer with Mixture-of-Experts FFN.
/// Norm weights are stored eagerly (small), but attention projections and
/// all expert weights are accessed lazily from the memory-mapped file.
pub struct LazyMoELayer {
    pub input_layernorm: RMSNorm,
    pub self_attn: LazyAttention,
    pub post_attention_layernorm: RMSNorm,
    pub moe: LazyMoE,
}

impl LazyMoELayer {
    pub fn new(
        input_layernorm: RMSNorm,
        self_attn: LazyAttention,
        post_attention_layernorm: RMSNorm,
        moe: LazyMoE,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            moe,
        }
    }

    /// Forward pass: norm -> attention -> residual -> norm -> MoE -> residual
    /// Uses lazy tensor loading for attention and MoE expert projections.
    pub fn forward(
        &self,
        state: &mut InferenceState,
        layer_idx: usize,
        debug: bool,
        params: &Parameters,
    ) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  MoE Layer {}/{}", layer_idx, state.config.n_layers);
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

        // Pre-MoE norm
        self.post_attention_layernorm.forward(state);

        // MoE FFN (lazy tensor access)
        self.moe.forward(state, params);

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
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  MoE Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm (SIMD when available)
        self.input_layernorm.fast_forward(state);

        // Self-attention (lazy + optimized)
        self.self_attn.fast_forward(state, params);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual again
        state.residual.assign(&state.hidden_state);

        // Pre-MoE norm (SIMD when available)
        self.post_attention_layernorm.fast_forward(state);

        // MoE FFN (lazy + optimized)
        self.moe.fast_forward(state, params);

        // Residual connection
        state.hidden_state += &state.residual;
    }
}
