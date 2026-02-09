use super::super::InferenceState;
use super::moe::MoE;
use super::{Attention, RMSNorm};

/// Transformer decoder layer with Mixture-of-Experts FFN.
/// Identical to Layer but uses MoE instead of a dense MLP.
pub struct MoELayer {
    pub input_layernorm: RMSNorm,
    pub self_attn: Attention,
    pub post_attention_layernorm: RMSNorm,
    pub moe: MoE,
}

impl MoELayer {
    pub fn new(
        input_layernorm: RMSNorm,
        self_attn: Attention,
        post_attention_layernorm: RMSNorm,
        moe: MoE,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            moe,
        }
    }

    /// Forward pass: norm -> attention -> residual -> norm -> MoE -> residual
    pub fn forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  MoE Layer {}/{}", layer_idx, state.config.n_layers);
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

        // Pre-MoE norm
        self.post_attention_layernorm.forward(state);

        // MoE FFN
        self.moe.forward(state);

        // Residual connection
        state.hidden_state += &state.residual;
    }

    /// Optimized forward pass: uses parallel attention and MoE when available
    pub fn fast_forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  MoE Layer {}/{}", layer_idx, state.config.n_layers);
        }

        // Save residual
        state.residual.assign(&state.hidden_state);

        // Pre-attention norm (SIMD when available)
        self.input_layernorm.fast_forward(state);

        // Self-attention (parallel when available)
        self.self_attn.fast_forward(state);

        // Residual connection
        state.hidden_state += &state.residual;

        // Save residual again
        state.residual.assign(&state.hidden_state);

        // Pre-MoE norm (SIMD when available)
        self.post_attention_layernorm.fast_forward(state);

        // MoE FFN (optimized)
        self.moe.fast_forward(state);

        // Residual connection
        state.hidden_state += &state.residual;
    }
}
