use super::super::InferenceState;
use super::{Attention, RMSNorm, MLP};

/// Transformer decoder layer
pub struct Layer {
    pub input_layernorm: RMSNorm,
    pub self_attn: Attention,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: MLP,
}

impl Layer {
    pub fn new(
        input_layernorm: RMSNorm,
        self_attn: Attention,
        post_attention_layernorm: RMSNorm,
        mlp: MLP,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    /// Forward pass: norm -> attention -> residual -> norm -> mlp -> residual
    pub fn forward(&self, state: &mut InferenceState, layer_idx: usize, debug: bool) {
        if debug && layer_idx.is_multiple_of(8) {
            eprintln!("  Layer {}/{}", layer_idx, state.config.n_layers);
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

        // Pre-MLP norm
        self.post_attention_layernorm.forward(state);

        // MLP
        self.mlp.forward(state);

        // Residual connection
        state.hidden_state += &state.residual;
    }
}
