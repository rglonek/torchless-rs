use super::super::InferenceState;
use crate::kernels;
use ndarray::Array2;

/// MLP (feedforward) layer with SwiGLU activation
pub struct MLP {
    pub gate_proj: Array2<f32>, // [intermediate_size, hidden_size]
    pub up_proj: Array2<f32>,   // [intermediate_size, hidden_size]
    pub down_proj: Array2<f32>, // [hidden_size, intermediate_size]
}

impl MLP {
    pub fn new(gate_proj: Array2<f32>, up_proj: Array2<f32>, down_proj: Array2<f32>) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Forward pass: SwiGLU feedforward
    /// out = down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
    pub fn forward(&self, state: &mut InferenceState) {
        // gate = gate_proj @ hidden_state
        state
            .mlp_gate
            .assign(&kernels::matmul_vec(&self.gate_proj, &state.hidden_state));

        // up = up_proj @ hidden_state
        state
            .mlp_up
            .assign(&kernels::matmul_vec(&self.up_proj, &state.hidden_state));

        // Apply SiLU to gate
        let gate_activated = kernels::silu(&state.mlp_gate);

        // Element-wise multiply: gate * up
        for i in 0..state.mlp_gate.len() {
            state.mlp_gate[i] = gate_activated[i] * state.mlp_up[i];
        }

        // down_proj @ (gate * up)
        state
            .hidden_state
            .assign(&kernels::matmul_vec(&self.down_proj, &state.mlp_gate));
    }

    /// Optimized forward pass: uses parallel matmul when available
    pub fn fast_forward(&self, state: &mut InferenceState) {
        // gate = gate_proj @ hidden_state (parallel when feature enabled)
        state
            .mlp_gate
            .assign(&kernels::fast_matmul_vec(&self.gate_proj, &state.hidden_state));

        // up = up_proj @ hidden_state (parallel when feature enabled)
        state
            .mlp_up
            .assign(&kernels::fast_matmul_vec(&self.up_proj, &state.hidden_state));

        // Apply SiLU to gate (uses SIMD when that feature is enabled)
        let gate_activated = kernels::fast_silu(&state.mlp_gate);

        // Element-wise multiply: gate * up
        for i in 0..state.mlp_gate.len() {
            state.mlp_gate[i] = gate_activated[i] * state.mlp_up[i];
        }

        // down_proj @ (gate * up) (parallel when feature enabled)
        state
            .hidden_state
            .assign(&kernels::fast_matmul_vec(&self.down_proj, &state.mlp_gate));
    }
}
