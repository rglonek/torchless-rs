use super::super::InferenceState;
use crate::kernels;
use crate::loader::WeightMatrix;

/// MLP (feedforward) layer with SwiGLU activation
pub struct MLP {
    pub gate_proj: WeightMatrix, // [intermediate_size, hidden_size]
    pub up_proj: WeightMatrix,   // [intermediate_size, hidden_size]
    pub down_proj: WeightMatrix, // [hidden_size, intermediate_size]
}

impl MLP {
    pub fn new(gate_proj: WeightMatrix, up_proj: WeightMatrix, down_proj: WeightMatrix) -> Self {
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
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let gate_slice = state.mlp_gate.as_slice_mut().unwrap();
        self.gate_proj.matmul_vec_into(hidden_slice, gate_slice);

        // up = up_proj @ hidden_state
        let up_slice = state.mlp_up.as_slice_mut().unwrap();
        self.up_proj.matmul_vec_into(hidden_slice, up_slice);

        // Apply SiLU to gate
        let gate_activated = kernels::silu(&state.mlp_gate);

        // Element-wise multiply: gate * up
        for i in 0..state.mlp_gate.len() {
            state.mlp_gate[i] = gate_activated[i] * state.mlp_up[i];
        }

        // down_proj @ (gate * up)
        let gate_slice = state.mlp_gate.as_slice().unwrap();
        let hidden_out_slice = state.hidden_state.as_slice_mut().unwrap();
        self.down_proj.matmul_vec_into(gate_slice, hidden_out_slice);
    }

    /// Optimized forward pass: uses parallel matmul when available
    pub fn fast_forward(&self, state: &mut InferenceState) {
        // gate = gate_proj @ hidden_state
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let gate_slice = state.mlp_gate.as_slice_mut().unwrap();
        self.gate_proj.matmul_vec_into(hidden_slice, gate_slice);

        // up = up_proj @ hidden_state
        let up_slice = state.mlp_up.as_slice_mut().unwrap();
        self.up_proj.matmul_vec_into(hidden_slice, up_slice);

        // Apply SiLU to gate (uses SIMD when that feature is enabled)
        let gate_activated = kernels::fast_silu(&state.mlp_gate);

        // Element-wise multiply: gate * up
        for i in 0..state.mlp_gate.len() {
            state.mlp_gate[i] = gate_activated[i] * state.mlp_up[i];
        }

        // down_proj @ (gate * up)
        let gate_slice = state.mlp_gate.as_slice().unwrap();
        let hidden_out_slice = state.hidden_state.as_slice_mut().unwrap();
        self.down_proj.matmul_vec_into(gate_slice, hidden_out_slice);
    }
}
