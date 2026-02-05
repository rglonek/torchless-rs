use super::super::InferenceState;
use crate::kernels;
use crate::loader::Parameters;

/// Lazy MLP (feedforward) layer with SwiGLU activation.
/// Stores tensor names instead of the actual weight data.
/// Weights are accessed lazily from the memory-mapped file during forward pass.
pub struct LazyMLP {
    pub gate_proj_name: String,
    pub up_proj_name: String,
    pub down_proj_name: String,
}

impl LazyMLP {
    pub fn new(gate_proj_name: String, up_proj_name: String, down_proj_name: String) -> Self {
        Self {
            gate_proj_name,
            up_proj_name,
            down_proj_name,
        }
    }

    /// Forward pass: SwiGLU feedforward with lazy tensor loading.
    /// out = down_proj @ (silu(gate_proj @ x) * (up_proj @ x))
    pub fn forward(&self, state: &mut InferenceState, params: &Parameters) {
        // Get tensor views (lazy - no data copy)
        let gate_proj = params.get_tensor_view(&self.gate_proj_name).unwrap();
        let up_proj = params.get_tensor_view(&self.up_proj_name).unwrap();
        let down_proj = params.get_tensor_view(&self.down_proj_name).unwrap();

        let hidden_slice = state.hidden_state.as_slice().unwrap();

        // gate = gate_proj @ hidden_state (lazy matmul)
        let gate_slice = state.mlp_gate.as_slice_mut().unwrap();
        gate_proj.matmul_vec_into(hidden_slice, gate_slice);

        // up = up_proj @ hidden_state (lazy matmul)
        let up_slice = state.mlp_up.as_slice_mut().unwrap();
        up_proj.matmul_vec_into(hidden_slice, up_slice);

        // Apply SiLU to gate
        let gate_activated = kernels::silu(&state.mlp_gate);

        // Element-wise multiply: gate * up
        for i in 0..state.mlp_gate.len() {
            state.mlp_gate[i] = gate_activated[i] * state.mlp_up[i];
        }

        // down_proj @ (gate * up) (lazy matmul)
        let gate_slice = state.mlp_gate.as_slice().unwrap();
        let hidden_out_slice = state.hidden_state.as_slice_mut().unwrap();
        down_proj.matmul_vec_into(gate_slice, hidden_out_slice);
    }

    /// Optimized forward pass: uses SIMD kernels where available.
    pub fn fast_forward(&self, state: &mut InferenceState, params: &Parameters) {
        // Get tensor views
        let gate_proj = params.get_tensor_view(&self.gate_proj_name).unwrap();
        let up_proj = params.get_tensor_view(&self.up_proj_name).unwrap();
        let down_proj = params.get_tensor_view(&self.down_proj_name).unwrap();

        let hidden_slice = state.hidden_state.as_slice().unwrap();

        // gate = gate_proj @ hidden_state
        let gate_slice = state.mlp_gate.as_slice_mut().unwrap();
        gate_proj.matmul_vec_into(hidden_slice, gate_slice);

        // up = up_proj @ hidden_state
        let up_slice = state.mlp_up.as_slice_mut().unwrap();
        up_proj.matmul_vec_into(hidden_slice, up_slice);

        // Apply SiLU to gate (uses SIMD when available)
        let gate_activated = kernels::fast_silu(&state.mlp_gate);

        // Element-wise multiply: gate * up
        for i in 0..state.mlp_gate.len() {
            state.mlp_gate[i] = gate_activated[i] * state.mlp_up[i];
        }

        // down_proj @ (gate * up)
        let gate_slice = state.mlp_gate.as_slice().unwrap();
        let hidden_out_slice = state.hidden_state.as_slice_mut().unwrap();
        down_proj.matmul_vec_into(gate_slice, hidden_out_slice);
    }
}
