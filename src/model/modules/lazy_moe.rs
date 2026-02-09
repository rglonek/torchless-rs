use super::super::InferenceState;
use crate::kernels;
use crate::loader::Parameters;
use ndarray::Array1;

/// Lazy router / gating module for Mixture-of-Experts.
/// Stores the gate weight tensor name instead of the actual data.
pub struct LazyMoERouter {
    pub gate_weight_name: String,
    /// Total number of routed experts
    pub n_experts: usize,
    /// Number of experts to activate per token (top-k)
    pub top_k: usize,
}

impl LazyMoERouter {
    pub fn new(gate_weight_name: String, n_experts: usize, top_k: usize) -> Self {
        Self {
            gate_weight_name,
            n_experts,
            top_k,
        }
    }

    /// Compute top-k expert indices and their normalized weights using lazy tensor access.
    pub fn route(&self, state: &InferenceState, params: &Parameters) -> (Vec<usize>, Vec<f32>) {
        // router_logits = gate_weight @ hidden_state -> [n_experts]
        let gate_view = params.get_tensor_view(&self.gate_weight_name).unwrap();
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let router_logits = gate_view.matmul_vec(hidden_slice);

        self.top_k_softmax(&router_logits)
    }

    /// Select top-k experts and compute softmax-normalized weights over them.
    fn top_k_softmax(&self, logits: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let n = logits.len();
        assert!(self.top_k <= n, "top_k must be <= n_experts");

        // Find top-k indices by partial sort
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_indices: Vec<usize> = indexed[..self.top_k].iter().map(|(i, _)| *i).collect();
        let top_logits: Vec<f32> = indexed[..self.top_k].iter().map(|(_, v)| *v).collect();

        // Softmax over top-k logits only
        let max_logit = top_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = top_logits.iter().map(|&v| (v - max_logit).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();
        let weights: Vec<f32> = exp_values.iter().map(|&v| v / sum_exp).collect();

        (top_indices, weights)
    }
}

/// Lazy expert FFN descriptor.
/// Stores tensor names for a single expert's projections.
pub struct LazyExpert {
    pub gate_proj_name: String,
    pub up_proj_name: String,
    pub down_proj_name: String,
}

impl LazyExpert {
    pub fn new(gate_proj_name: String, up_proj_name: String, down_proj_name: String) -> Self {
        Self {
            gate_proj_name,
            up_proj_name,
            down_proj_name,
        }
    }

    /// Run this expert's FFN on the current hidden state.
    /// Uses lazy tensor access from memory-mapped parameters.
    pub fn forward(&self, state: &mut InferenceState, params: &Parameters) {
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

        // Apply SiLU to gate
        let gate_activated = kernels::silu(&state.mlp_gate);

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

/// Lazy shared expert FFN descriptor.
/// Same as LazyExpert but conceptually always activated.
pub type LazySharedExpert = LazyExpert;

/// Lazy Mixture-of-Experts FFN layer.
/// Stores tensor names for the router, all experts, and optional shared expert.
/// Only accesses the tensors for the router and selected top-k experts during forward pass,
/// making it feasible to run models with hundreds of experts.
pub struct LazyMoE {
    pub router: LazyMoERouter,
    pub experts: Vec<LazyExpert>,
    /// Shared expert that always activates (DeepSeek-style). None for Mixtral-style.
    pub shared_expert: Option<LazySharedExpert>,
}

impl LazyMoE {
    pub fn new(
        router: LazyMoERouter,
        experts: Vec<LazyExpert>,
        shared_expert: Option<LazySharedExpert>,
    ) -> Self {
        Self {
            router,
            experts,
            shared_expert,
        }
    }

    /// Forward pass with lazy tensor loading.
    /// Only the router gate and the selected top-k expert weights are accessed
    /// from the memory map, keeping memory usage bounded regardless of total expert count.
    pub fn forward(&self, state: &mut InferenceState, params: &Parameters) {
        // 1. Router: select top-k experts and get weights
        let (expert_indices, expert_weights) = self.router.route(state, params);

        // 2. Save input hidden state
        let input = state.hidden_state.clone();

        // 3. Accumulate weighted expert outputs
        let hidden_size = state.hidden_state.len();
        let mut accumulated = Array1::<f32>::zeros(hidden_size);

        for (idx, &expert_idx) in expert_indices.iter().enumerate() {
            let weight = expert_weights[idx];

            // Restore input for this expert
            state.hidden_state.assign(&input);

            // Run expert FFN (lazy tensor access)
            self.experts[expert_idx].forward(state, params);

            // Accumulate weighted output
            for j in 0..hidden_size {
                accumulated[j] += weight * state.hidden_state[j];
            }
        }

        // 4. Shared expert (if present)
        if let Some(ref shared) = self.shared_expert {
            state.hidden_state.assign(&input);
            shared.forward(state, params);
            for j in 0..hidden_size {
                accumulated[j] += state.hidden_state[j];
            }
        }

        // 5. Write final result
        state.hidden_state.assign(&accumulated);
    }

    /// Optimized forward pass with lazy tensor loading.
    /// Uses SIMD kernels where available.
    pub fn fast_forward(&self, state: &mut InferenceState, params: &Parameters) {
        // Same logic as forward -- the lazy tensor views already use optimized
        // dequantize+matmul paths, so the main optimization here is the same
        // as forward. Future: could parallelize expert computation.
        self.forward(state, params);
    }
}
