use super::super::InferenceState;
use super::MLP;
use crate::loader::WeightMatrix;
use ndarray::Array1;

/// Router / gating module for Mixture-of-Experts.
/// Computes expert selection probabilities via a learned gate weight matrix.
pub struct MoERouter {
    /// Gate weight matrix: [n_experts, hidden_size]
    pub gate_weight: WeightMatrix,
    /// Total number of routed experts
    pub n_experts: usize,
    /// Number of experts to activate per token (top-k)
    pub top_k: usize,
}

impl MoERouter {
    pub fn new(gate_weight: WeightMatrix, n_experts: usize, top_k: usize) -> Self {
        Self {
            gate_weight,
            n_experts,
            top_k,
        }
    }

    /// Compute top-k expert indices and their normalized weights.
    /// Returns (expert_indices, expert_weights) where:
    /// - expert_indices: Vec of top-k expert indices
    /// - expert_weights: Vec of corresponding softmax-normalized weights
    pub fn route(&self, state: &InferenceState) -> (Vec<usize>, Vec<f32>) {
        // router_logits = gate_weight @ hidden_state -> [n_experts]
        let logits_vec = self
            .gate_weight
            .matmul_vec(state.hidden_state.as_slice().unwrap());
        let router_logits = Array1::from_vec(logits_vec);

        self.top_k_softmax(&router_logits)
    }

    /// Select top-k experts and compute softmax-normalized weights over them.
    fn top_k_softmax(&self, logits: &Array1<f32>) -> (Vec<usize>, Vec<f32>) {
        let n = logits.len();
        assert!(self.top_k <= n, "top_k must be <= n_experts");

        // Find top-k indices by partial sort
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        // Sort descending by logit value
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

/// Mixture-of-Experts FFN layer (eager variant).
/// Contains a router, a set of expert MLPs, and optionally a shared expert.
pub struct MoE {
    pub router: MoERouter,
    pub experts: Vec<MLP>,
    /// Shared expert that always activates (DeepSeek-style). None for Mixtral-style.
    pub shared_expert: Option<MLP>,
}

impl MoE {
    pub fn new(router: MoERouter, experts: Vec<MLP>, shared_expert: Option<MLP>) -> Self {
        Self {
            router,
            experts,
            shared_expert,
        }
    }

    /// Forward pass: route to top-k experts, run selected FFNs, weighted sum.
    /// Overwrites state.hidden_state with the MoE output.
    pub fn forward(&self, state: &mut InferenceState) {
        // 1. Router: select top-k experts and get weights
        let (expert_indices, expert_weights) = self.router.route(state);

        // 2. Save input hidden state (needed for each expert)
        let input = state.hidden_state.clone();

        // 3. Accumulate weighted expert outputs
        let hidden_size = state.hidden_state.len();
        let mut accumulated = Array1::<f32>::zeros(hidden_size);

        for (idx, &expert_idx) in expert_indices.iter().enumerate() {
            let weight = expert_weights[idx];

            // Restore input for this expert
            state.hidden_state.assign(&input);

            // Run expert FFN (updates state.hidden_state in place)
            self.experts[expert_idx].forward(state);

            // Accumulate weighted output
            for j in 0..hidden_size {
                accumulated[j] += weight * state.hidden_state[j];
            }
        }

        // 4. If shared expert exists, run it and add to output
        if let Some(ref shared) = self.shared_expert {
            state.hidden_state.assign(&input);
            shared.forward(state);
            // Add shared expert output (unweighted)
            for j in 0..hidden_size {
                accumulated[j] += state.hidden_state[j];
            }
        }

        // 5. Write final result
        state.hidden_state.assign(&accumulated);
    }

    /// Optimized forward pass: uses parallel matmul when available.
    pub fn fast_forward(&self, state: &mut InferenceState) {
        // 1. Router: select top-k experts and get weights
        let (expert_indices, expert_weights) = self.router.route(state);

        // 2. Save input hidden state
        let input = state.hidden_state.clone();

        // 3. Accumulate weighted expert outputs
        let hidden_size = state.hidden_state.len();
        let mut accumulated = Array1::<f32>::zeros(hidden_size);

        for (idx, &expert_idx) in expert_indices.iter().enumerate() {
            let weight = expert_weights[idx];

            // Restore input for this expert
            state.hidden_state.assign(&input);

            // Run expert FFN with optimized kernels
            self.experts[expert_idx].fast_forward(state);

            // Accumulate weighted output
            for j in 0..hidden_size {
                accumulated[j] += weight * state.hidden_state[j];
            }
        }

        // 4. Shared expert
        if let Some(ref shared) = self.shared_expert {
            state.hidden_state.assign(&input);
            shared.fast_forward(state);
            for j in 0..hidden_size {
                accumulated[j] += state.hidden_state[j];
            }
        }

        // 5. Write final result
        state.hidden_state.assign(&accumulated);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::{Config, WeightMatrix};
    use ndarray::Array2;

    fn make_test_config() -> Config {
        Config {
            hidden_size: 4,
            intermediate_size: 8,
            n_layers: 1,
            n_heads: 1,
            n_kv_heads: 1,
            vocab_size: 10,
            max_position_embeddings: 32,
            sliding_window: 0,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            act_type: "silu".to_string(),
            quant: "none".to_string(),
            n_routed_experts: 4,
            n_experts_per_token: 2,
            n_shared_experts: 0,
            moe_intermediate_size: 8,
            first_moe_layer: 0,
            head_dim: 0,
            swiglu_limit: 0.0,
            attention_sliding_window: 0,
            attention_bias: false,
        }
    }

    #[test]
    fn test_router_top_k() {
        let config = make_test_config();
        let mut state = InferenceState::new(config);
        state.hidden_state.fill(1.0);

        // Create a gate weight that clearly favors experts 1 and 3
        let mut gate = Array2::zeros((4, 4));
        gate[[0, 0]] = 0.1; // expert 0: weak
        gate[[1, 0]] = 10.0; // expert 1: strong
        gate[[2, 0]] = 0.2; // expert 2: weak
        gate[[3, 0]] = 5.0; // expert 3: moderate

        let router = MoERouter::new(WeightMatrix::from_f32(gate), 4, 2);
        let (indices, weights) = router.route(&state);

        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        // Expert 1 should be selected (highest logit)
        assert!(indices.contains(&1));
        // Expert 3 should be selected (second highest)
        assert!(indices.contains(&3));
        // Weights should sum to 1.0
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_moe_forward_basic() {
        let config = make_test_config();
        let mut state = InferenceState::new(config);
        state.hidden_state.fill(1.0);

        // Create identity-like gate that selects expert 0
        let mut gate = Array2::zeros((4, 4));
        gate[[0, 0]] = 100.0; // strongly favor expert 0

        let router = MoERouter::new(WeightMatrix::from_f32(gate), 4, 1);

        // Create trivial experts (identity-like: down_proj @ (gate_proj @ x * up_proj @ x))
        let experts: Vec<MLP> = (0..4)
            .map(|_| {
                MLP::new(
                    WeightMatrix::from_f32(Array2::from_elem((8, 4), 0.1)),
                    WeightMatrix::from_f32(Array2::from_elem((8, 4), 0.1)),
                    WeightMatrix::from_f32(Array2::from_elem((4, 8), 0.1)),
                )
            })
            .collect();

        let moe = MoE::new(router, experts, None);
        moe.forward(&mut state);

        // Output should be non-zero (exact value depends on SiLU activation)
        let norm: f32 = state.hidden_state.iter().map(|x| x * x).sum();
        assert!(norm > 0.0, "MoE output should be non-zero");
    }
}
