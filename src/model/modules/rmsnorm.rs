use super::super::InferenceState;
use crate::kernels;
use ndarray::Array1;

/// RMSNorm layer
pub struct RMSNorm {
    pub weight: Array1<f32>, // [hidden_size]
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Array1<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Forward pass: normalize hidden_state
    pub fn forward(&self, state: &mut InferenceState) {
        kernels::rmsnorm(&mut state.hidden_state, &self.weight, self.eps);
    }
}
