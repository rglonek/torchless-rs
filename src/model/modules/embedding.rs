use super::super::InferenceState;
use ndarray::Array2;

/// Embedding lookup table
pub struct Embedding {
    pub table: Array2<f32>, // [vocab_size, hidden_size]
}

impl Embedding {
    pub fn new(table: Array2<f32>) -> Self {
        Self { table }
    }

    /// Forward pass: look up token embedding and copy to hidden_state
    pub fn forward(&self, state: &mut InferenceState, token_id: u32) {
        let token_idx = token_id as usize;
        let embedding = self.table.row(token_idx);
        state.hidden_state.assign(&embedding);
    }
}
