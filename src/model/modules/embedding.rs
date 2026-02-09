use super::super::InferenceState;
use crate::loader::WeightMatrix;

/// Embedding lookup table
pub struct Embedding {
    pub table: WeightMatrix, // [vocab_size, hidden_size]
}

impl Embedding {
    pub fn new(table: WeightMatrix) -> Self {
        Self { table }
    }

    /// Forward pass: look up token embedding and copy to hidden_state
    pub fn forward(&self, state: &mut InferenceState, token_id: u32) {
        let token_idx = token_id as usize;
        let row = self.table.get_row(token_idx);
        for (i, &val) in row.iter().enumerate() {
            state.hidden_state[i] = val;
        }
    }
}
