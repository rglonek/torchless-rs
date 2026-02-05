use super::super::InferenceState;
use crate::loader::Parameters;

/// Lazy embedding lookup table that accesses the memory-mapped tensor on demand.
/// Instead of storing the full embedding table in memory (~vocab_size * hidden_size * 4 bytes),
/// this only stores the tensor name and looks up embeddings lazily.
pub struct LazyEmbedding {
    pub tensor_name: String,
}

impl LazyEmbedding {
    pub fn new(tensor_name: String) -> Self {
        Self { tensor_name }
    }

    /// Forward pass: look up token embedding from memory-mapped tensor and copy to hidden_state.
    /// This dequantizes a single row on-demand without loading the entire embedding table.
    pub fn forward(&self, state: &mut InferenceState, token_id: u32, params: &Parameters) {
        let token_idx = token_id as usize;
        let embed_view = params.get_tensor_view(&self.tensor_name).unwrap();

        // Get the embedding row lazily (dequantizes if needed)
        let embedding = embed_view.get_row(token_idx);

        // Copy to hidden state
        for (i, &v) in embedding.iter().enumerate() {
            state.hidden_state[i] = v;
        }
    }
}
