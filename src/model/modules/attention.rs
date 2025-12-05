use super::super::InferenceState;
use crate::kernels;
use ndarray::{s, Array1, Array2};

/// Multi-head attention with grouped-query attention (GQA)
pub struct Attention {
    pub layer_idx: usize,
    pub q_proj: Array2<f32>, // [n_heads * head_dim, hidden_size]
    pub k_proj: Array2<f32>, // [n_kv_heads * head_dim, hidden_size]
    pub v_proj: Array2<f32>, // [n_kv_heads * head_dim, hidden_size]
    pub o_proj: Array2<f32>, // [hidden_size, n_heads * head_dim]
}

impl Attention {
    pub fn new(
        layer_idx: usize,
        q_proj: Array2<f32>,
        k_proj: Array2<f32>,
        v_proj: Array2<f32>,
        o_proj: Array2<f32>,
    ) -> Self {
        Self {
            layer_idx,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        }
    }

    /// Forward pass: multi-head attention with GQA
    pub fn forward(&self, state: &mut InferenceState) {
        let head_dim = state.config.hidden_size / state.config.n_heads;

        // Project to Q, K, V
        let q_flat = kernels::matmul_vec(&self.q_proj, &state.hidden_state);
        let k_flat = kernels::matmul_vec(&self.k_proj, &state.hidden_state);
        let v_flat = kernels::matmul_vec(&self.v_proj, &state.hidden_state);

        // Reshape to [n_heads, head_dim] and [n_kv_heads, head_dim]
        state.q_state = q_flat
            .into_shape_with_order((state.config.n_heads, head_dim))
            .unwrap();
        state.k_state = k_flat
            .into_shape_with_order((state.config.n_kv_heads, head_dim))
            .unwrap();
        state.v_state = v_flat
            .into_shape_with_order((state.config.n_kv_heads, head_dim))
            .unwrap();

        // Generate RoPE embeddings for current position
        let (cos, sin) = kernels::rope_embeddings(&state.inv_freq, state.pos);
        state.cos.assign(&cos);
        state.sin.assign(&sin);

        // Apply RoPE to Q and K
        kernels::apply_rope(&mut state.q_state, &state.cos, &state.sin);
        kernels::apply_rope(&mut state.k_state, &state.cos, &state.sin);

        // Push K, V to cache
        state.push_kv(self.layer_idx);

        // Perform attention: for each query head, attend to corresponding KV head
        // GQA: multiple query heads share the same KV head
        let kv_groups = state.config.n_heads / state.config.n_kv_heads;

        // Zero out context
        state.context.fill(0.0);

        for h in 0..state.config.n_heads {
            let kv_head = h / kv_groups;
            let seq_len = state.pos + 1;

            // Get query for this head: [head_dim]
            let q_head = state.q_state.row(h);

            // Get K cache for this KV head up to current position: [seq_len, head_dim]
            let k_cache_slice = state
                .k_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);
            let k_head = k_cache_slice
                .to_owned()
                .into_shape_with_order((seq_len, head_dim))
                .unwrap();

            // Compute attention scores: K @ Q
            let mut scores = Array1::zeros(seq_len);
            for i in 0..seq_len {
                scores[i] = k_head.row(i).dot(&q_head);
            }

            // Scale by 1/sqrt(head_dim)
            let scale = 1.0 / (head_dim as f32).sqrt();
            scores.mapv_inplace(|v| v * scale);

            // Softmax
            kernels::softmax(&mut scores);

            // Get V cache for this KV head: [seq_len, head_dim]
            let v_cache_slice = state
                .v_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);
            let v_head = v_cache_slice
                .to_owned()
                .into_shape_with_order((seq_len, head_dim))
                .unwrap();

            // Weighted sum: scores @ V
            let context_head = kernels::row_matmul(&scores, &v_head);

            // Store in context[h, :]
            for d in 0..head_dim {
                state.context[[h, d]] = context_head[d];
            }
        }

        // Flatten context and apply output projection
        let context_flat = state
            .context
            .clone()
            .into_shape_with_order(state.config.hidden_size)
            .unwrap();
        state
            .hidden_state
            .assign(&kernels::matmul_vec(&self.o_proj, &context_flat));
    }
}
