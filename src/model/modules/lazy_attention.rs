use super::super::InferenceState;
use crate::kernels;
use crate::loader::Parameters;
use ndarray::{s, ArrayView2};

/// Lazy multi-head attention with grouped-query attention (GQA).
/// Stores tensor names instead of the actual weight data.
/// Weights are accessed lazily from the memory-mapped file during forward pass.
pub struct LazyAttention {
    pub layer_idx: usize,
    pub q_proj_name: String,
    pub k_proj_name: String,
    pub v_proj_name: String,
    pub o_proj_name: String,
}

impl LazyAttention {
    pub fn new(
        layer_idx: usize,
        q_proj_name: String,
        k_proj_name: String,
        v_proj_name: String,
        o_proj_name: String,
    ) -> Self {
        Self {
            layer_idx,
            q_proj_name,
            k_proj_name,
            v_proj_name,
            o_proj_name,
        }
    }

    /// Forward pass: multi-head attention with GQA using lazy tensor loading.
    /// Projection weights are read from the memory-mapped file on demand.
    pub fn forward(&self, state: &mut InferenceState, params: &Parameters) {
        let head_dim = state.config.hidden_size / state.config.n_heads;

        // Get tensor views for projections (lazy - no copy yet)
        let q_proj = params.get_tensor_view(&self.q_proj_name).unwrap();
        let k_proj = params.get_tensor_view(&self.k_proj_name).unwrap();
        let v_proj = params.get_tensor_view(&self.v_proj_name).unwrap();
        let o_proj = params.get_tensor_view(&self.o_proj_name).unwrap();

        // Project to Q, K, V using lazy matmul (fused dequant + matmul)
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let q_flat_slice = state.q_flat.as_slice_mut().unwrap();
        let k_flat_slice = state.k_flat.as_slice_mut().unwrap();
        let v_flat_slice = state.v_flat.as_slice_mut().unwrap();

        q_proj.matmul_vec_into(hidden_slice, q_flat_slice);
        k_proj.matmul_vec_into(hidden_slice, k_flat_slice);
        v_proj.matmul_vec_into(hidden_slice, v_flat_slice);

        // Copy flat buffers into shaped state arrays
        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.q_state[[h, d]] = state.q_flat[h * head_dim + d];
            }
        }
        for h in 0..state.config.n_kv_heads {
            for d in 0..head_dim {
                state.k_state[[h, d]] = state.k_flat[h * head_dim + d];
                state.v_state[[h, d]] = state.v_flat[h * head_dim + d];
            }
        }

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
        let kv_groups = state.config.n_heads / state.config.n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Zero out context
        state.context.fill(0.0);

        for h in 0..state.config.n_heads {
            let kv_head = h / kv_groups;
            let seq_len = state.pos + 1;

            // Get query for this head
            let q_head = state.q_state.row(h);

            // Get K cache as f32 slice
            let k_data = state
                .k_cache
                .get_slice_f32(self.layer_idx, kv_head, seq_len);
            let k_arr = ArrayView2::from_shape((seq_len, head_dim), &k_data).unwrap();

            // Compute attention scores
            {
                let mut scores_slice = state.scores.slice_mut(s![h, ..seq_len]);
                kernels::compute_attention_scores(q_head, k_arr, &mut scores_slice, scale);
                kernels::softmax_view(&mut scores_slice);
            }

            // Get V cache as f32 slice
            let v_data = state
                .v_cache
                .get_slice_f32(self.layer_idx, kv_head, seq_len);
            let v_arr = ArrayView2::from_shape((seq_len, head_dim), &v_data).unwrap();

            // Compute weighted sum
            let scores_view = state.scores.slice(s![h, ..seq_len]);
            let mut context_row = state.context.slice_mut(s![h, ..]);
            kernels::weighted_sum_rows(scores_view, v_arr, &mut context_row);
        }

        // Flatten context
        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.context_flat[h * head_dim + d] = state.context[[h, d]];
            }
        }

        // Apply output projection (lazy)
        let context_slice = state.context_flat.as_slice().unwrap();
        let hidden_out_slice = state.hidden_state.as_slice_mut().unwrap();
        o_proj.matmul_vec_into(context_slice, hidden_out_slice);
    }

    /// Optimized forward pass with SIMD kernels where available.
    pub fn fast_forward(&self, state: &mut InferenceState, params: &Parameters) {
        let head_dim = state.config.hidden_size / state.config.n_heads;

        // Get tensor views for projections
        let q_proj = params.get_tensor_view(&self.q_proj_name).unwrap();
        let k_proj = params.get_tensor_view(&self.k_proj_name).unwrap();
        let v_proj = params.get_tensor_view(&self.v_proj_name).unwrap();
        let o_proj = params.get_tensor_view(&self.o_proj_name).unwrap();

        // Project to Q, K, V
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        let q_flat_slice = state.q_flat.as_slice_mut().unwrap();
        let k_flat_slice = state.k_flat.as_slice_mut().unwrap();
        let v_flat_slice = state.v_flat.as_slice_mut().unwrap();

        q_proj.matmul_vec_into(hidden_slice, q_flat_slice);
        k_proj.matmul_vec_into(hidden_slice, k_flat_slice);
        v_proj.matmul_vec_into(hidden_slice, v_flat_slice);

        // Copy flat buffers into shaped state arrays
        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.q_state[[h, d]] = state.q_flat[h * head_dim + d];
            }
        }
        for h in 0..state.config.n_kv_heads {
            for d in 0..head_dim {
                state.k_state[[h, d]] = state.k_flat[h * head_dim + d];
                state.v_state[[h, d]] = state.v_flat[h * head_dim + d];
            }
        }

        // Generate RoPE embeddings
        let (cos, sin) = kernels::rope_embeddings(&state.inv_freq, state.pos);
        state.cos.assign(&cos);
        state.sin.assign(&sin);

        // Apply RoPE (uses SIMD when available)
        kernels::fast_apply_rope(&mut state.q_state, &state.cos, &state.sin);
        kernels::fast_apply_rope(&mut state.k_state, &state.cos, &state.sin);

        // Push K, V to cache
        state.push_kv(self.layer_idx);

        // Attention computation
        let kv_groups = state.config.n_heads / state.config.n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        state.context.fill(0.0);

        for h in 0..state.config.n_heads {
            let kv_head = h / kv_groups;
            let seq_len = state.pos + 1;

            let q_head = state.q_state.row(h);
            let k_data = state
                .k_cache
                .get_slice_f32(self.layer_idx, kv_head, seq_len);
            let k_arr = ArrayView2::from_shape((seq_len, head_dim), &k_data).unwrap();

            {
                let mut scores_slice = state.scores.slice_mut(s![h, ..seq_len]);
                kernels::fast_compute_attention_scores(q_head, k_arr, &mut scores_slice, scale);
                kernels::fast_softmax_view(&mut scores_slice);
            }

            let v_data = state
                .v_cache
                .get_slice_f32(self.layer_idx, kv_head, seq_len);
            let v_arr = ArrayView2::from_shape((seq_len, head_dim), &v_data).unwrap();
            let scores_view = state.scores.slice(s![h, ..seq_len]);
            let mut context_row = state.context.slice_mut(s![h, ..]);
            kernels::fast_weighted_sum_rows(scores_view, v_arr, &mut context_row);
        }

        // Flatten context
        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.context_flat[h * head_dim + d] = state.context[[h, d]];
            }
        }

        // Output projection
        let context_slice = state.context_flat.as_slice().unwrap();
        let hidden_out_slice = state.hidden_state.as_slice_mut().unwrap();
        o_proj.matmul_vec_into(context_slice, hidden_out_slice);
    }
}
