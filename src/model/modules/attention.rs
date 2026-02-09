use super::super::InferenceState;
use crate::kernels;
use crate::loader::WeightMatrix;
use ndarray::s;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Multi-head attention with grouped-query attention (GQA)
pub struct Attention {
    pub layer_idx: usize,
    pub q_proj: WeightMatrix, // [n_heads * head_dim, hidden_size]
    pub k_proj: WeightMatrix, // [n_kv_heads * head_dim, hidden_size]
    pub v_proj: WeightMatrix, // [n_kv_heads * head_dim, hidden_size]
    pub o_proj: WeightMatrix, // [hidden_size, n_heads * head_dim]
}

impl Attention {
    pub fn new(
        layer_idx: usize,
        q_proj: WeightMatrix,
        k_proj: WeightMatrix,
        v_proj: WeightMatrix,
        o_proj: WeightMatrix,
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
    ///
    /// Optimized to minimize allocations:
    /// - Uses pre-allocated flat buffers for Q, K, V projections
    /// - Uses pre-allocated scores buffer from InferenceState
    /// - Uses views into KV cache instead of cloning
    /// - Uses pre-allocated context_flat buffer for output projection
    pub fn forward(&self, state: &mut InferenceState) {
        let head_dim = state.config.hidden_size / state.config.n_heads;

        // Project to Q, K, V using pre-allocated flat buffers (no allocation)
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        self.q_proj
            .matmul_vec_into(hidden_slice, state.q_flat.as_slice_mut().unwrap());
        self.k_proj
            .matmul_vec_into(hidden_slice, state.k_flat.as_slice_mut().unwrap());
        self.v_proj
            .matmul_vec_into(hidden_slice, state.v_flat.as_slice_mut().unwrap());

        // Copy flat buffers into shaped state arrays
        // (This avoids the allocation of into_shape_with_order which takes ownership)
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
        // GQA: multiple query heads share the same KV head
        let kv_groups = state.config.n_heads / state.config.n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Zero out context
        state.context.fill(0.0);

        for h in 0..state.config.n_heads {
            let kv_head = h / kv_groups;
            let seq_len = state.pos + 1;

            // Get query for this head: [head_dim] (view, no allocation)
            let q_head = state.q_state.row(h);

            // Get K cache view for this KV head up to current position: [seq_len, head_dim]
            // Use view directly instead of cloning with to_owned()
            let k_cache_view = state
                .k_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);

            // Compute attention scores directly into pre-allocated scores buffer
            // Use the h-th row of the scores matrix, sliced to seq_len
            {
                let mut scores_slice = state.scores.slice_mut(s![h, ..seq_len]);
                kernels::compute_attention_scores(q_head, k_cache_view, &mut scores_slice, scale);

                // Softmax in-place (use view variant to avoid allocation)
                kernels::softmax_view(&mut scores_slice);
            }

            // Get V cache view for this KV head: [seq_len, head_dim] (view, no allocation)
            let v_cache_view = state
                .v_cache
                .slice(s![self.layer_idx, kv_head, ..seq_len, ..]);

            // Compute weighted sum directly into context[h, :] (no allocation)
            let scores_view = state.scores.slice(s![h, ..seq_len]);
            let mut context_row = state.context.slice_mut(s![h, ..]);
            kernels::weighted_sum_rows(scores_view, v_cache_view, &mut context_row);
        }

        // Flatten context into pre-allocated context_flat buffer (no clone)
        for h in 0..state.config.n_heads {
            for d in 0..head_dim {
                state.context_flat[h * head_dim + d] = state.context[[h, d]];
            }
        }

        // Apply output projection
        self.o_proj.matmul_vec_into(
            state.context_flat.as_slice().unwrap(),
            state.hidden_state.as_slice_mut().unwrap(),
        );
    }

    /// Parallel forward pass: multi-head attention with GQA
    ///
    /// Parallelizes attention computation across heads using Rayon.
    /// Each head's computation is independent, allowing for parallel execution.
    #[cfg(feature = "parallel")]
    pub fn forward_parallel(&self, state: &mut InferenceState) {
        let head_dim = state.config.hidden_size / state.config.n_heads;
        let n_heads = state.config.n_heads;
        let n_kv_heads = state.config.n_kv_heads;
        let kv_groups = n_heads / n_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = state.pos + 1;

        // Project to Q, K, V
        let hidden_slice = state.hidden_state.as_slice().unwrap();
        self.q_proj
            .matmul_vec_into(hidden_slice, state.q_flat.as_slice_mut().unwrap());
        self.k_proj
            .matmul_vec_into(hidden_slice, state.k_flat.as_slice_mut().unwrap());
        self.v_proj
            .matmul_vec_into(hidden_slice, state.v_flat.as_slice_mut().unwrap());

        // Copy flat buffers into shaped state arrays
        for h in 0..n_heads {
            for d in 0..head_dim {
                state.q_state[[h, d]] = state.q_flat[h * head_dim + d];
            }
        }
        for h in 0..n_kv_heads {
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

        // Parallel attention computation across heads
        // We collect the head computations and then assign to avoid borrow issues
        let head_results: Vec<(usize, Vec<f32>)> = (0..n_heads)
            .into_par_iter()
            .map(|h| {
                let kv_head = h / kv_groups;

                // Get query for this head
                let q_head: Vec<f32> = (0..head_dim).map(|d| state.q_state[[h, d]]).collect();

                // Compute attention scores: scores[i] = k_cache[i].dot(q) * scale
                let mut scores = vec![0.0f32; seq_len];
                for i in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += state.k_cache[[self.layer_idx, kv_head, i, d]] * q_head[d];
                    }
                    scores[i] = dot * scale;
                }

                // Softmax
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for s in &mut scores {
                    *s /= sum;
                }

                // Compute weighted sum: context[d] = sum_i(scores[i] * v_cache[i, d])
                let mut context = vec![0.0f32; head_dim];
                for i in 0..seq_len {
                    let w = scores[i];
                    for d in 0..head_dim {
                        context[d] += w * state.v_cache[[self.layer_idx, kv_head, i, d]];
                    }
                }

                (h, context)
            })
            .collect();

        // Copy results back to state
        for (h, context) in head_results {
            for d in 0..head_dim {
                state.context[[h, d]] = context[d];
            }
        }

        // Flatten context into pre-allocated context_flat buffer
        for h in 0..n_heads {
            for d in 0..head_dim {
                state.context_flat[h * head_dim + d] = state.context[[h, d]];
            }
        }

        // Apply output projection
        self.o_proj.matmul_vec_into(
            state.context_flat.as_slice().unwrap(),
            state.hidden_state.as_slice_mut().unwrap(),
        );
    }

    /// Auto-selecting forward: uses parallel when available
    #[cfg(feature = "parallel")]
    pub fn fast_forward(&self, state: &mut InferenceState) {
        self.forward_parallel(state)
    }

    #[cfg(not(feature = "parallel"))]
    pub fn fast_forward(&self, state: &mut InferenceState) {
        self.forward(state)
    }
}
