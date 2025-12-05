use crate::loader::Config;
use crate::model::{Attention, Embedding, InferenceState, RMSNorm, MLP};
use ndarray::{Array1, Array2};

fn test_config() -> Config {
    Config {
        hidden_size: 128,
        intermediate_size: 256,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 1000,
        max_position_embeddings: 512,
        sliding_window: 128,
        rope_theta: 10000.0,
        norm_eps: 1e-5,
        act_type: "silu".to_string(),
        quant: "f32".to_string(),
    }
}

#[test]
fn test_embedding_forward() {
    let config = test_config();
    let mut state = InferenceState::new(config.clone());

    // Create embedding table
    let table = Array2::from_shape_fn((config.vocab_size, config.hidden_size), |(i, j)| {
        (i * config.hidden_size + j) as f32
    });
    let embedding = Embedding::new(table.clone());

    // Test lookup
    let token_id = 42u32;
    embedding.forward(&mut state, token_id);

    // Verify hidden_state matches the embedding row
    for j in 0..config.hidden_size {
        assert_eq!(state.hidden_state[j], table[[token_id as usize, j]]);
    }
}

#[test]
fn test_rmsnorm_forward() {
    let config = test_config();
    let mut state = InferenceState::new(config.clone());

    // Set hidden state to known values
    state.hidden_state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let weight = Array1::from_vec(vec![1.0, 1.0, 1.0, 1.0]);

    let norm = RMSNorm::new(weight, 1e-5);
    norm.forward(&mut state);

    // After RMSNorm, the RMS should be approximately 1
    let squares: f32 = state.hidden_state.iter().map(|v| v * v).sum();
    let rms = (squares / state.hidden_state.len() as f32).sqrt();
    assert!((rms - 1.0).abs() < 1e-2);
}

#[test]
fn test_mlp_forward() {
    let config = test_config();
    let mut state = InferenceState::new(config.clone());

    // Initialize hidden state
    state.hidden_state =
        Array1::from_vec((0..config.hidden_size).map(|i| i as f32 * 0.01).collect());

    // Create MLP weights
    let gate_proj =
        Array2::from_shape_fn((config.intermediate_size, config.hidden_size), |(i, j)| {
            (i + j) as f32 * 0.01
        });
    let up_proj =
        Array2::from_shape_fn((config.intermediate_size, config.hidden_size), |(i, j)| {
            (i as f32 - j as f32) * 0.01
        });
    let down_proj =
        Array2::from_shape_fn((config.hidden_size, config.intermediate_size), |(i, j)| {
            (i + j) as f32 * 0.001
        });

    let mlp = MLP::new(gate_proj, up_proj, down_proj);

    let hidden_before = state.hidden_state.clone();
    mlp.forward(&mut state);

    // Verify output changed
    assert_ne!(state.hidden_state, hidden_before);

    // Verify shape is preserved
    assert_eq!(state.hidden_state.len(), config.hidden_size);
}

#[test]
fn test_attention_shape_preservation() {
    let config = test_config();
    let mut state = InferenceState::new(config.clone());

    // Initialize hidden state with non-zero values
    state.hidden_state =
        Array1::from_vec((0..config.hidden_size).map(|i| i as f32 * 0.01).collect());

    // Create attention weights with non-zero values
    let head_dim = config.hidden_size / config.n_heads;
    let q_proj =
        Array2::from_shape_fn((config.n_heads * head_dim, config.hidden_size), |(i, j)| {
            (i as f32 + j as f32) * 0.01
        });
    let k_proj = Array2::from_shape_fn(
        (config.n_kv_heads * head_dim, config.hidden_size),
        |(i, j)| (i as f32 + j as f32) * 0.01,
    );
    let v_proj = Array2::from_shape_fn(
        (config.n_kv_heads * head_dim, config.hidden_size),
        |(i, j)| (i as f32 + j as f32) * 0.01,
    );
    let o_proj =
        Array2::from_shape_fn((config.hidden_size, config.n_heads * head_dim), |(i, j)| {
            (i as f32 + j as f32) * 0.01
        });

    let attention = Attention::new(0, q_proj, k_proj, v_proj, o_proj);

    attention.forward(&mut state);

    // Verify shape preservation
    assert_eq!(state.hidden_state.len(), config.hidden_size);

    // Verify KV cache was updated (should have non-zero values after projection)
    let has_nonzero = state
        .k_cache
        .slice(ndarray::s![0, 0, 0, ..])
        .iter()
        .any(|&v| v != 0.0);
    assert!(
        has_nonzero,
        "KV cache should have non-zero values after forward pass"
    );
}

#[test]
fn test_inference_state_initialization() {
    let config = test_config();
    let state = InferenceState::new(config.clone());

    // Verify all tensors are initialized with correct shapes
    assert_eq!(state.hidden_state.len(), config.hidden_size);
    assert_eq!(state.residual.len(), config.hidden_size);
    assert_eq!(state.logits.len(), config.vocab_size);
    assert_eq!(state.probs.len(), config.vocab_size);

    let head_dim = config.hidden_size / config.n_heads;
    assert_eq!(state.q_state.shape(), &[config.n_heads, head_dim]);
    assert_eq!(state.k_state.shape(), &[config.n_kv_heads, head_dim]);

    // Verify position starts at 0
    assert_eq!(state.pos, 0);
}

#[test]
fn test_kv_cache_push() {
    let config = test_config();
    let mut state = InferenceState::new(config.clone());

    // Set K and V states
    state.k_state.fill(1.5);
    state.v_state.fill(2.5);

    // Push to cache at layer 0, position 0
    state.push_kv(0);

    // Verify values were copied
    assert_eq!(state.k_cache[[0, 0, 0, 0]], 1.5);
    assert_eq!(state.v_cache[[0, 0, 0, 0]], 2.5);
}
