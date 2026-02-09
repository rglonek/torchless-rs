use crate::loader::{Config, Parameters};
use crate::model::{Attention, Embedding, InferenceState, LazyMistral, Mistral, RMSNorm, MLP};
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
        n_routed_experts: 0,
        n_experts_per_token: 0,
        n_shared_experts: 0,
        moe_intermediate_size: 0,
        first_moe_layer: 0,
        head_dim: 0,
        swiglu_limit: 0.0,
        attention_sliding_window: 0,
        attention_bias: false,
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

// =============================================================================
// Lazy Model Tests
// =============================================================================

#[test]
fn test_lazy_mistral_load() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let lazy_model = LazyMistral::load(&params).unwrap();

    // Verify config loaded
    assert_eq!(lazy_model.config.hidden_size, 32);
    assert_eq!(lazy_model.config.n_layers, 2);

    // Verify layers created
    assert_eq!(lazy_model.layers.len(), 2);
}

#[test]
fn test_lazy_vs_eager_forward_produces_same_results() {
    // Load parameters
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    // Create eager model (consumes params)
    let params_for_eager = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let eager_model = Mistral::load(params_for_eager).unwrap();

    // Create lazy model (borrows params)
    let lazy_model = LazyMistral::load(&params).unwrap();

    // Create inference states
    let mut eager_state = InferenceState::new(eager_model.config.clone());
    let mut lazy_state = InferenceState::new(lazy_model.config.clone());

    // Run forward pass with same token
    let token = 1u32;
    eager_model.forward(&mut eager_state, token, false);
    lazy_model.forward(&mut lazy_state, token, false);

    // Results should match (within floating point tolerance)
    for i in 0..lazy_state.logits.len() {
        let diff = (lazy_state.logits[i] - eager_state.logits[i]).abs();
        assert!(
            diff < 1e-4,
            "Logits mismatch at {}: lazy={}, eager={}, diff={}",
            i,
            lazy_state.logits[i],
            eager_state.logits[i],
            diff
        );
    }
}

#[test]
fn test_lazy_model_multiple_tokens() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let lazy_model = LazyMistral::load(&params).unwrap();
    let mut state = InferenceState::new(lazy_model.config.clone());

    // Process multiple tokens
    let tokens = [1u32, 5, 10, 15, 20];

    for &token in &tokens {
        lazy_model.forward(&mut state, token, false);
        state.pos += 1;

        // Verify state is not all zeros
        let sum: f32 = state.logits.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "Logits should be non-zero after forward pass");
    }

    // Position should have advanced
    assert_eq!(state.pos, tokens.len());
}

#[test]
fn test_lazy_embedding_lookup() {
    use crate::model::modules::LazyEmbedding;

    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let mut state = InferenceState::new(params.config.clone());

    let lazy_embed = LazyEmbedding::new("model.embed_tokens.weight".to_string());

    // Lookup token 0
    lazy_embed.forward(&mut state, 0, &params);
    let state0 = state.hidden_state.clone();

    // Lookup different token
    lazy_embed.forward(&mut state, 10, &params);

    // Should be different
    assert_ne!(state.hidden_state, state0);

    // Compare with eager embedding
    let embed_data = params.get_tensor("model.embed_tokens.weight").unwrap();
    let embed_table = Array2::from_shape_vec((300, 32), embed_data).unwrap();
    let eager_embed = Embedding::new(embed_table);

    let mut eager_state = InferenceState::new(params.config.clone());
    eager_embed.forward(&mut eager_state, 10);

    // Should match lazy embedding
    for i in 0..state.hidden_state.len() {
        assert!(
            (state.hidden_state[i] - eager_state.hidden_state[i]).abs() < 1e-6,
            "Embedding mismatch at {}: lazy={}, eager={}",
            i,
            state.hidden_state[i],
            eager_state.hidden_state[i]
        );
    }
}
