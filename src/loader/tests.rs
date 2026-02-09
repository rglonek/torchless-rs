use super::{Config, Parameters, TensorDtype};
use std::collections::HashMap;

#[test]
fn test_config_deserialization() {
    let json = r#"{
        "hidden_size": "4096",
        "intermediate_size": "14336",
        "n_layers": "32",
        "n_heads": "32",
        "n_kv_heads": "8",
        "vocab_size": "32000",
        "max_position_embeddings": "32768",
        "sliding_window": "4096",
        "rope_theta": "10000.0",
        "norm_eps": "0.00001",
        "act_type": "silu",
        "quant": "f32"
    }"#;

    let metadata: HashMap<String, String> = serde_json::from_str(json).unwrap();

    let config = Config {
        hidden_size: metadata["hidden_size"].parse().unwrap(),
        intermediate_size: metadata["intermediate_size"].parse().unwrap(),
        n_layers: metadata["n_layers"].parse().unwrap(),
        n_heads: metadata["n_heads"].parse().unwrap(),
        n_kv_heads: metadata["n_kv_heads"].parse().unwrap(),
        vocab_size: metadata["vocab_size"].parse().unwrap(),
        max_position_embeddings: metadata["max_position_embeddings"].parse().unwrap(),
        sliding_window: metadata["sliding_window"].parse().unwrap(),
        rope_theta: metadata["rope_theta"].parse().unwrap(),
        norm_eps: metadata["norm_eps"].parse().unwrap(),
        act_type: metadata["act_type"].clone(),
        quant: metadata["quant"].clone(),
        n_routed_experts: 0,
        n_experts_per_token: 0,
        n_shared_experts: 0,
        moe_intermediate_size: 0,
        first_moe_layer: 0,
    };

    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.intermediate_size, 14336);
    assert_eq!(config.n_layers, 32);
    assert_eq!(config.n_heads, 32);
    assert_eq!(config.n_kv_heads, 8);
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.max_position_embeddings, 32768);
    assert_eq!(config.rope_theta, 10000.0);
    assert!((config.norm_eps - 1e-5).abs() < 1e-8);
}

// Integration tests with test model binary
#[test]
fn test_load_model_binary() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    assert_eq!(params.config.hidden_size, 32);
    assert_eq!(params.config.intermediate_size, 64);
    assert_eq!(params.config.n_layers, 2);
    assert_eq!(params.config.vocab_size, 300);
}

#[test]
fn test_load_tensor_f32() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    // Load norm weight
    let tensor = params.get_tensor("model.norm.weight").unwrap();
    assert_eq!(tensor.len(), 32); // hidden_size

    // Load embedding table
    let embed = params.get_tensor("model.embed_tokens.weight").unwrap();
    assert_eq!(embed.len(), 300 * 32); // vocab_size * hidden_size
}

#[test]
fn test_load_layer_weights() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    // Load layer 0 attention weights
    let q_proj = params
        .get_tensor("model.layers.0.self_attn.q_proj.weight")
        .unwrap();
    assert_eq!(q_proj.len(), 32 * 32); // hidden_size * hidden_size

    let k_proj = params
        .get_tensor("model.layers.0.self_attn.k_proj.weight")
        .unwrap();
    assert_eq!(k_proj.len(), 16 * 32); // (hidden_size / 2) * hidden_size

    // Load layer 0 MLP weights
    let gate = params
        .get_tensor("model.layers.0.mlp.gate_proj.weight")
        .unwrap();
    assert_eq!(gate.len(), 64 * 32); // intermediate_size * hidden_size
}

// =============================================================================
// TensorView Tests - Lazy Loading
// =============================================================================

#[test]
fn test_tensor_view_f32() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    // Get tensor view (lazy - no copy)
    let view = params.get_tensor_view("model.norm.weight").unwrap();

    assert_eq!(view.dtype, TensorDtype::F32);
    assert_eq!(view.shape, vec![32]); // hidden_size
    assert_eq!(view.numel(), 32);
    assert!(view.scales.is_none()); // f32 has no scales
}

#[test]
fn test_tensor_view_get_row() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    // Get embedding table view
    let view = params.get_tensor_view("model.embed_tokens.weight").unwrap();

    assert_eq!(view.nrows(), 300); // vocab_size
    assert_eq!(view.ncols(), 32); // hidden_size

    // Get row 0 lazily
    let row0 = view.get_row(0);
    assert_eq!(row0.len(), 32);

    // Get another row
    let row10 = view.get_row(10);
    assert_eq!(row10.len(), 32);

    // Rows should be different
    assert_ne!(row0, row10);
}

#[test]
fn test_tensor_view_matmul_vec() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    // Get Q projection view [hidden_size, hidden_size] = [32, 32]
    let view = params
        .get_tensor_view("model.layers.0.self_attn.q_proj.weight")
        .unwrap();

    assert_eq!(view.nrows(), 32);
    assert_eq!(view.ncols(), 32);

    // Create input vector
    let x = vec![1.0f32; 32];

    // Lazy matmul
    let result = view.matmul_vec(&x);
    assert_eq!(result.len(), 32);

    // Compare with eager loading
    let tensor = params
        .get_tensor("model.layers.0.self_attn.q_proj.weight")
        .unwrap();

    // Manual matmul for comparison
    let mut expected = vec![0.0f32; 32];
    for row in 0..32 {
        let mut dot = 0.0f32;
        for col in 0..32 {
            dot += tensor[row * 32 + col] * x[col];
        }
        expected[row] = dot;
    }

    // Results should match (within floating point tolerance)
    for i in 0..32 {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "Mismatch at {}: {} vs {}",
            i,
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_tensor_view_matmul_vec_into() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    let view = params
        .get_tensor_view("model.layers.0.self_attn.q_proj.weight")
        .unwrap();

    let x = vec![0.5f32; 32];
    let mut out = vec![0.0f32; 32];

    // Lazy matmul into pre-allocated buffer
    view.matmul_vec_into(&x, &mut out);

    // Verify non-zero output
    let sum: f32 = out.iter().sum();
    assert!(sum.abs() > 0.0, "Output should be non-zero");
}

#[test]
fn test_tensor_view_matches_eager_loading() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    // Get same tensor both ways
    let lazy_view = params.get_tensor_view("model.norm.weight").unwrap();
    let eager_vec = params.get_tensor("model.norm.weight").unwrap();

    // Get row from lazy view (for 1D tensor, get_row(0) returns the whole tensor)
    let lazy_row = lazy_view.get_row(0);

    // Should match
    assert_eq!(lazy_row.len(), eager_vec.len());
    for i in 0..lazy_row.len() {
        assert!(
            (lazy_row[i] - eager_vec[i]).abs() < 1e-6,
            "Mismatch at {}: {} vs {}",
            i,
            lazy_row[i],
            eager_vec[i]
        );
    }
}

#[test]
fn test_tensor_view_rows_iterator() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    let view = params.get_tensor_view("model.embed_tokens.weight").unwrap();

    // Count rows via iterator
    let row_count = view.rows().count();
    assert_eq!(row_count, 300);

    // Verify first row matches get_row(0)
    let first_via_iter = view.rows().next().unwrap();
    let first_via_get = view.get_row(0);
    assert_eq!(first_via_iter, first_via_get);
}
