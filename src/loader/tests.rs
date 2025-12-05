use super::{Config, Parameters};
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
