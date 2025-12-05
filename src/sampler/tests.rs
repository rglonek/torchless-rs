use super::{sample_greedy, sample_multinomial};
use crate::loader::Config;
use crate::model::InferenceState;

fn test_config() -> Config {
    Config {
        hidden_size: 64,
        intermediate_size: 128,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2,
        vocab_size: 100,
        max_position_embeddings: 512,
        sliding_window: 128,
        rope_theta: 10000.0,
        norm_eps: 1e-5,
        act_type: "silu".to_string(),
        quant: "f32".to_string(),
    }
}

#[test]
fn test_sample_greedy() {
    let config = test_config();
    let mut state = InferenceState::new(config);

    // Set first few logits with a clear maximum
    state.logits[0] = 0.1;
    state.logits[1] = 0.5;
    state.logits[2] = 0.3;
    state.logits[3] = 0.9;
    state.logits[4] = 0.2;

    let token = sample_greedy(&state);

    // Should select token 3 (highest logit)
    assert_eq!(token, 3);
}

#[test]
fn test_sample_multinomial_zero_temp() {
    let config = test_config();
    let mut state = InferenceState::new(config);

    // Set first few logits
    state.logits[0] = 0.1;
    state.logits[1] = 0.5;
    state.logits[2] = 0.3;
    state.logits[3] = 0.9;
    state.logits[4] = 0.2;

    // Zero temperature should be greedy
    let token = sample_multinomial(&mut state, 0.0);
    assert_eq!(token, 3);
}

#[test]
fn test_sample_multinomial_with_temp() {
    let config = test_config();
    let mut state = InferenceState::new(config.clone());

    // Set first few logits
    state.logits[0] = 1.0;
    state.logits[1] = 2.0;
    state.logits[2] = 3.0;
    state.logits[3] = 4.0;
    state.logits[4] = 5.0;

    // With temperature > 0, should sample from distribution
    let token = sample_multinomial(&mut state, 1.0);

    // Should return a valid token ID
    assert!(token < config.vocab_size as u32);

    // Probs should sum to 1 (after softmax)
    let sum: f32 = state.probs.sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_temperature_scaling() {
    let config = test_config();
    let mut state1 = InferenceState::new(config.clone());
    let mut state2 = InferenceState::new(config);

    // Set first few logits
    state1.logits[0] = 1.0;
    state1.logits[1] = 2.0;
    state1.logits[2] = 3.0;
    state2.logits[0] = 1.0;
    state2.logits[1] = 2.0;
    state2.logits[2] = 3.0;

    // Higher temperature should flatten distribution
    sample_multinomial(&mut state1, 0.5); // Lower temp
    sample_multinomial(&mut state2, 2.0); // Higher temp

    // With higher temp, distribution should be more uniform
    // (difference between max and min prob should be smaller)
    let diff1 = state1
        .probs
        .iter()
        .take(3)
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - state1
            .probs
            .iter()
            .take(3)
            .fold(f32::INFINITY, |a, &b| a.min(b));
    let diff2 = state2
        .probs
        .iter()
        .take(3)
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - state2
            .probs
            .iter()
            .take(3)
            .fold(f32::INFINITY, |a, &b| a.min(b));

    assert!(diff2 < diff1);
}
