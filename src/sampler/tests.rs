use super::{
    sample_greedy, sample_multinomial, sample_top_k, sample_top_p, sample_with_config,
    SamplingConfig,
};
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
        n_routed_experts: 0,
        n_experts_per_token: 0,
        n_shared_experts: 0,
        moe_intermediate_size: 0,
        first_moe_layer: 0,
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

// =============================================================================
// SamplingConfig Tests
// =============================================================================

#[test]
fn test_sampling_config_greedy() {
    let config = test_config();
    let mut state = InferenceState::new(config);

    state.logits[0] = 0.1;
    state.logits[1] = 0.5;
    state.logits[2] = 0.9;
    state.logits[3] = 0.3;

    let token = sample_with_config(&mut state, &SamplingConfig::greedy());
    assert_eq!(token, 2);
}

#[test]
fn test_sampling_config_with_temperature() {
    let config = test_config();
    let mut state = InferenceState::new(config.clone());

    state.logits[0] = 1.0;
    state.logits[1] = 2.0;
    state.logits[2] = 3.0;

    let token = sample_with_config(&mut state, &SamplingConfig::with_temperature(1.0));
    assert!(token < config.vocab_size as u32);

    // Probs should sum to 1
    let sum: f32 = state.probs.sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_sampling_config_default() {
    let cfg = SamplingConfig::default();
    assert_eq!(cfg.temperature, 1.0);
    assert!(cfg.top_k.is_none());
    assert!(cfg.top_p.is_none());
}

// =============================================================================
// Top-K Sampling Tests
// =============================================================================

#[test]
fn test_top_k_filters_to_k_tokens() {
    let config = test_config();
    let mut state = InferenceState::new(config);

    // Set logits with distinct values so top-k filtering is clear
    state.logits[0] = 5.0;
    state.logits[1] = 4.0;
    state.logits[2] = 3.0;
    state.logits[3] = 2.0;
    state.logits[4] = 1.0;
    // rest are 0.0

    // With top_k=3, only tokens 0, 1, 2 should have nonzero probability
    let token = sample_top_k(&mut state, 1.0, 3);
    assert!(
        token <= 2,
        "top_k=3 should only produce tokens 0..2, got {}",
        token
    );
}

#[test]
fn test_top_k_greedy_with_zero_temp() {
    let config = test_config();
    let mut state = InferenceState::new(config);

    state.logits[0] = 1.0;
    state.logits[1] = 5.0;
    state.logits[2] = 3.0;

    // top_k with temperature=0 should still be greedy
    let token = sample_with_config(
        &mut state,
        &SamplingConfig {
            temperature: 0.0,
            top_k: Some(2),
            top_p: None,
        },
    );
    assert_eq!(token, 1);
}

#[test]
fn test_top_k_equals_vocab_is_noop() {
    let config = test_config();
    let mut state1 = InferenceState::new(config.clone());
    let mut state2 = InferenceState::new(config.clone());

    for i in 0..5 {
        state1.logits[i] = (i + 1) as f32;
        state2.logits[i] = (i + 1) as f32;
    }

    // top_k >= vocab_size should behave like no top-k
    sample_with_config(
        &mut state1,
        &SamplingConfig {
            temperature: 1.0,
            top_k: Some(config.vocab_size),
            top_p: None,
        },
    );
    sample_with_config(&mut state2, &SamplingConfig::with_temperature(1.0));

    // Probability distributions should be identical
    for (a, b) in state1.probs.iter().zip(state2.probs.iter()) {
        assert!((a - b).abs() < 1e-6, "probs differ: {} vs {}", a, b);
    }
}

// =============================================================================
// Top-P (Nucleus) Sampling Tests
// =============================================================================

#[test]
fn test_top_p_filters_low_prob_tokens() {
    let config = test_config();
    let mut state = InferenceState::new(config);

    // Create a distribution where one token dominates
    state.logits[0] = 10.0; // will have very high probability after softmax
    state.logits[1] = 1.0;
    state.logits[2] = 0.5;
    state.logits[3] = 0.1;
    // rest are 0.0

    // With a tight top_p, only the dominant token(s) should survive
    let token = sample_top_p(&mut state, 1.0, 0.5);

    // Token 0 dominates, so with p=0.5 it should almost always be token 0
    assert_eq!(
        token, 0,
        "top_p=0.5 with dominant token 0 should select token 0"
    );
}

#[test]
fn test_top_p_one_is_noop() {
    let config = test_config();
    let mut state1 = InferenceState::new(config.clone());
    let mut state2 = InferenceState::new(config);

    for i in 0..5 {
        state1.logits[i] = (i + 1) as f32;
        state2.logits[i] = (i + 1) as f32;
    }

    // top_p=1.0 should not filter anything (>= 1.0 is a no-op in the code)
    sample_with_config(
        &mut state1,
        &SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: Some(1.0),
        },
    );
    sample_with_config(&mut state2, &SamplingConfig::with_temperature(1.0));

    for (a, b) in state1.probs.iter().zip(state2.probs.iter()) {
        assert!((a - b).abs() < 1e-6, "probs differ: {} vs {}", a, b);
    }
}

#[test]
fn test_top_p_renormalizes() {
    let config = test_config();
    let mut state = InferenceState::new(config);

    state.logits[0] = 5.0;
    state.logits[1] = 4.0;
    state.logits[2] = 1.0;
    state.logits[3] = 0.5;

    sample_with_config(
        &mut state,
        &SamplingConfig {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.9),
        },
    );

    // After top-p filtering and renormalization, probs should still sum to ~1
    let sum: f32 = state.probs.sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "probs should sum to 1 after top-p renorm, got {}",
        sum
    );
}

// =============================================================================
// Combined Top-K + Top-P Tests
// =============================================================================

#[test]
fn test_combined_top_k_top_p() {
    let config = test_config();
    let mut state = InferenceState::new(config);

    state.logits[0] = 10.0;
    state.logits[1] = 8.0;
    state.logits[2] = 6.0;
    state.logits[3] = 4.0;
    state.logits[4] = 2.0;

    let token = sample_with_config(
        &mut state,
        &SamplingConfig {
            temperature: 1.0,
            top_k: Some(3),   // keep tokens 0, 1, 2
            top_p: Some(0.8), // then further filter by cumulative prob
        },
    );

    // Should only produce one of the top-3 tokens
    assert!(
        token <= 2,
        "combined top_k=3 + top_p=0.8 should only produce tokens 0..2, got {}",
        token
    );

    // Probs should sum to ~1 after both filters
    let sum: f32 = state.probs.sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "probs should sum to 1 after combined filtering, got {}",
        sum
    );
}
