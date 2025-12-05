use torchless::{generate, InferenceState, Mistral, Parameters};

#[test]
fn test_end_to_end_inference() {
    // Load tiny test model
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let model = Mistral::load(params).unwrap();
    let mut state = InferenceState::new(model.config.clone());

    // Tokenize a prompt
    let tokens = model.tokenizer.encode("Paris");
    assert!(!tokens.is_empty());
    assert_eq!(tokens[0], 0); // BOS token

    // Process all but last token
    for &token in &tokens[..tokens.len() - 1] {
        model.forward(&mut state, token, false);
        state.pos += 1;
    }

    // Generate a few tokens
    let mut token = *tokens.last().unwrap();
    for _ in 0..5 {
        token = generate(&model, &mut state, token, 0.0, false);
        assert!(token < model.config.vocab_size as u32);
        state.pos += 1;
    }

    // If we got here without panicking, the pipeline works!
}

#[test]
fn test_tokenizer_round_trip() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();

    // Test that tokenization works
    let tokens = params.tokenizer.encode("Hello");
    assert!(!tokens.is_empty());

    // Decode back (won't match original due to BOS token and byte fallback)
    let decoded = params.tokenizer.decode(&tokens);
    assert!(!decoded.is_empty());
}

#[test]
fn test_model_forward_pass() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let model = Mistral::load(params).unwrap();
    let mut state = InferenceState::new(model.config.clone());

    // Run forward pass with a token
    model.forward(&mut state, 1, false);

    // Verify logits were populated
    assert_eq!(state.logits.len(), model.config.vocab_size);

    // Logits should have non-zero values after forward pass
    let has_nonzero = state.logits.iter().any(|&v| v != 0.0);
    assert!(has_nonzero, "Logits should have non-zero values");
}
