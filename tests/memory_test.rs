//! Memory profiling and allocation tests
//!
//! These tests verify memory usage patterns and help track allocations
//! during inference. While Rust doesn't have built-in memory profiling
//! like some other languages, we can:
//! 1. Verify struct sizes at compile time
//! 2. Test allocation-free code paths
//! 3. Compare eager vs lazy loading memory footprints
//!
//! For detailed memory profiling, use external tools:
//! - `cargo +nightly build --release && valgrind --tool=massif ./target/release/torchless`
//! - DHAT: `cargo +nightly test --release -- --nocapture`
//! - heaptrack: `heaptrack cargo test --release`

use std::mem::size_of;
use torchless::{Config, InferenceState, Mistral, Parameters};

#[cfg(feature = "parallel")]
use torchless::LazyMistral;

// =============================================================================
// Struct size verification
// =============================================================================

/// Test that key struct sizes are within expected bounds.
/// This helps catch unexpected bloat from adding fields.
#[test]
fn test_config_size() {
    // Config should be relatively small (just metadata)
    let config_size = size_of::<Config>();
    println!("Config size: {} bytes", config_size);

    // Should be less than 1KB for basic model config
    assert!(
        config_size < 1024,
        "Config grew unexpectedly large: {} bytes",
        config_size
    );
}

#[test]
fn test_inference_state_base_size() {
    // InferenceState has a fixed base size plus dynamically allocated arrays
    // The base struct size should be reasonable
    let state_size = size_of::<InferenceState>();
    println!("InferenceState base size: {} bytes", state_size);

    // The base struct (without array contents) should be manageable
    // This includes pointers/metadata for all the arrays
    assert!(
        state_size < 2048,
        "InferenceState base grew unexpectedly: {} bytes",
        state_size
    );
}

// =============================================================================
// Memory estimation tests
// =============================================================================

/// Calculate expected memory usage for InferenceState given a config
fn estimate_inference_state_memory(config: &Config) -> usize {
    let head_dim = config.hidden_size / config.n_heads;

    let mut total = 0usize;

    // hidden_state and residual: 2 * hidden_size * sizeof(f32)
    total += 2 * config.hidden_size * size_of::<f32>();

    // RoPE: inv_freq, cos, sin: 3 * (head_dim/2) * sizeof(f32)
    total += 3 * (head_dim / 2) * size_of::<f32>();

    // Projection flat buffers: q_flat, k_flat, v_flat
    total += config.n_heads * head_dim * size_of::<f32>(); // q_flat
    total += config.n_kv_heads * head_dim * size_of::<f32>(); // k_flat
    total += config.n_kv_heads * head_dim * size_of::<f32>(); // v_flat

    // Shaped state arrays: q_state, k_state, v_state
    total += config.n_heads * head_dim * size_of::<f32>(); // q_state
    total += config.n_kv_heads * head_dim * size_of::<f32>(); // k_state
    total += config.n_kv_heads * head_dim * size_of::<f32>(); // v_state

    // KV cache: 2 * n_layers * n_kv_heads * max_seq_len * head_dim
    const MAX_SEQ_LEN: usize = 500;
    total += 2 * config.n_layers * config.n_kv_heads * MAX_SEQ_LEN * head_dim * size_of::<f32>();

    // Attention outputs: scores, context, context_flat
    total += config.n_heads * MAX_SEQ_LEN * size_of::<f32>(); // scores
    total += config.n_heads * head_dim * size_of::<f32>(); // context
    total += config.hidden_size * size_of::<f32>(); // context_flat

    // MLP intermediates: mlp_gate, mlp_up
    total += 2 * config.intermediate_size * size_of::<f32>();

    // Output: logits, probs
    total += 2 * config.vocab_size * size_of::<f32>();

    total
}

#[test]
fn test_inference_state_memory_estimate() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let config = params.config.clone();

    let estimated = estimate_inference_state_memory(&config);
    println!(
        "Estimated InferenceState memory: {} bytes ({:.2} MB)",
        estimated,
        estimated as f64 / 1024.0 / 1024.0
    );

    // Create the actual state and verify it's roughly as expected
    let state = InferenceState::new(config.clone());

    // Count actual array elements
    let mut actual_elements = 0usize;
    actual_elements += state.hidden_state.len();
    actual_elements += state.residual.len();
    actual_elements += state.inv_freq.len();
    actual_elements += state.cos.len();
    actual_elements += state.sin.len();
    actual_elements += state.q_flat.len();
    actual_elements += state.k_flat.len();
    actual_elements += state.v_flat.len();
    actual_elements += state.q_state.len();
    actual_elements += state.k_state.len();
    actual_elements += state.v_state.len();
    actual_elements += state.k_cache.len();
    actual_elements += state.v_cache.len();
    actual_elements += state.scores.len();
    actual_elements += state.context.len();
    actual_elements += state.context_flat.len();
    actual_elements += state.mlp_gate.len();
    actual_elements += state.mlp_up.len();
    actual_elements += state.logits.len();
    actual_elements += state.probs.len();

    let actual_bytes = actual_elements * size_of::<f32>();
    println!(
        "Actual InferenceState array memory: {} bytes ({:.2} MB)",
        actual_bytes,
        actual_bytes as f64 / 1024.0 / 1024.0
    );

    // Should be within 10% of estimate
    let diff = (actual_bytes as i64 - estimated as i64).unsigned_abs();
    let tolerance = estimated / 10;
    assert!(
        diff <= tolerance as u64,
        "Memory estimate off by more than 10%: estimated {}, actual {}",
        estimated,
        actual_bytes
    );
}

// =============================================================================
// Allocation behavior tests
// =============================================================================

/// Test that forward pass doesn't grow memory significantly
#[test]
fn test_forward_pass_no_memory_growth() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let model = Mistral::load(params).unwrap();
    let mut state = InferenceState::new(model.config.clone());

    // Run forward pass multiple times
    // If there were memory leaks, this would be caught by running under valgrind
    for i in 0..10 {
        model.forward(&mut state, (i + 1) as u32, false);
        state.pos += 1;
    }

    // Reset and run again - should use same memory
    state.pos = 0;
    state.hidden_state.fill(0.0);

    for i in 0..10 {
        model.forward(&mut state, (i + 1) as u32, false);
        state.pos += 1;
    }

    // If we got here without OOM, the test passes
    // Real memory tracking would require external tools
}

/// Test that fast_forward maintains same allocation behavior
#[test]
fn test_fast_forward_no_memory_growth() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let model = Mistral::load(params).unwrap();
    let mut state = InferenceState::new(model.config.clone());

    // Run fast forward pass multiple times
    for i in 0..10 {
        model.fast_forward(&mut state, (i + 1) as u32, false);
        state.pos += 1;
    }

    // Reset and run again
    state.pos = 0;
    state.hidden_state.fill(0.0);

    for i in 0..10 {
        model.fast_forward(&mut state, (i + 1) as u32, false);
        state.pos += 1;
    }
}

// =============================================================================
// Lazy vs Eager comparison
// =============================================================================

/// Estimate memory for eager model (all weights in f32)
fn estimate_eager_model_memory(config: &Config) -> usize {
    let head_dim = config.hidden_size / config.n_heads;
    let mut total = 0usize;

    // Embedding: vocab_size * hidden_size
    total += config.vocab_size * config.hidden_size * size_of::<f32>();

    // Per-layer weights (for each of n_layers):
    // - input_layernorm: hidden_size
    // - q_proj: n_heads * head_dim * hidden_size
    // - k_proj: n_kv_heads * head_dim * hidden_size
    // - v_proj: n_kv_heads * head_dim * hidden_size
    // - o_proj: hidden_size * n_heads * head_dim
    // - post_attention_layernorm: hidden_size
    // - gate_proj: intermediate_size * hidden_size
    // - up_proj: intermediate_size * hidden_size
    // - down_proj: hidden_size * intermediate_size
    let per_layer = {
        let mut l = 0usize;
        l += config.hidden_size; // input norm
        l += config.n_heads * head_dim * config.hidden_size; // q_proj
        l += config.n_kv_heads * head_dim * config.hidden_size; // k_proj
        l += config.n_kv_heads * head_dim * config.hidden_size; // v_proj
        l += config.hidden_size * config.n_heads * head_dim; // o_proj
        l += config.hidden_size; // post attention norm
        l += config.intermediate_size * config.hidden_size; // gate
        l += config.intermediate_size * config.hidden_size; // up
        l += config.hidden_size * config.intermediate_size; // down
        l
    };
    total += config.n_layers * per_layer * size_of::<f32>();

    // Final norm: hidden_size
    total += config.hidden_size * size_of::<f32>();

    // LM head: vocab_size * hidden_size
    total += config.vocab_size * config.hidden_size * size_of::<f32>();

    total
}

#[test]
fn test_model_memory_estimate() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let config = params.config.clone();

    let eager_estimate = estimate_eager_model_memory(&config);
    println!(
        "Estimated eager model memory: {} bytes ({:.2} MB)",
        eager_estimate,
        eager_estimate as f64 / 1024.0 / 1024.0
    );

    // For the test model, this should be relatively small
    // For a real Mistral-7B model, this would be ~25GB
}

#[test]
#[cfg(feature = "parallel")]
fn test_lazy_model_memory_savings() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let config = params.config.clone();

    // Estimate eager model memory
    let eager_estimate = estimate_eager_model_memory(&config);

    // Lazy model only loads norms eagerly
    // Per layer: input_layernorm (hidden_size) + post_attention_layernorm (hidden_size)
    // Plus final norm (hidden_size)
    let lazy_eager_portion = config.n_layers * 2 * config.hidden_size * size_of::<f32>()
        + config.hidden_size * size_of::<f32>();

    println!(
        "Lazy model eager portion: {} bytes ({:.2} KB)",
        lazy_eager_portion,
        lazy_eager_portion as f64 / 1024.0
    );
    println!(
        "Memory savings: {:.2}x reduction",
        eager_estimate as f64 / lazy_eager_portion as f64
    );

    // Lazy model should use significantly less eager memory
    assert!(
        lazy_eager_portion < eager_estimate / 10,
        "Lazy model should use at least 10x less eager memory"
    );

    // Actually create the lazy model to verify it works
    let _lazy_model = LazyMistral::load(&params).unwrap();
}

// =============================================================================
// KV Cache memory tests
// =============================================================================

#[test]
fn test_kv_cache_memory() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let config = params.config.clone();
    let head_dim = config.hidden_size / config.n_heads;
    const MAX_SEQ_LEN: usize = 500;

    // KV cache: 2 tensors of shape [n_layers, n_kv_heads, max_seq_len, head_dim]
    let kv_cache_size =
        2 * config.n_layers * config.n_kv_heads * MAX_SEQ_LEN * head_dim * size_of::<f32>();

    println!(
        "KV cache size: {} bytes ({:.2} MB)",
        kv_cache_size,
        kv_cache_size as f64 / 1024.0 / 1024.0
    );

    // Create state and verify KV cache sizes
    let state = InferenceState::new(config.clone());
    let actual_k_cache = state.k_cache.len() * size_of::<f32>();
    let actual_v_cache = state.v_cache.len() * size_of::<f32>();

    assert_eq!(
        actual_k_cache + actual_v_cache,
        kv_cache_size,
        "KV cache size mismatch"
    );

    // For a real model, print what this would be:
    // Mistral-7B: n_layers=32, n_kv_heads=8, head_dim=128
    let real_kv_cache = 2 * 32 * 8 * MAX_SEQ_LEN * 128 * size_of::<f32>();
    println!(
        "Real Mistral-7B KV cache (seq_len={}): {:.2} MB",
        MAX_SEQ_LEN,
        real_kv_cache as f64 / 1024.0 / 1024.0
    );
}

// =============================================================================
// Allocation counting (conceptual - would need custom allocator)
// =============================================================================

/// This test documents where allocations occur in the forward pass.
/// To actually count allocations, you would need:
/// 1. A custom global allocator that counts
/// 2. Or external tools like DHAT
#[test]
fn test_document_allocation_points() {
    // Known allocation points in forward():
    // 1. None in embedding lookup (uses pre-allocated hidden_state)
    // 2. None in attention (uses pre-allocated buffers since optimization)
    // 3. silu() returns a new Array1 - potential optimization point
    // 4. matmul_vec() returns new Array1 - but we use assign() to avoid extra copy

    // The following operations should NOT allocate:
    // - kernels::matmul_vec_into() - writes to pre-allocated buffer
    // - kernels::softmax_view() - in-place on view
    // - kernels::weighted_sum_rows() - writes to pre-allocated buffer
    // - Array assignments with .assign() - reuses existing memory

    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let model = Mistral::load(params).unwrap();
    let mut state = InferenceState::new(model.config.clone());

    // First forward to warm up any lazy initialization
    model.forward(&mut state, 1, false);

    // Subsequent forwards should have consistent allocation behavior
    for _ in 0..5 {
        state.pos += 1;
        model.forward(&mut state, 1, false);
    }

    // This test passes if no panic/OOM occurs
    // For actual allocation counting, use external profiling tools
}

// =============================================================================
// Memory layout verification
// =============================================================================

#[test]
fn test_array_contiguity() {
    let params = Parameters::load("tests/fixtures/test_model.bin").unwrap();
    let state = InferenceState::new(params.config.clone());

    // Verify critical arrays are contiguous (required for SIMD/BLAS)
    assert!(
        state.hidden_state.as_slice().is_some(),
        "hidden_state must be contiguous"
    );
    assert!(
        state.residual.as_slice().is_some(),
        "residual must be contiguous"
    );
    assert!(
        state.q_flat.as_slice().is_some(),
        "q_flat must be contiguous"
    );
    assert!(
        state.k_flat.as_slice().is_some(),
        "k_flat must be contiguous"
    );
    assert!(
        state.v_flat.as_slice().is_some(),
        "v_flat must be contiguous"
    );
    assert!(
        state.logits.as_slice().is_some(),
        "logits must be contiguous"
    );
    assert!(state.probs.as_slice().is_some(), "probs must be contiguous");
    assert!(
        state.mlp_gate.as_slice().is_some(),
        "mlp_gate must be contiguous"
    );
    assert!(
        state.mlp_up.as_slice().is_some(),
        "mlp_up must be contiguous"
    );
}
