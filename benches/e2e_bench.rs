//! End-to-end inference benchmarks
//!
//! Profiles full forward pass timing using the test model.
//! Measures the entire inference pipeline including:
//! - Embedding lookup
//! - All transformer layers
//! - Final norm and LM head projection
//!
//! Run with: cargo bench --bench e2e_bench
//! With all optimizations: cargo bench --bench e2e_bench --features "simd,parallel"

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use torchless::{InferenceState, Mistral, Parameters};

#[cfg(feature = "parallel")]
use torchless::LazyMistral;

/// Load the test model once and reuse across benchmarks
fn load_test_model() -> Mistral {
    let params = Parameters::load("tests/fixtures/test_model.bin")
        .expect("Failed to load test model. Run `python tests/generate_test_model.py` first.");
    Mistral::load(params).expect("Failed to initialize model")
}

fn bench_forward_pass(c: &mut Criterion) {
    let model = load_test_model();
    let config = model.config.clone();

    let mut group = c.benchmark_group("forward_pass");

    // Single token forward pass
    group.bench_function("single_token", |b| {
        let mut state = InferenceState::new(config.clone());
        let token = 1u32; // Use a simple token ID

        b.iter(|| {
            // Reset state for each iteration
            state.pos = 0;
            state.hidden_state.fill(0.0);

            model.forward(&mut state, token, false);
            black_box(state.logits[0])
        })
    });

    // Fast forward pass (with SIMD/parallel when enabled)
    group.bench_function("single_token_fast", |b| {
        let mut state = InferenceState::new(config.clone());
        let token = 1u32;

        b.iter(|| {
            state.pos = 0;
            state.hidden_state.fill(0.0);

            model.fast_forward(&mut state, token, false);
            black_box(state.logits[0])
        })
    });

    // Multiple positions (simulating sequence processing)
    for n_positions in [1, 5, 10] {
        let config_clone = config.clone();
        group.bench_with_input(
            BenchmarkId::new("sequence", n_positions),
            &n_positions,
            |b, &n| {
                let mut state = InferenceState::new(config_clone.clone());
                let tokens: Vec<u32> = (0..n).map(|i| (i + 1) as u32).collect();

                b.iter(|| {
                    state.pos = 0;
                    state.hidden_state.fill(0.0);

                    for &token in &tokens {
                        model.forward(&mut state, token, false);
                        state.pos += 1;
                    }
                    black_box(state.logits[0])
                })
            },
        );
    }

    group.finish();
}

fn bench_model_components(c: &mut Criterion) {
    let model = load_test_model();
    let config = model.config.clone();

    let mut group = c.benchmark_group("model_components");

    // Embedding lookup only
    group.bench_function("embedding", |b| {
        let mut state = InferenceState::new(config.clone());
        let token = 1u32;

        b.iter(|| {
            model.embedding.forward(&mut state, token);
            black_box(state.hidden_state[0])
        })
    });

    // Single layer only
    group.bench_function("single_layer", |b| {
        let mut state = InferenceState::new(config.clone());

        // Initialize hidden state with some values
        let hidden_len = state.hidden_state.len();
        for (i, v) in state.hidden_state.iter_mut().enumerate() {
            *v = (i as f32 / hidden_len as f32) - 0.5;
        }

        b.iter(|| {
            // Reset residual
            state.residual.assign(&state.hidden_state);

            model.layers[0].forward(&mut state, 0, false);
            black_box(state.hidden_state[0])
        })
    });

    // Single layer fast path
    group.bench_function("single_layer_fast", |b| {
        let mut state = InferenceState::new(config.clone());

        let hidden_len = state.hidden_state.len();
        for (i, v) in state.hidden_state.iter_mut().enumerate() {
            *v = (i as f32 / hidden_len as f32) - 0.5;
        }

        b.iter(|| {
            state.residual.assign(&state.hidden_state);

            model.layers[0].fast_forward(&mut state, 0, false);
            black_box(state.hidden_state[0])
        })
    });

    // Final norm only
    group.bench_function("final_norm", |b| {
        let mut state = InferenceState::new(config.clone());

        let hidden_len = state.hidden_state.len();
        for (i, v) in state.hidden_state.iter_mut().enumerate() {
            *v = (i as f32 / hidden_len as f32) - 0.5;
        }

        b.iter(|| {
            let mut hidden = state.hidden_state.clone();
            // Manually apply norm (model.norm.forward takes state by mut ref)
            let squares: f32 = hidden.iter().map(|v| v * v).sum();
            let rms = (squares / hidden.len() as f32 + config.norm_eps).sqrt();
            hidden.mapv_inplace(|v| v / rms);
            black_box(hidden[0])
        })
    });

    // LM head projection only
    group.bench_function("lm_head", |b| {
        let mut state = InferenceState::new(config.clone());

        let hidden_len = state.hidden_state.len();
        for (i, v) in state.hidden_state.iter_mut().enumerate() {
            *v = (i as f32 / hidden_len as f32) - 0.5;
        }

        b.iter(|| {
            state.logits.assign(&torchless::kernels::matmul_vec(&model.lm_head, &state.hidden_state));
            black_box(state.logits[0])
        })
    });

    group.finish();
}

fn bench_inference_state_allocation(c: &mut Criterion) {
    let model = load_test_model();
    let config = model.config.clone();

    let mut group = c.benchmark_group("inference_state");

    // Measure state creation time
    group.bench_function("create_state", |b| {
        b.iter(|| {
            let state = InferenceState::new(config.clone());
            black_box(state.pos)
        })
    });

    // Measure state reset overhead (simulating batch processing)
    group.bench_function("reset_state", |b| {
        let mut state = InferenceState::new(config.clone());

        b.iter(|| {
            state.pos = 0;
            state.hidden_state.fill(0.0);
            state.residual.fill(0.0);
            state.logits.fill(0.0);
            state.probs.fill(0.0);
            // Note: We don't reset KV cache here as that would be done selectively
            black_box(state.pos)
        })
    });

    group.finish();
}

fn bench_generation_loop(c: &mut Criterion) {
    let model = load_test_model();
    let config = model.config.clone();

    let mut group = c.benchmark_group("generation");

    // Simulate token generation (forward + sampling)
    for n_tokens in [1, 5, 10] {
        let config_clone = config.clone();
        group.throughput(Throughput::Elements(n_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("greedy", n_tokens),
            &n_tokens,
            |b, &n| {
                let mut state = InferenceState::new(config_clone.clone());
                let initial_token = 1u32;

                b.iter(|| {
                    state.pos = 0;
                    state.hidden_state.fill(0.0);

                    let mut token = initial_token;
                    for _ in 0..n {
                        model.forward(&mut state, token, false);
                        token = torchless::sample_greedy(&state);
                        state.pos += 1;
                    }
                    black_box(token)
                })
            },
        );

        let config_clone2 = config.clone();
        group.bench_with_input(
            BenchmarkId::new("greedy_fast", n_tokens),
            &n_tokens,
            |b, &n| {
                let mut state = InferenceState::new(config_clone2.clone());
                let initial_token = 1u32;

                b.iter(|| {
                    state.pos = 0;
                    state.hidden_state.fill(0.0);

                    let mut token = initial_token;
                    for _ in 0..n {
                        model.fast_forward(&mut state, token, false);
                        token = torchless::sample_greedy(&state);
                        state.pos += 1;
                    }
                    black_box(token)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Lazy model benchmarks (when applicable)
// =============================================================================

#[cfg(feature = "parallel")]
fn bench_lazy_model(c: &mut Criterion) {
    let params = Parameters::load("tests/fixtures/test_model.bin")
        .expect("Failed to load test model");
    let lazy_model = LazyMistral::load(&params)
        .expect("Failed to initialize lazy model");
    let config = lazy_model.config.clone();

    let mut group = c.benchmark_group("lazy_model");

    // Compare eager vs lazy forward pass
    group.bench_function("lazy_forward", |b| {
        let mut state = InferenceState::new(config.clone());
        let token = 1u32;

        b.iter(|| {
            state.pos = 0;
            state.hidden_state.fill(0.0);

            lazy_model.forward(&mut state, token, false);
            black_box(state.logits[0])
        })
    });

    group.bench_function("lazy_forward_fast", |b| {
        let mut state = InferenceState::new(config.clone());
        let token = 1u32;

        b.iter(|| {
            state.pos = 0;
            state.hidden_state.fill(0.0);

            lazy_model.fast_forward(&mut state, token, false);
            black_box(state.logits[0])
        })
    });

    group.finish();
}

#[cfg(not(feature = "parallel"))]
fn bench_lazy_model(_c: &mut Criterion) {
    // No-op when parallel feature is disabled
    // (LazyMistral uses parallel features internally)
}

criterion_group!(
    benches,
    bench_forward_pass,
    bench_model_components,
    bench_inference_state_allocation,
    bench_generation_loop,
    bench_lazy_model,
);
criterion_main!(benches);
