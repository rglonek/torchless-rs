//! Attention module benchmarks
//!
//! Profiles attention computation with different optimizations:
//! - Allocation patterns (pre-allocated vs dynamic)
//! - SIMD kernels (when feature enabled)
//! - Parallel attention heads (when feature enabled)
//!
//! Run with: cargo bench --bench attention_bench
//! With SIMD: cargo bench --bench attention_bench --features simd
//! With parallel: cargo bench --bench attention_bench --features parallel

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{s, Array1, Array2, ArrayViewMut1};

// =============================================================================
// Attention score computation benchmarks
// =============================================================================

/// Compute attention scores: scores[i] = keys[i, :].dot(query) * scale
/// This is the core computation that happens for each head.
fn compute_attention_scores_allocating(
    query: &Array1<f32>,
    keys: &Array2<f32>,
    scale: f32,
) -> Array1<f32> {
    let seq_len = keys.nrows();
    let mut scores = Array1::zeros(seq_len);

    for i in 0..seq_len {
        let dot: f32 = keys
            .row(i)
            .iter()
            .zip(query.iter())
            .map(|(k, q)| k * q)
            .sum();
        scores[i] = dot * scale;
    }

    scores
}

/// Pre-allocated version: writes directly to output buffer
fn compute_attention_scores_preallocated(
    query: &Array1<f32>,
    keys: &Array2<f32>,
    scores: &mut ArrayViewMut1<f32>,
    scale: f32,
) {
    let seq_len = keys.nrows();
    for i in 0..seq_len {
        let dot: f32 = keys
            .row(i)
            .iter()
            .zip(query.iter())
            .map(|(k, q)| k * q)
            .sum();
        scores[i] = dot * scale;
    }
}

/// Create test data for attention benchmarks
fn create_attention_test_data(
    seq_len: usize,
    head_dim: usize,
) -> (Array1<f32>, Array2<f32>, Array2<f32>) {
    // Query vector
    let query = Array1::from_vec(
        (0..head_dim)
            .map(|i| (i as f32 / head_dim as f32) - 0.5)
            .collect(),
    );

    // Keys and values matrices [seq_len, head_dim]
    let keys_data: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 1000) as f32 - 500.0) / 1000.0)
        .collect();
    let keys = Array2::from_shape_vec((seq_len, head_dim), keys_data.clone()).unwrap();

    let values = Array2::from_shape_vec((seq_len, head_dim), keys_data).unwrap();

    (query, keys, values)
}

fn bench_attention_scores(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_scores");

    // Test various sequence lengths and head dimensions
    let configs = [
        (64, 64),   // Small
        (128, 128), // Medium
        (256, 128), // Longer sequence
        (500, 128), // Max sequence (matches MAX_SEQ_LEN)
    ];

    for (seq_len, head_dim) in configs {
        let (query, keys, _) = create_attention_test_data(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let ops = seq_len * head_dim; // dot products

        group.throughput(Throughput::Elements(ops as u64));

        // Allocating version (baseline)
        group.bench_with_input(
            BenchmarkId::new("allocating", format!("seq{}_dim{}", seq_len, head_dim)),
            &(&query, &keys, scale),
            |b, (q, k, s)| b.iter(|| black_box(compute_attention_scores_allocating(q, k, *s))),
        );

        // Pre-allocated version
        group.bench_with_input(
            BenchmarkId::new("preallocated", format!("seq{}_dim{}", seq_len, head_dim)),
            &(&query, &keys, scale),
            |b, (q, k, s)| {
                let mut scores = Array1::zeros(k.nrows());
                b.iter(|| {
                    let mut view = scores.view_mut();
                    compute_attention_scores_preallocated(q, k, &mut view, *s);
                    black_box(scores[0])
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Softmax benchmarks
// =============================================================================

fn softmax_allocating(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_x = x.mapv(|v| (v - max).exp());
    let sum: f32 = exp_x.sum();
    exp_x.mapv(|v| v / sum)
}

fn softmax_inplace(x: &mut Array1<f32>) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    x.mapv_inplace(|v| (v - max).exp());
    let sum: f32 = x.sum();
    x.mapv_inplace(|v| v / sum);
}

fn softmax_view_inplace(x: &mut ArrayViewMut1<f32>) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    x.mapv_inplace(|v| (v - max).exp());
    let sum: f32 = x.sum();
    x.mapv_inplace(|v| v / sum);
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    let sizes = [64, 128, 256, 500, 1024];

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 / size as f32) - 0.5).collect();
        let x = Array1::from_vec(data);

        group.throughput(Throughput::Elements(size as u64));

        // Allocating version
        group.bench_with_input(BenchmarkId::new("allocating", size), &x, |b, x| {
            b.iter(|| black_box(softmax_allocating(x)))
        });

        // In-place version
        group.bench_with_input(BenchmarkId::new("inplace", size), &x, |b, x| {
            let mut x_copy = x.clone();
            b.iter(|| {
                x_copy.assign(x);
                softmax_inplace(&mut x_copy);
                black_box(x_copy[0])
            })
        });

        // View version (simulates sliced buffer)
        group.bench_with_input(BenchmarkId::new("view_inplace", size), &x, |b, x| {
            let mut x_copy = x.clone();
            b.iter(|| {
                x_copy.assign(x);
                let mut view = x_copy.view_mut();
                softmax_view_inplace(&mut view);
                black_box(x_copy[0])
            })
        });
    }

    group.finish();
}

// =============================================================================
// Weighted sum benchmarks (attention output computation)
// =============================================================================

/// Weighted sum: out[j] = sum_i(weights[i] * matrix[i, j])
fn weighted_sum_allocating(weights: &Array1<f32>, matrix: &Array2<f32>) -> Array1<f32> {
    let d = matrix.ncols();
    let mut out = Array1::zeros(d);

    for (i, w) in weights.iter().enumerate() {
        let row = matrix.row(i);
        for (j, &v) in row.iter().enumerate() {
            out[j] += w * v;
        }
    }

    out
}

fn weighted_sum_preallocated(
    weights: &Array1<f32>,
    matrix: &Array2<f32>,
    out: &mut ArrayViewMut1<f32>,
) {
    out.fill(0.0);
    for (i, w) in weights.iter().enumerate() {
        let row = matrix.row(i);
        for (j, &v) in row.iter().enumerate() {
            out[j] += w * v;
        }
    }
}

fn bench_weighted_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_sum");

    let configs = [(64, 64), (128, 128), (256, 128), (500, 128)];

    for (seq_len, head_dim) in configs {
        // Create normalized weights (like softmax output)
        let weights_raw: Vec<f32> = (0..seq_len).map(|i| (i as f32 + 1.0)).collect();
        let sum: f32 = weights_raw.iter().sum();
        let weights = Array1::from_vec(weights_raw.iter().map(|&w| w / sum).collect());

        // Values matrix
        let values_data: Vec<f32> = (0..seq_len * head_dim)
            .map(|i| ((i % 1000) as f32 - 500.0) / 1000.0)
            .collect();
        let values = Array2::from_shape_vec((seq_len, head_dim), values_data).unwrap();

        let ops = seq_len * head_dim;
        group.throughput(Throughput::Elements(ops as u64));

        // Allocating version
        group.bench_with_input(
            BenchmarkId::new("allocating", format!("seq{}_dim{}", seq_len, head_dim)),
            &(&weights, &values),
            |b, (w, v)| b.iter(|| black_box(weighted_sum_allocating(w, v))),
        );

        // Pre-allocated version
        group.bench_with_input(
            BenchmarkId::new("preallocated", format!("seq{}_dim{}", seq_len, head_dim)),
            &(&weights, &values),
            |b, (w, v)| {
                let mut out = Array1::zeros(v.ncols());
                b.iter(|| {
                    let mut view = out.view_mut();
                    weighted_sum_preallocated(w, v, &mut view);
                    black_box(out[0])
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Full attention head simulation
// =============================================================================

/// Simulate a single attention head computation
fn attention_head_allocating(
    query: &Array1<f32>,
    keys: &Array2<f32>,
    values: &Array2<f32>,
    scale: f32,
) -> Array1<f32> {
    // Compute attention scores
    let mut scores = compute_attention_scores_allocating(query, keys, scale);

    // Softmax
    softmax_inplace(&mut scores);

    // Weighted sum
    weighted_sum_allocating(&scores, values)
}

/// Attention head with pre-allocated buffers
struct AttentionBuffers {
    scores: Array1<f32>,
    output: Array1<f32>,
}

impl AttentionBuffers {
    fn new(max_seq_len: usize, head_dim: usize) -> Self {
        Self {
            scores: Array1::zeros(max_seq_len),
            output: Array1::zeros(head_dim),
        }
    }
}

fn attention_head_preallocated(
    query: &Array1<f32>,
    keys: &Array2<f32>,
    values: &Array2<f32>,
    scale: f32,
    buffers: &mut AttentionBuffers,
) {
    let seq_len = keys.nrows();

    // Compute attention scores into pre-allocated buffer
    {
        let mut scores_view = buffers.scores.slice_mut(s![..seq_len]);
        compute_attention_scores_preallocated(query, keys, &mut scores_view, scale);
        softmax_view_inplace(&mut scores_view);
    }

    // Weighted sum using scores slice
    {
        let scores_view = buffers.scores.slice(s![..seq_len]);
        let mut out_view = buffers.output.view_mut();
        out_view.fill(0.0);
        for (i, &w) in scores_view.iter().enumerate() {
            let row = values.row(i);
            for (j, &v) in row.iter().enumerate() {
                out_view[j] += w * v;
            }
        }
    }
}

fn bench_attention_head(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_head");

    let configs = [(64, 64), (128, 128), (256, 128), (500, 128)];

    for (seq_len, head_dim) in configs {
        let (query, keys, values) = create_attention_test_data(seq_len, head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Total ops: scores (seq_len * head_dim) + softmax (seq_len) + weighted_sum (seq_len * head_dim)
        let ops = 2 * seq_len * head_dim + seq_len;
        group.throughput(Throughput::Elements(ops as u64));

        // Allocating version
        group.bench_with_input(
            BenchmarkId::new("allocating", format!("seq{}_dim{}", seq_len, head_dim)),
            &(&query, &keys, &values, scale),
            |b, (q, k, v, s)| b.iter(|| black_box(attention_head_allocating(q, k, v, *s))),
        );

        // Pre-allocated version
        group.bench_with_input(
            BenchmarkId::new("preallocated", format!("seq{}_dim{}", seq_len, head_dim)),
            &(&query, &keys, &values, scale),
            |b, (q, k, v, s)| {
                let mut buffers = AttentionBuffers::new(500, v.ncols());
                b.iter(|| {
                    attention_head_preallocated(q, k, v, *s, &mut buffers);
                    black_box(buffers.output[0])
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Multi-head attention simulation
// =============================================================================

fn bench_multihead_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("multihead_attention");

    // Simulate multi-head attention with different head counts
    let configs = [
        (8, 128, 64),   // 8 heads, seq_len=128, head_dim=64
        (32, 128, 128), // 32 heads, seq_len=128, head_dim=128 (typical for large models)
    ];

    for (n_heads, seq_len, head_dim) in configs {
        // Create data for all heads
        let queries: Vec<Array1<f32>> = (0..n_heads)
            .map(|h| {
                Array1::from_vec(
                    (0..head_dim)
                        .map(|i| ((i + h * 100) as f32 / head_dim as f32) - 0.5)
                        .collect(),
                )
            })
            .collect();

        let keys_values: Vec<(Array2<f32>, Array2<f32>)> = (0..n_heads)
            .map(|h| {
                let data: Vec<f32> = (0..seq_len * head_dim)
                    .map(|i| ((i + h * 1000) as f32 / 1000.0) - 0.5)
                    .collect();
                let keys = Array2::from_shape_vec((seq_len, head_dim), data.clone()).unwrap();
                let values = Array2::from_shape_vec((seq_len, head_dim), data).unwrap();
                (keys, values)
            })
            .collect();

        let scale = 1.0 / (head_dim as f32).sqrt();
        let ops_per_head = 2 * seq_len * head_dim + seq_len;
        let total_ops = ops_per_head * n_heads;

        group.throughput(Throughput::Elements(total_ops as u64));

        // Sequential heads, allocating
        group.bench_with_input(
            BenchmarkId::new(
                "sequential_allocating",
                format!("{}heads_seq{}_dim{}", n_heads, seq_len, head_dim),
            ),
            &(&queries, &keys_values, scale),
            |b, (qs, kvs, s)| {
                b.iter(|| {
                    let outputs: Vec<_> = qs
                        .iter()
                        .zip(kvs.iter())
                        .map(|(q, (k, v))| attention_head_allocating(q, k, v, *s))
                        .collect();
                    black_box(outputs)
                })
            },
        );

        // Sequential heads, pre-allocated
        group.bench_with_input(
            BenchmarkId::new(
                "sequential_preallocated",
                format!("{}heads_seq{}_dim{}", n_heads, seq_len, head_dim),
            ),
            &(&queries, &keys_values, scale),
            |b, (qs, kvs, s)| {
                let mut all_buffers: Vec<AttentionBuffers> = (0..qs.len())
                    .map(|_| AttentionBuffers::new(500, head_dim))
                    .collect();
                b.iter(|| {
                    for ((q, (k, v)), buf) in qs.iter().zip(kvs.iter()).zip(all_buffers.iter_mut())
                    {
                        attention_head_preallocated(q, k, v, *s, buf);
                    }
                    black_box(all_buffers[0].output[0])
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// SIMD kernel benchmarks (when feature enabled)
// =============================================================================

#[cfg(feature = "simd")]
fn bench_simd_kernels(c: &mut Criterion) {
    use torchless::kernels::{rmsnorm, rmsnorm_simd, silu, silu_simd, softmax, softmax_simd};

    let mut group = c.benchmark_group("simd_kernels");

    let sizes = [128, 512, 1024, 4096];

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32 / size as f32) - 0.5).collect();
        let x = Array1::from_vec(data);
        let weight = Array1::from_vec(
            (0..size)
                .map(|i| 1.0 + (i as f32 / size as f32) * 0.1)
                .collect(),
        );
        let eps = 1e-5;

        group.throughput(Throughput::Elements(size as u64));

        // RMSNorm
        group.bench_with_input(
            BenchmarkId::new("rmsnorm_scalar", size),
            &(&x, &weight, eps),
            |b, (x, w, e)| {
                let mut x_copy = (*x).clone();
                b.iter(|| {
                    x_copy.assign(x);
                    rmsnorm(&mut x_copy, w, *e);
                    black_box(x_copy[0])
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("rmsnorm_simd", size),
            &(&x, &weight, eps),
            |b, (x, w, e)| {
                let mut x_copy = (*x).clone();
                b.iter(|| {
                    x_copy.assign(x);
                    rmsnorm_simd(&mut x_copy, w, *e);
                    black_box(x_copy[0])
                })
            },
        );

        // Softmax
        group.bench_with_input(BenchmarkId::new("softmax_scalar", size), &x, |b, x| {
            let mut x_copy = x.clone();
            b.iter(|| {
                x_copy.assign(x);
                softmax(&mut x_copy);
                black_box(x_copy[0])
            })
        });

        group.bench_with_input(BenchmarkId::new("softmax_simd", size), &x, |b, x| {
            let mut x_copy = x.clone();
            b.iter(|| {
                x_copy.assign(x);
                softmax_simd(&mut x_copy);
                black_box(x_copy[0])
            })
        });

        // SiLU
        group.bench_with_input(BenchmarkId::new("silu_scalar", size), &x, |b, x| {
            b.iter(|| black_box(silu(x)))
        });

        group.bench_with_input(BenchmarkId::new("silu_simd", size), &x, |b, x| {
            b.iter(|| black_box(silu_simd(x)))
        });
    }

    group.finish();
}

#[cfg(not(feature = "simd"))]
fn bench_simd_kernels(_c: &mut Criterion) {
    // No-op when SIMD feature is disabled
}

criterion_group!(
    benches,
    bench_attention_scores,
    bench_softmax,
    bench_weighted_sum,
    bench_attention_head,
    bench_multihead_attention,
    bench_simd_kernels,
);
criterion_main!(benches);
