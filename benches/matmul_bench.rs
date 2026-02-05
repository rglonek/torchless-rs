//! Matrix multiplication benchmarks
//!
//! Compares different matmul implementations:
//! - Pure Rust (ndarray without BLAS)
//! - BLAS-accelerated (when feature enabled)
//! - Parallel (rayon, when feature enabled)
//!
//! Run with: cargo bench --bench matmul_bench
//! With BLAS: cargo bench --bench matmul_bench --features openblas
//! With parallel: cargo bench --bench matmul_bench --features parallel

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};

/// Pure Rust matmul: W @ x
/// W: (n, d), x: (d,) -> out: (n,)
fn matmul_vec_pure(w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
    w.dot(x)
}

/// Manual row-by-row matmul (baseline, no BLAS)
fn matmul_vec_manual(w: &Array2<f32>, x: &Array1<f32>) -> Array1<f32> {
    let (n, d) = w.dim();
    let x_slice = x.as_slice().expect("x must be contiguous");
    let mut result = vec![0.0f32; n];

    for (i, row) in w.outer_iter().enumerate() {
        let row_slice = row.as_slice().expect("row must be contiguous");
        let mut sum = 0.0f32;
        for j in 0..d {
            sum += row_slice[j] * x_slice[j];
        }
        result[i] = sum;
    }

    Array1::from_vec(result)
}

/// Matmul with pre-allocated output buffer
fn matmul_vec_into_pure(w: &Array2<f32>, x: &Array1<f32>, out: &mut Array1<f32>) {
    let result = w.dot(x);
    out.assign(&result);
}

/// Create test matrices of a given size
fn create_test_matrices(n: usize, d: usize) -> (Array2<f32>, Array1<f32>) {
    // Use deterministic values for reproducibility
    let w_data: Vec<f32> = (0..n * d).map(|i| ((i % 1000) as f32 - 500.0) / 1000.0).collect();
    let x_data: Vec<f32> = (0..d).map(|i| ((i % 1000) as f32 - 500.0) / 1000.0).collect();

    let w = Array2::from_shape_vec((n, d), w_data).unwrap();
    let x = Array1::from_vec(x_data);

    (w, x)
}

fn bench_matmul_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_vec");

    // Test various sizes representative of LLM inference:
    // Small: embedding lookups, attention scores
    // Medium: attention projection (hidden_size x hidden_size)
    // Large: LM head (vocab_size x hidden_size)
    let sizes = [
        (128, 128),      // Small
        (512, 512),      // Medium-small
        (1024, 1024),    // Medium
        (4096, 4096),    // Large (typical hidden_size)
        (32000, 4096),   // Very large (vocab_size x hidden_size for LM head)
    ];

    for (n, d) in sizes {
        let (w, x) = create_test_matrices(n, d);
        let ops = n * d; // Number of multiply-add operations
        group.throughput(Throughput::Elements(ops as u64));

        // Pure ndarray (uses BLAS when feature enabled)
        group.bench_with_input(
            BenchmarkId::new("ndarray_dot", format!("{}x{}", n, d)),
            &(&w, &x),
            |b, (w, x)| {
                b.iter(|| black_box(matmul_vec_pure(w, x)))
            },
        );

        // Manual row-by-row (baseline, never uses BLAS)
        group.bench_with_input(
            BenchmarkId::new("manual", format!("{}x{}", n, d)),
            &(&w, &x),
            |b, (w, x)| {
                b.iter(|| black_box(matmul_vec_manual(w, x)))
            },
        );

        // With pre-allocated output
        group.bench_with_input(
            BenchmarkId::new("ndarray_into", format!("{}x{}", n, d)),
            &(&w, &x),
            |b, (w, x)| {
                let mut out_buf = Array1::zeros(w.nrows());
                b.iter(|| {
                    matmul_vec_into_pure(w, x, &mut out_buf);
                    black_box(out_buf[0])
                })
            },
        );
    }

    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_matmul_parallel(c: &mut Criterion) {
    use torchless::kernels::matmul_vec_parallel;
    use torchless::kernels::matmul_vec_into_parallel;

    let mut group = c.benchmark_group("matmul_parallel");

    let sizes = [
        (1024, 1024),
        (4096, 4096),
        (32000, 4096),
    ];

    for (n, d) in sizes {
        let (w, x) = create_test_matrices(n, d);
        let ops = n * d;
        group.throughput(Throughput::Elements(ops as u64));

        // Serial (baseline)
        group.bench_with_input(
            BenchmarkId::new("serial", format!("{}x{}", n, d)),
            &(&w, &x),
            |b, (w, x)| {
                b.iter(|| black_box(matmul_vec_pure(w, x)))
            },
        );

        // Parallel
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}x{}", n, d)),
            &(&w, &x),
            |b, (w, x)| {
                b.iter(|| black_box(matmul_vec_parallel(w, x)))
            },
        );

        // Parallel with pre-allocated output
        group.bench_with_input(
            BenchmarkId::new("parallel_into", format!("{}x{}", n, d)),
            &(&w, &x),
            |b, (w, x)| {
                let mut out_buf = Array1::zeros(w.nrows());
                b.iter(|| {
                    matmul_vec_into_parallel(w, x, &mut out_buf);
                    black_box(out_buf[0])
                })
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "parallel"))]
fn bench_matmul_parallel(_c: &mut Criterion) {
    // No-op when parallel feature is disabled
}

fn bench_matmul_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_batch");

    // Simulate batch processing: multiple matmuls in sequence
    // This is representative of processing multiple tokens or heads
    let (n, d) = (4096, 4096);
    let (w, _x) = create_test_matrices(n, d);
    let batch_sizes = [1, 4, 8, 16];

    for batch in batch_sizes {
        let inputs: Vec<Array1<f32>> = (0..batch)
            .map(|i| {
                Array1::from_vec(
                    (0..d).map(|j| ((j + i * 100) % 1000) as f32 / 1000.0).collect()
                )
            })
            .collect();

        let ops = n * d * batch;
        group.throughput(Throughput::Elements(ops as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential", batch),
            &(&w, &inputs),
            |b, (w, inputs)| {
                b.iter(|| {
                    let results: Vec<_> = inputs.iter().map(|x| matmul_vec_pure(w, x)).collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_sizes,
    bench_matmul_parallel,
    bench_matmul_batch,
);
criterion_main!(benches);
