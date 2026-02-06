# Parallelization

Advanced parallelization techniques for improved CPU utilization and multi-device execution.

| Feature | Impact | Description |
|---------|--------|-------------|
| Better Work Distribution | Reduced load imbalance | Adaptive chunk sizes, work stealing |
| Pipeline Parallelism | Overlapped computation | Multi-token pipeline |
| Tensor Parallelism | Large model support | Split matrices across devices |

## Better Work Distribution

Optimizes Rayon-based parallel execution with adaptive chunk sizes and work-stealing.

### WorkDistributionConfig

```rust
use torchless::WorkDistributionConfig;

// Default configuration
let config = WorkDistributionConfig::default();

// Optimized for matrix multiplication
let config = WorkDistributionConfig::for_matmul(rows, cols);

// Optimized for attention computation
let config = WorkDistributionConfig::for_attention(n_heads, seq_len);
```

Configuration options:
- `min_chunk_size` - Minimum chunk size (default: 64)
- `max_chunk_size` - Maximum chunk size (default: 4096)
- `tasks_per_thread` - Target tasks per thread (default: 4)
- `numa_aware` - Enable NUMA hints (default: false)
- `cache_line_size` - Cache line size (default: 64)

### WorkStealingStats

```rust
use torchless::WorkStealingStats;

let stats = WorkStealingStats::new();
// ... after parallel work ...
println!("Tasks created: {}", stats.tasks_created.load(Ordering::Relaxed));
println!("Steal ratio: {:.1}%", stats.steal_ratio() * 100.0);
```

### Adaptive Parallel Operations

```rust
use torchless::{matmul_vec_adaptive, matmul_vec_adaptive_into};

let config = WorkDistributionConfig::for_matmul(1024, 4096);

// Adaptive matrix-vector multiplication
let result = matmul_vec_adaptive(&weights, &input, &config);

// Zero-allocation version
matmul_vec_adaptive_into(&weights, &input, &mut output, &config);
```

---

## Pipeline Parallelism

Overlapping computation across transformer layers by processing multiple tokens simultaneously.

```
Time ->
Layer 0: [Token 1] [Token 2] [Token 3]
Layer 1:          [Token 1] [Token 2] [Token 3]
Layer 2:                    [Token 1] [Token 2]
```

### PipelineState

```rust
use torchless::PipelineState;

let mut state = PipelineState::new(
    num_stages,  // Number of pipeline stages (typically = n_layers)
    max_depth,   // Maximum micro-batches in flight
);

// Check if pipeline can accept more input
if state.can_accept() {
    state.push();
}

// Check if output is ready
if state.has_output() {
    let completed = state.pop();
}

// Advance pipeline by one step
state.advance();
```

### PipelineConfig

```rust
use torchless::PipelineConfig;

// Default configuration
let config = PipelineConfig::default();

// Compute optimal config for model
let config = PipelineConfig::for_model(n_layers, hidden_size, memory_budget);
```

Configuration options:
- `num_micro_batches` - Micro-batches in pipeline (default: 4)
- `async_execution` - Enable async execution (default: false)
- `memory_budget` - Memory budget for buffers (default: 512MB)

---

## Tensor Parallelism

Splits large matrices across multiple threads/devices for parallel computation.

### TensorParallelConfig

```rust
use torchless::{TensorParallelConfig, TensorParallelStrategy};

// Single device (no parallelism)
let config = TensorParallelConfig::default();

// 2-way tensor parallelism, rank 0
let config = TensorParallelConfig::new(2, 0);

// Compute local size for partitioned dimension
let local_size = config.local_size(hidden_size);  // hidden_size / world_size
let offset = config.local_offset(hidden_size);    // rank * local_size
```

### TensorParallelStrategy

```rust
use torchless::TensorParallelStrategy;

match strategy {
    TensorParallelStrategy::ColumnParallel => {
        // Split output dimension - each device computes partial output
    }
    TensorParallelStrategy::RowParallel => {
        // Split input dimension - needs all-reduce for final result
    }
    TensorParallelStrategy::None => {
        // No parallelism
    }
}
```

### Column-Parallel Linear

Splits output dimension across devices:

```rust
use torchless::column_parallel_linear;

// Each device computes a subset of output features
let local_output = column_parallel_linear(&weights, &input, &config);
// Outputs from all devices are concatenated
```

### Row-Parallel Linear

Splits input dimension across devices:

```rust
use torchless::row_parallel_linear;

// Each device computes a partial result
let partial_output = row_parallel_linear(&weights, &input, &config);
// Partial results need all-reduce to get final output
```

### All-Reduce Operations

```rust
use torchless::{all_reduce_sum, all_reduce_sum_inplace};

// Combine partial results
let partials = vec![partial_0, partial_1, partial_2];
let combined = all_reduce_sum(&partials);

// In-place version
all_reduce_sum_inplace(&mut result, &partial_refs);
```

### Tensor-Parallel MLP

```rust
use torchless::mlp_tensor_parallel;

let output = mlp_tensor_parallel(
    &input,
    &gate_proj,
    &up_proj,
    &down_proj,
    &tp_config,
);
```

---

## Parallel Model Implementations

### ParallelConfig

Unified configuration for all parallelization features:

```rust
use torchless::ParallelConfig;

// Default (adaptive matmul, parallel heads)
let config = ParallelConfig::default();

// Pipeline parallelism with 4 micro-batches
let config = ParallelConfig::pipeline(4);

// Tensor parallelism (2-way, rank 0)
let config = ParallelConfig::tensor_parallel(2, 0);

// Auto-configure for model
let config = ParallelConfig::for_model(&model_config);
```

### PipelineParallelMistral

Model with pipeline parallelism:

```rust
use torchless::PipelineParallelMistral;

let config = ParallelConfig::pipeline(4);
let model = PipelineParallelMistral::load(params, config)?;

model.forward(&mut state, token, false);
model.fast_forward(&mut state, token, false);
```

### TensorParallelMistral

Model with tensor parallelism:

```rust
use torchless::TensorParallelMistral;

let tp_config = TensorParallelConfig::new(2, 0);  // 2-way, rank 0
let model = TensorParallelMistral::load(params, tp_config)?;

model.forward(&mut state, token, false);
```

---

## Performance Characteristics

| Feature | Benefit | When to Use |
|---------|---------|-------------|
| Adaptive Chunk Sizes | 5-15% speedup | Always (automatic) |
| Work Stealing | Better load balance | Uneven workloads |
| Pipeline Parallelism | Overlapped computation | Batch inference |
| Column-Parallel | Split output features | QKV projections |
| Row-Parallel | Split input features | Output projections |
| Tensor Parallelism | Large model support | Models > 13B |

## Building

```bash
# Build with parallel feature
cargo build --release --features parallel

# Run tests
cargo test --features parallel kernels::parallel
cargo test --features parallel model::parallel
```
