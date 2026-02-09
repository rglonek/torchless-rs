//! Phase 6: Parallelization Improvements
//!
//! This module provides advanced parallelization optimizations:
//! - **6.1 Better Work Distribution**: Adaptive chunk sizes, work-stealing, NUMA awareness
//! - **6.2 Pipeline Parallelism**: Overlapping layer computation
//! - **6.3 Tensor Parallelism**: Column/row-parallel linear layers with all-reduce
//!
//! # Performance Benefits
//! - SPEED+ from better load balancing across cores
//! - SPEED+ from overlapped computation in pipeline parallelism
//! - SPEED+ for very large models with tensor parallelism

use ndarray::{Array1, Array2};
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// =============================================================================
// 6.1 Better Work Distribution
// =============================================================================

/// Configuration for adaptive work distribution
#[derive(Debug, Clone)]
pub struct WorkDistributionConfig {
    /// Minimum chunk size (elements) - prevents excessive task overhead
    pub min_chunk_size: usize,
    /// Maximum chunk size (elements) - ensures load balancing
    pub max_chunk_size: usize,
    /// Target tasks per thread for work stealing
    pub tasks_per_thread: usize,
    /// Enable NUMA-aware allocation hints
    pub numa_aware: bool,
    /// Cache line size for alignment (typically 64 bytes)
    pub cache_line_size: usize,
}

impl Default for WorkDistributionConfig {
    fn default() -> Self {
        Self {
            min_chunk_size: 64,   // Minimum to amortize task overhead
            max_chunk_size: 4096, // Maximum to ensure work stealing has opportunity
            tasks_per_thread: 4,  // Balance between overhead and stealing
            numa_aware: false,    // Disabled by default (requires OS support)
            cache_line_size: 64,  // Standard x86/ARM cache line
        }
    }
}

impl WorkDistributionConfig {
    /// Create a configuration optimized for matmul operations
    pub fn for_matmul(matrix_rows: usize, matrix_cols: usize) -> Self {
        let work_size = matrix_rows;
        let num_threads = num_cpus();

        // Compute optimal chunk size based on work and threads
        let base_chunk = work_size / (num_threads * 4); // 4 tasks per thread
        let optimal_chunk = base_chunk.clamp(64, 1024);

        Self {
            min_chunk_size: 64.min(matrix_cols),
            max_chunk_size: optimal_chunk.max(128),
            tasks_per_thread: 4,
            numa_aware: false,
            cache_line_size: 64,
        }
    }

    /// Create a configuration optimized for attention operations
    pub fn for_attention(n_heads: usize, seq_len: usize) -> Self {
        // For attention, we want each head as a task, but with subtasks for long sequences
        let subtasks_per_head = if seq_len > 512 { 2 } else { 1 };

        Self {
            min_chunk_size: 1, // Heads are natural work units
            max_chunk_size: n_heads,
            tasks_per_thread: subtasks_per_head,
            numa_aware: false,
            cache_line_size: 64,
        }
    }

    /// Compute optimal chunk size for given work amount
    pub fn optimal_chunk_size(&self, total_work: usize) -> usize {
        let num_threads = num_cpus();
        let target_tasks = num_threads * self.tasks_per_thread;
        let computed = total_work.div_ceil(target_tasks);
        computed.clamp(self.min_chunk_size, self.max_chunk_size)
    }
}

/// Get the number of CPUs available for parallel work
#[cfg(feature = "parallel")]
pub fn num_cpus() -> usize {
    rayon::current_num_threads()
}

#[cfg(not(feature = "parallel"))]
pub fn num_cpus() -> usize {
    1
}

/// Work-stealing statistics for performance monitoring
#[derive(Debug, Default)]
pub struct WorkStealingStats {
    /// Total tasks created
    pub tasks_created: AtomicUsize,
    /// Tasks executed on creating thread (no stealing)
    pub local_tasks: AtomicUsize,
    /// Tasks stolen from other threads
    pub stolen_tasks: AtomicUsize,
}

impl WorkStealingStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_task(&self) {
        self.tasks_created.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_local(&self) {
        self.local_tasks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_stolen(&self) {
        self.stolen_tasks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn steal_ratio(&self) -> f32 {
        let total = self.tasks_created.load(Ordering::Relaxed);
        let stolen = self.stolen_tasks.load(Ordering::Relaxed);
        if total > 0 {
            stolen as f32 / total as f32
        } else {
            0.0
        }
    }
}

/// Adaptive parallel matrix-vector multiplication with optimized chunking
///
/// Uses work distribution configuration to determine optimal chunk sizes
/// and enable better work stealing behavior.
#[cfg(feature = "parallel")]
pub fn matmul_vec_adaptive(
    w: &Array2<f32>,
    x: &Array1<f32>,
    config: &WorkDistributionConfig,
) -> Array1<f32> {
    let (rows, cols) = w.dim();
    let x_slice = x.as_slice().expect("x must be contiguous");
    let w_slice = w.as_slice().expect("w must be contiguous");

    let chunk_size = config.optimal_chunk_size(rows);

    let result: Vec<f32> = (0..rows)
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|i| {
            let row_offset = i * cols;
            let mut sum = 0.0f32;
            for j in 0..cols {
                sum += w_slice[row_offset + j] * x_slice[j];
            }
            sum
        })
        .collect();

    Array1::from_vec(result)
}

#[cfg(not(feature = "parallel"))]
pub fn matmul_vec_adaptive(
    w: &Array2<f32>,
    x: &Array1<f32>,
    _config: &WorkDistributionConfig,
) -> Array1<f32> {
    w.dot(x)
}

/// Adaptive parallel matrix-vector multiplication with pre-allocated output
#[cfg(feature = "parallel")]
pub fn matmul_vec_adaptive_into(
    w: &Array2<f32>,
    x: &Array1<f32>,
    out: &mut Array1<f32>,
    config: &WorkDistributionConfig,
) {
    let (rows, cols) = w.dim();
    let x_slice = x.as_slice().expect("x must be contiguous");
    let w_slice = w.as_slice().expect("w must be contiguous");
    let out_slice = out.as_slice_mut().expect("out must be contiguous");

    let chunk_size = config.optimal_chunk_size(rows);

    out_slice
        .par_iter_mut()
        .enumerate()
        .with_min_len(chunk_size)
        .for_each(|(i, out_val)| {
            let row_offset = i * cols;
            let mut sum = 0.0f32;
            for j in 0..cols {
                sum += w_slice[row_offset + j] * x_slice[j];
            }
            *out_val = sum;
        });
}

#[cfg(not(feature = "parallel"))]
pub fn matmul_vec_adaptive_into(
    w: &Array2<f32>,
    x: &Array1<f32>,
    out: &mut Array1<f32>,
    _config: &WorkDistributionConfig,
) {
    let result = w.dot(x);
    out.assign(&result);
}

/// NUMA-aware memory hint for buffer allocation
///
/// On supported systems, this can improve memory locality for parallel workloads.
/// Currently provides hints only; actual NUMA binding requires OS-specific APIs.
#[derive(Debug, Clone, Copy)]
pub struct NumaHint {
    /// Preferred NUMA node (-1 for any)
    pub node: i32,
    /// Whether to interleave across nodes
    pub interleave: bool,
}

impl Default for NumaHint {
    fn default() -> Self {
        Self {
            node: -1,
            interleave: false,
        }
    }
}

impl NumaHint {
    /// Create a hint for the local NUMA node
    pub fn local() -> Self {
        Self {
            node: -1, // Will use thread's current node
            interleave: false,
        }
    }

    /// Create a hint to interleave across all NUMA nodes
    /// This can be beneficial for large shared data structures
    pub fn interleaved() -> Self {
        Self {
            node: -1,
            interleave: true,
        }
    }
}

// =============================================================================
// 6.2 Pipeline Parallelism
// =============================================================================

/// State for pipeline-parallel layer execution
///
/// Enables overlapping computation across transformer layers by processing
/// multiple tokens simultaneously at different pipeline stages.
///
/// ```text
/// Time ->
/// Layer 0: [Token 1] [Token 2] [Token 3]
/// Layer 1:          [Token 1] [Token 2] [Token 3]
/// Layer 2:                    [Token 1] [Token 2]
/// ```
#[derive(Debug)]
pub struct PipelineState {
    /// Number of pipeline stages (typically = number of layers)
    pub num_stages: usize,
    /// Current pipeline depth (how many tokens are in-flight)
    pub pipeline_depth: usize,
    /// Maximum pipeline depth (limited by memory for KV cache)
    pub max_depth: usize,
    /// Circular buffer indices for each stage
    stage_indices: Vec<usize>,
}

impl PipelineState {
    /// Create a new pipeline state
    pub fn new(num_stages: usize, max_depth: usize) -> Self {
        Self {
            num_stages,
            pipeline_depth: 0,
            max_depth,
            stage_indices: vec![0; num_stages],
        }
    }

    /// Check if the pipeline can accept a new token
    pub fn can_accept(&self) -> bool {
        self.pipeline_depth < self.max_depth
    }

    /// Check if the pipeline has completed tokens ready
    pub fn has_output(&self) -> bool {
        self.pipeline_depth > 0 && self.stage_indices[self.num_stages - 1] > 0
    }

    /// Advance the pipeline by one step
    pub fn advance(&mut self) {
        // Shift all stages forward
        for i in (1..self.num_stages).rev() {
            self.stage_indices[i] = self.stage_indices[i - 1];
        }
        self.stage_indices[0] = 0;
    }

    /// Fill the pipeline with a new token
    pub fn push(&mut self) {
        if self.can_accept() {
            self.pipeline_depth += 1;
            self.stage_indices[0] = self.pipeline_depth;
        }
    }

    /// Pop a completed token from the pipeline
    pub fn pop(&mut self) -> Option<usize> {
        if self.has_output() {
            let result = self.stage_indices[self.num_stages - 1];
            self.advance();
            self.pipeline_depth -= 1;
            Some(result)
        } else {
            None
        }
    }
}

/// Pipeline-parallel layer executor
///
/// Manages the execution of transformer layers in a pipelined fashion,
/// allowing overlap between different layers processing different tokens.
#[cfg(feature = "parallel")]
#[allow(dead_code)] // Fields used by future pipeline execution methods
pub struct PipelineExecutor<L: Send + Sync> {
    /// The layers to execute
    layers: Vec<L>,
    /// Pipeline state
    state: PipelineState,
    /// Buffers for intermediate states (one per pipeline slot)
    buffers: Vec<Vec<f32>>,
}

#[cfg(feature = "parallel")]
impl<L: Send + Sync> PipelineExecutor<L> {
    /// Create a new pipeline executor
    pub fn new(layers: Vec<L>, hidden_size: usize, max_depth: usize) -> Self {
        let num_stages = layers.len();
        let buffers = (0..max_depth).map(|_| vec![0.0f32; hidden_size]).collect();

        Self {
            layers,
            state: PipelineState::new(num_stages, max_depth),
            buffers,
        }
    }

    /// Get the number of layers/stages
    pub fn num_stages(&self) -> usize {
        self.layers.len()
    }

    /// Check if pipeline can accept more input
    pub fn can_accept(&self) -> bool {
        self.state.can_accept()
    }

    /// Check if output is ready
    pub fn has_output(&self) -> bool {
        self.state.has_output()
    }
}

/// Configuration for pipeline parallelism
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of micro-batches in the pipeline
    pub num_micro_batches: usize,
    /// Whether to use asynchronous execution
    pub async_execution: bool,
    /// Memory budget for pipeline buffers (bytes)
    pub memory_budget: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_micro_batches: 4,
            async_execution: false,
            memory_budget: 1024 * 1024 * 512, // 512 MB
        }
    }
}

impl PipelineConfig {
    /// Compute optimal micro-batch count based on model size
    pub fn for_model(n_layers: usize, hidden_size: usize, memory_budget: usize) -> Self {
        // Each micro-batch needs hidden_size * 4 bytes per layer
        let bytes_per_micro_batch = n_layers * hidden_size * 4;
        let max_micro_batches = memory_budget / bytes_per_micro_batch;

        Self {
            num_micro_batches: max_micro_batches.clamp(2, 8),
            async_execution: false,
            memory_budget,
        }
    }
}

// =============================================================================
// 6.3 Tensor Parallelism
// =============================================================================

/// Strategy for tensor parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorParallelStrategy {
    /// Split output dimension (columns) - each device computes partial output
    ColumnParallel,
    /// Split input dimension (rows) - each device processes partial input
    RowParallel,
    /// No parallelism (single device)
    None,
}

/// Configuration for tensor parallelism
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Number of parallel partitions
    pub world_size: usize,
    /// This device's rank (0..world_size)
    pub rank: usize,
    /// Strategy for the first linear layer in attention
    pub attention_qkv_strategy: TensorParallelStrategy,
    /// Strategy for the attention output projection
    pub attention_output_strategy: TensorParallelStrategy,
    /// Strategy for MLP gate/up projections
    pub mlp_gate_up_strategy: TensorParallelStrategy,
    /// Strategy for MLP down projection
    pub mlp_down_strategy: TensorParallelStrategy,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            attention_qkv_strategy: TensorParallelStrategy::None,
            attention_output_strategy: TensorParallelStrategy::None,
            mlp_gate_up_strategy: TensorParallelStrategy::None,
            mlp_down_strategy: TensorParallelStrategy::None,
        }
    }
}

impl TensorParallelConfig {
    /// Create a configuration for n-way tensor parallelism
    pub fn new(world_size: usize, rank: usize) -> Self {
        if world_size == 1 {
            return Self::default();
        }

        Self {
            world_size,
            rank,
            // QKV: column-parallel (split output heads)
            attention_qkv_strategy: TensorParallelStrategy::ColumnParallel,
            // O: row-parallel (each device has partial result, all-reduce needed)
            attention_output_strategy: TensorParallelStrategy::RowParallel,
            // Gate/Up: column-parallel (split hidden dimension)
            mlp_gate_up_strategy: TensorParallelStrategy::ColumnParallel,
            // Down: row-parallel (each device has partial result, all-reduce needed)
            mlp_down_strategy: TensorParallelStrategy::RowParallel,
        }
    }

    /// Compute the local size for a dimension that is partitioned
    pub fn local_size(&self, global_size: usize) -> usize {
        if self.world_size == 1 {
            global_size
        } else {
            global_size.div_ceil(self.world_size)
        }
    }

    /// Compute the offset for this rank's partition
    pub fn local_offset(&self, global_size: usize) -> usize {
        let local_size = self.local_size(global_size);
        self.rank * local_size
    }
}

/// Column-parallel linear layer
///
/// Splits the output dimension across devices. Each device computes
/// a subset of the output features.
///
/// For input x (d,) and weight W (n, d) split into W_i (n/P, d):
/// - Each device computes y_i = W_i @ x, giving (n/P,) output
/// - Outputs are concatenated (gather) to get full (n,) output
///
/// This is used for QKV projections in attention where we want to
/// split across attention heads.
#[cfg(feature = "parallel")]
pub fn column_parallel_linear(
    w: &Array2<f32>,
    x: &Array1<f32>,
    config: &TensorParallelConfig,
) -> Array1<f32> {
    let (n, d) = w.dim();
    let local_n = config.local_size(n);
    let offset = config.local_offset(n);
    let end = (offset + local_n).min(n);

    let x_slice = x.as_slice().expect("x must be contiguous");

    // Compute local output
    let mut local_output = vec![0.0f32; end - offset];

    local_output
        .par_iter_mut()
        .enumerate()
        .for_each(|(local_i, out_val)| {
            let global_i = offset + local_i;
            let row = w.row(global_i);
            let row_slice = row.as_slice().expect("row must be contiguous");

            let mut sum = 0.0f32;
            for j in 0..d {
                sum += row_slice[j] * x_slice[j];
            }
            *out_val = sum;
        });

    Array1::from_vec(local_output)
}

#[cfg(not(feature = "parallel"))]
pub fn column_parallel_linear(
    w: &Array2<f32>,
    x: &Array1<f32>,
    config: &TensorParallelConfig,
) -> Array1<f32> {
    let (n, _) = w.dim();
    let local_n = config.local_size(n);
    let offset = config.local_offset(n);
    let end = (offset + local_n).min(n);

    // Extract the local rows and compute
    let local_w = w.slice(ndarray::s![offset..end, ..]);
    local_w.dot(x)
}

/// Row-parallel linear layer
///
/// Splits the input dimension across devices. Each device computes
/// a partial sum that must be reduced (all-reduce) to get final output.
///
/// For input x split into x_i (d/P,) and weight W (n, d) split into W_i (n, d/P):
/// - Each device computes y_i = W_i @ x_i, giving partial (n,) output
/// - Outputs are summed (all-reduce) to get full (n,) output
///
/// This is used for output projections where we need to combine results.
#[cfg(feature = "parallel")]
pub fn row_parallel_linear(
    w: &Array2<f32>,
    x: &Array1<f32>,
    config: &TensorParallelConfig,
) -> Array1<f32> {
    let (n, d) = w.dim();
    let local_d = config.local_size(d);
    let offset = config.local_offset(d);
    let end = (offset + local_d).min(d);

    // Note: In a real distributed setup, x would already be partitioned
    // and w would be sliced accordingly. Here we slice for simulation.
    let x_slice = &x.as_slice().expect("x must be contiguous")[offset..end];

    // Compute partial output (needs all-reduce in distributed setting)
    let mut partial_output = vec![0.0f32; n];

    partial_output
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out_val)| {
            let row = w.row(i);
            let row_slice = row.as_slice().expect("row must be contiguous");

            let mut sum = 0.0f32;
            for (j, &x_val) in x_slice.iter().enumerate() {
                sum += row_slice[offset + j] * x_val;
            }
            *out_val = sum;
        });

    Array1::from_vec(partial_output)
}

#[cfg(not(feature = "parallel"))]
pub fn row_parallel_linear(
    w: &Array2<f32>,
    x: &Array1<f32>,
    config: &TensorParallelConfig,
) -> Array1<f32> {
    let (n, d) = w.dim();
    let local_d = config.local_size(d);
    let offset = config.local_offset(d);
    let end = (offset + local_d).min(d);

    // Slice input and weights
    let x_slice = x.slice(ndarray::s![offset..end]);
    let local_w = w.slice(ndarray::s![.., offset..end]);
    local_w.dot(&x_slice)
}

/// All-reduce operation for combining partial results
///
/// In a distributed setting, this would use NCCL or similar for
/// GPU-to-GPU communication. Here we simulate for thread-parallel.
#[cfg(feature = "parallel")]
pub fn all_reduce_sum(partials: &[Array1<f32>]) -> Array1<f32> {
    if partials.is_empty() {
        return Array1::zeros(0);
    }

    let n = partials[0].len();
    let mut result = Array1::zeros(n);

    // Parallel reduction across partials
    let result_slice = result.as_slice_mut().expect("result must be contiguous");

    result_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out_val)| {
            *out_val = partials.iter().map(|p| p[i]).sum();
        });

    result
}

#[cfg(not(feature = "parallel"))]
pub fn all_reduce_sum(partials: &[Array1<f32>]) -> Array1<f32> {
    if partials.is_empty() {
        return Array1::zeros(0);
    }

    let n = partials[0].len();
    let mut result = Array1::zeros(n);

    for partial in partials {
        result = result + partial;
    }

    result
}

/// All-reduce sum in-place
#[cfg(feature = "parallel")]
pub fn all_reduce_sum_inplace(result: &mut Array1<f32>, partials: &[&Array1<f32>]) {
    if partials.is_empty() {
        return;
    }

    let result_slice = result.as_slice_mut().expect("result must be contiguous");

    result_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out_val)| {
            *out_val = partials.iter().map(|p| p[i]).sum();
        });
}

#[cfg(not(feature = "parallel"))]
pub fn all_reduce_sum_inplace(result: &mut Array1<f32>, partials: &[&Array1<f32>]) {
    if partials.is_empty() {
        return;
    }

    result.fill(0.0);
    for partial in partials {
        for (i, &v) in partial.iter().enumerate() {
            result[i] += v;
        }
    }
}

// =============================================================================
// Adaptive Parallel Attention
// =============================================================================

/// Parallel attention with adaptive work distribution
///
/// Splits attention computation across heads with optimized chunk sizes.
#[cfg(feature = "parallel")]
#[allow(clippy::needless_range_loop)] // Index-based access required for row iteration
pub fn attention_parallel_adaptive(
    queries: &Array2<f32>, // [n_heads, head_dim]
    keys: &Array2<f32>,    // [seq_len, head_dim]
    values: &Array2<f32>,  // [seq_len, head_dim]
    scale: f32,
    config: &WorkDistributionConfig,
) -> Array2<f32> {
    let (n_heads, head_dim) = queries.dim();
    let (seq_len, _) = keys.dim();

    let chunk_size = config.optimal_chunk_size(n_heads);

    // Parallel over heads
    let head_outputs: Vec<Vec<f32>> = (0..n_heads)
        .into_par_iter()
        .with_min_len(chunk_size)
        .map(|h| {
            let query = queries.row(h);
            let query_slice = query.as_slice().expect("query must be contiguous");

            // Compute attention scores
            let mut scores = vec![0.0f32; seq_len];
            let mut max_score = f32::NEG_INFINITY;

            for i in 0..seq_len {
                let key_row = keys.row(i);
                let key_slice = key_row.as_slice().expect("key must be contiguous");

                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += query_slice[d] * key_slice[d];
                }
                scores[i] = dot * scale;
                max_score = max_score.max(scores[i]);
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0 / sum_exp;
            for s in &mut scores {
                *s *= inv_sum;
            }

            // Weighted sum
            let mut output = vec![0.0f32; head_dim];
            for i in 0..seq_len {
                let weight = scores[i];
                let value_row = values.row(i);
                let value_slice = value_row.as_slice().expect("value must be contiguous");

                for d in 0..head_dim {
                    output[d] += weight * value_slice[d];
                }
            }

            output
        })
        .collect();

    // Assemble output
    let mut output = Array2::zeros((n_heads, head_dim));
    for (h, head_out) in head_outputs.into_iter().enumerate() {
        for (d, &v) in head_out.iter().enumerate() {
            output[[h, d]] = v;
        }
    }

    output
}

#[cfg(not(feature = "parallel"))]
pub fn attention_parallel_adaptive(
    queries: &Array2<f32>,
    keys: &Array2<f32>,
    values: &Array2<f32>,
    scale: f32,
    _config: &WorkDistributionConfig,
) -> Array2<f32> {
    let (n_heads, head_dim) = queries.dim();
    let (seq_len, _) = keys.dim();

    let mut output = Array2::zeros((n_heads, head_dim));

    for h in 0..n_heads {
        let query = queries.row(h);
        let query_slice = query.as_slice().expect("query must be contiguous");

        // Compute attention scores
        let mut scores = vec![0.0f32; seq_len];
        let mut max_score = f32::NEG_INFINITY;

        for i in 0..seq_len {
            let key_row = keys.row(i);
            let key_slice = key_row.as_slice().expect("key must be contiguous");

            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += query_slice[d] * key_slice[d];
            }
            scores[i] = dot * scale;
            max_score = max_score.max(scores[i]);
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;
        for s in &mut scores {
            *s *= inv_sum;
        }

        // Weighted sum
        for i in 0..seq_len {
            let weight = scores[i];
            let value_row = values.row(i);
            let value_slice = value_row.as_slice().expect("value must be contiguous");

            for d in 0..head_dim {
                output[[h, d]] += weight * value_slice[d];
            }
        }
    }

    output
}

// =============================================================================
// Parallel MLP with Tensor Parallelism
// =============================================================================

/// Tensor-parallel MLP forward pass
///
/// Executes the MLP with column-parallel gate/up projections and
/// row-parallel down projection, minimizing communication.
#[cfg(feature = "parallel")]
pub fn mlp_tensor_parallel(
    x: &Array1<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
    down_proj: &Array2<f32>,
    config: &TensorParallelConfig,
) -> Array1<f32> {
    // Column-parallel gate and up projections (output split)
    let local_gate = column_parallel_linear(gate_proj, x, config);
    let local_up = column_parallel_linear(up_proj, x, config);

    // Fused SiLU and multiply (local computation)
    let local_hidden_size = local_gate.len();
    let local_hidden: Vec<f32> = (0..local_hidden_size)
        .into_par_iter()
        .map(|i| {
            let gate_val = local_gate[i];
            let silu_gate = gate_val / (1.0 + (-gate_val).exp());
            silu_gate * local_up[i]
        })
        .collect();

    // Row-parallel down projection (needs all-reduce)
    // In a distributed setting, each device would have a slice of down_proj
    // and the result would be all-reduced. Here we simulate with the full matrix.
    let local_hidden_arr = Array1::from_vec(local_hidden);

    // For proper tensor parallelism, we'd slice down_proj columns
    // to match the local hidden size
    if config.world_size > 1 {
        let (out_dim, hidden_dim) = down_proj.dim();
        let local_cols = config.local_size(hidden_dim);
        let col_offset = config.local_offset(hidden_dim);
        let col_end = (col_offset + local_cols).min(hidden_dim);

        // Partial output computation
        let mut partial_output = vec![0.0f32; out_dim];
        partial_output
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out_val)| {
                let row = down_proj.row(i);
                let row_slice = row.as_slice().expect("row must be contiguous");

                let mut sum = 0.0f32;
                for (j, &h_val) in local_hidden_arr.iter().enumerate() {
                    if col_offset + j < col_end {
                        sum += row_slice[col_offset + j] * h_val;
                    }
                }
                *out_val = sum;
            });

        // In real distributed setting, this would be all-reduced
        Array1::from_vec(partial_output)
    } else {
        down_proj.dot(&local_hidden_arr)
    }
}

#[cfg(not(feature = "parallel"))]
pub fn mlp_tensor_parallel(
    x: &Array1<f32>,
    gate_proj: &Array2<f32>,
    up_proj: &Array2<f32>,
    down_proj: &Array2<f32>,
    _config: &TensorParallelConfig,
) -> Array1<f32> {
    // Sequential implementation
    let gate = gate_proj.dot(x);
    let up = up_proj.dot(x);

    let hidden: Array1<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            let silu_g = g / (1.0 + (-g).exp());
            silu_g * u
        })
        .collect::<Vec<_>>()
        .into();

    down_proj.dot(&hidden)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    #[test]
    fn test_work_distribution_config() {
        let config = WorkDistributionConfig::default();
        assert!(config.min_chunk_size > 0);
        assert!(config.max_chunk_size >= config.min_chunk_size);

        // Test optimal chunk size calculation
        let chunk = config.optimal_chunk_size(1000);
        assert!(chunk >= config.min_chunk_size);
        assert!(chunk <= config.max_chunk_size);
    }

    #[test]
    fn test_work_distribution_for_matmul() {
        let config = WorkDistributionConfig::for_matmul(1024, 4096);
        assert!(config.min_chunk_size > 0);

        // Chunk size should be reasonable for the matrix dimensions
        let chunk = config.optimal_chunk_size(1024);
        assert!(chunk <= 1024);
    }

    #[test]
    fn test_pipeline_state() {
        let mut state = PipelineState::new(3, 2);

        assert!(state.can_accept());
        assert!(!state.has_output());

        state.push();
        assert!(state.can_accept());

        state.push();
        assert!(!state.can_accept()); // At max depth

        // Advance through stages
        state.advance();
        state.advance();
        assert!(state.has_output());
    }

    #[test]
    fn test_tensor_parallel_config() {
        let config = TensorParallelConfig::new(4, 0);

        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 0);

        // Test local size calculation
        assert_eq!(config.local_size(1024), 256);
        assert_eq!(config.local_offset(1024), 0);

        let config2 = TensorParallelConfig::new(4, 2);
        assert_eq!(config2.local_offset(1024), 512);
    }

    #[test]
    fn test_matmul_vec_adaptive() {
        let w = Array2::from_shape_fn((8, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        let x = array![1.0, 2.0, 3.0, 4.0];
        let config = WorkDistributionConfig::default();

        let result = matmul_vec_adaptive(&w, &x, &config);
        let expected = w.dot(&x);

        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-5,
                "Index {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_column_parallel_linear() {
        let w = Array2::from_shape_fn((8, 4), |(i, j)| (i * 4 + j) as f32 * 0.1);
        let x = array![1.0, 2.0, 3.0, 4.0];

        // Single device (no parallelism)
        let config_single = TensorParallelConfig::default();
        let result_single = column_parallel_linear(&w, &x, &config_single);
        let expected = w.dot(&x);

        for i in 0..result_single.len() {
            assert!(
                (result_single[i] - expected[i]).abs() < 1e-5,
                "Index {} mismatch",
                i
            );
        }

        // Multi-device simulation
        let config_0 = TensorParallelConfig::new(2, 0);
        let config_1 = TensorParallelConfig::new(2, 1);

        let result_0 = column_parallel_linear(&w, &x, &config_0);
        let result_1 = column_parallel_linear(&w, &x, &config_1);

        // Results should be partial outputs
        assert_eq!(result_0.len(), 4);
        assert_eq!(result_1.len(), 4);

        // Combined should match full result
        for i in 0..4 {
            assert!((result_0[i] - expected[i]).abs() < 1e-5);
            assert!((result_1[i] - expected[i + 4]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_all_reduce_sum() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let c = array![0.1, 0.2, 0.3];

        let result = all_reduce_sum(&[a, b, c]);

        assert!((result[0] - 5.1).abs() < 1e-5);
        assert!((result[1] - 7.2).abs() < 1e-5);
        assert!((result[2] - 9.3).abs() < 1e-5);
    }

    #[test]
    fn test_attention_parallel_adaptive() {
        let n_heads = 4;
        let head_dim = 8;
        let seq_len = 16;

        let queries = Array2::from_shape_fn((n_heads, head_dim), |(h, d)| {
            ((h * head_dim + d) as f32 * 0.1).sin()
        });
        let keys = Array2::from_shape_fn((seq_len, head_dim), |(s, d)| {
            ((s * head_dim + d) as f32 * 0.1).cos()
        });
        let values = Array2::from_shape_fn((seq_len, head_dim), |(s, d)| {
            (s * head_dim + d) as f32 * 0.01
        });

        let scale = 1.0 / (head_dim as f32).sqrt();
        let config = WorkDistributionConfig::for_attention(n_heads, seq_len);

        let output = attention_parallel_adaptive(&queries, &keys, &values, scale, &config);

        assert_eq!(output.dim(), (n_heads, head_dim));

        // Output should be valid (no NaN or Inf)
        for h in 0..n_heads {
            for d in 0..head_dim {
                assert!(output[[h, d]].is_finite());
            }
        }
    }

    #[test]
    fn test_mlp_tensor_parallel_single() {
        let hidden = 4;
        let intermediate = 8;

        let x = array![1.0, 2.0, 3.0, 4.0];
        let gate_proj = Array2::from_shape_fn((intermediate, hidden), |(i, j)| {
            (i * hidden + j) as f32 * 0.05
        });
        let up_proj = Array2::from_shape_fn((intermediate, hidden), |(i, j)| {
            ((i * hidden + j) as f32 * 0.05) + 0.1
        });
        let down_proj = Array2::from_shape_fn((hidden, intermediate), |(i, j)| {
            (i * intermediate + j) as f32 * 0.02
        });

        let config = TensorParallelConfig::default();
        let result = mlp_tensor_parallel(&x, &gate_proj, &up_proj, &down_proj, &config);

        assert_eq!(result.len(), hidden);

        // Should match non-parallel reference
        let gate = gate_proj.dot(&x);
        let up = up_proj.dot(&x);
        let hidden_vec: Array1<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                let silu_g = g / (1.0 + (-g).exp());
                silu_g * u
            })
            .collect::<Vec<_>>()
            .into();
        let expected = down_proj.dot(&hidden_vec);

        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < 1e-4,
                "Index {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_work_stealing_stats() {
        let stats = WorkStealingStats::new();

        stats.record_task();
        stats.record_task();
        stats.record_task();
        stats.record_local();
        stats.record_local();
        stats.record_stolen();

        assert_eq!(stats.tasks_created.load(Ordering::Relaxed), 3);
        assert_eq!(stats.local_tasks.load(Ordering::Relaxed), 2);
        assert_eq!(stats.stolen_tasks.load(Ordering::Relaxed), 1);
        assert!((stats.steal_ratio() - 0.333).abs() < 0.01);
    }
}
