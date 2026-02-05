//! Continuous Batching Implementation
//!
//! Continuous batching enables efficient processing of multiple sequences simultaneously.
//! Unlike static batching, sequences can dynamically join and leave the batch as they
//! complete or new requests arrive.
//!
//! # Features
//! - Dynamic sequence management (add/remove sequences at any time)
//! - PagedAttention-style KV cache management
//! - Preemption support for priority-based scheduling
//! - Memory-efficient variable-length sequence handling
//!
//! # Performance
//! - Throughput improvement: 2-5x for server workloads with varying sequence lengths
//! - Memory efficiency: Better GPU/CPU utilization through dynamic batching
//!
//! # References
//! - Orca: A Distributed Serving System for Transformer-Based Generative Models
//! - vLLM: Efficient Memory Management for Large Language Model Serving

use crate::loader::Config;
use ndarray::{Array1, Array2, Array4};
use std::collections::{HashMap, VecDeque};

/// Unique identifier for a sequence in the batch.
pub type SequenceId = u64;

/// Status of a sequence in the batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Sequence is waiting to be processed
    Pending,
    /// Sequence is actively being generated
    Running,
    /// Sequence has completed (EOS or max length)
    Finished,
    /// Sequence was preempted (paused to make room for others)
    Preempted,
    /// Sequence was cancelled
    Cancelled,
}

/// A sequence being processed in the batch.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique identifier
    pub id: SequenceId,
    /// Current status
    pub status: SequenceStatus,
    /// Input token IDs
    pub input_tokens: Vec<u32>,
    /// Generated output token IDs
    pub output_tokens: Vec<u32>,
    /// Current position in generation
    pub position: usize,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Priority (higher = more important)
    pub priority: i32,
    /// KV cache block indices for this sequence
    pub cache_blocks: Vec<usize>,
    /// Timestamp when sequence was added
    pub created_at: std::time::Instant,
}

impl Sequence {
    /// Create a new sequence.
    pub fn new(
        id: SequenceId,
        input_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
    ) -> Self {
        let position = input_tokens.len();
        Self {
            id,
            status: SequenceStatus::Pending,
            input_tokens,
            output_tokens: Vec::new(),
            position,
            max_tokens,
            temperature,
            priority: 0,
            cache_blocks: Vec::new(),
            created_at: std::time::Instant::now(),
        }
    }

    /// Total length of the sequence (input + output).
    pub fn total_length(&self) -> usize {
        self.input_tokens.len() + self.output_tokens.len()
    }

    /// Check if the sequence has finished generating.
    pub fn is_finished(&self) -> bool {
        self.status == SequenceStatus::Finished ||
        self.status == SequenceStatus::Cancelled ||
        self.output_tokens.len() >= self.max_tokens
    }

    /// Get all tokens (input + output).
    pub fn all_tokens(&self) -> Vec<u32> {
        let mut tokens = self.input_tokens.clone();
        tokens.extend(&self.output_tokens);
        tokens
    }
}

/// Configuration for continuous batching.
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    /// Maximum number of sequences in a batch
    pub max_batch_size: usize,
    /// Maximum total tokens across all sequences in a batch
    pub max_batch_tokens: usize,
    /// Maximum sequence length supported
    pub max_seq_len: usize,
    /// Number of KV cache blocks
    pub num_cache_blocks: usize,
    /// Size of each cache block (in tokens)
    pub block_size: usize,
    /// Enable preemption for fairness
    pub enable_preemption: bool,
    /// Time threshold for preemption (in seconds)
    pub preemption_threshold: f64,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_batch_tokens: 4096,
            max_seq_len: 2048,
            num_cache_blocks: 256,
            block_size: 16,
            enable_preemption: true,
            preemption_threshold: 30.0,
        }
    }
}

impl BatchingConfig {
    /// Create config for low-memory systems.
    pub fn low_memory() -> Self {
        Self {
            max_batch_size: 8,
            max_batch_tokens: 1024,
            max_seq_len: 512,
            num_cache_blocks: 64,
            block_size: 16,
            enable_preemption: true,
            preemption_threshold: 10.0,
        }
    }

    /// Create config for high-throughput systems.
    pub fn high_throughput() -> Self {
        Self {
            max_batch_size: 64,
            max_batch_tokens: 8192,
            max_seq_len: 4096,
            num_cache_blocks: 512,
            block_size: 32,
            enable_preemption: true,
            preemption_threshold: 60.0,
        }
    }
}

/// Block-based KV cache manager for efficient memory allocation.
///
/// Uses a block allocation strategy similar to PagedAttention:
/// - KV cache is divided into fixed-size blocks
/// - Sequences are allocated blocks as they grow
/// - Blocks can be freed and reused when sequences complete
pub struct KVCachePool {
    /// Number of layers
    n_layers: usize,
    /// Number of KV heads
    n_kv_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Size of each block (in tokens)
    block_size: usize,
    /// Total number of blocks
    num_blocks: usize,
    /// Free block indices
    free_blocks: VecDeque<usize>,
    /// K cache blocks: [n_blocks, n_layers, n_kv_heads, block_size, head_dim]
    k_blocks: Array4<f32>,
    /// V cache blocks: [n_blocks, n_layers, n_kv_heads, block_size, head_dim]
    v_blocks: Array4<f32>,
}

impl KVCachePool {
    /// Create a new KV cache pool.
    pub fn new(
        config: &Config,
        batching_config: &BatchingConfig,
    ) -> Self {
        let n_layers = config.n_layers;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.hidden_size / config.n_heads;
        let block_size = batching_config.block_size;
        let num_blocks = batching_config.num_cache_blocks;

        // Note: We're simplifying the shape here for the implementation.
        // In a full implementation, blocks would be [n_blocks, block_size, n_kv_heads, head_dim]
        // per layer, stored separately.
        let k_blocks = Array4::zeros((num_blocks * n_layers, n_kv_heads, block_size, head_dim));
        let v_blocks = Array4::zeros((num_blocks * n_layers, n_kv_heads, block_size, head_dim));

        let free_blocks: VecDeque<usize> = (0..num_blocks).collect();

        Self {
            n_layers,
            n_kv_heads,
            head_dim,
            block_size,
            num_blocks,
            free_blocks,
            k_blocks,
            v_blocks,
        }
    }

    /// Allocate a block, returning its index.
    pub fn allocate_block(&mut self) -> Option<usize> {
        self.free_blocks.pop_front()
    }

    /// Free a block, returning it to the pool.
    pub fn free_block(&mut self, block_idx: usize) {
        self.free_blocks.push_back(block_idx);
    }

    /// Get number of free blocks.
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get number of allocated blocks.
    pub fn allocated_block_count(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Check if there are enough free blocks for a sequence of given length.
    pub fn can_allocate(&self, token_count: usize) -> bool {
        let blocks_needed = (token_count + self.block_size - 1) / self.block_size;
        self.free_block_count() >= blocks_needed * self.n_layers
    }

    /// Copy KV state for a position into cache block.
    pub fn write_kv(
        &mut self,
        block_idx: usize,
        layer_idx: usize,
        position_in_block: usize,
        k_state: &Array2<f32>,  // [n_kv_heads, head_dim]
        v_state: &Array2<f32>,
    ) {
        let linear_block = block_idx * self.n_layers + layer_idx;
        
        for h in 0..self.n_kv_heads {
            for d in 0..self.head_dim {
                self.k_blocks[[linear_block, h, position_in_block, d]] = k_state[[h, d]];
                self.v_blocks[[linear_block, h, position_in_block, d]] = v_state[[h, d]];
            }
        }
    }

    /// Read KV state from cache block.
    pub fn read_kv(
        &self,
        block_idx: usize,
        layer_idx: usize,
        position_in_block: usize,
    ) -> (Array2<f32>, Array2<f32>) {
        let linear_block = block_idx * self.n_layers + layer_idx;
        
        let mut k_state = Array2::zeros((self.n_kv_heads, self.head_dim));
        let mut v_state = Array2::zeros((self.n_kv_heads, self.head_dim));
        
        for h in 0..self.n_kv_heads {
            for d in 0..self.head_dim {
                k_state[[h, d]] = self.k_blocks[[linear_block, h, position_in_block, d]];
                v_state[[h, d]] = self.v_blocks[[linear_block, h, position_in_block, d]];
            }
        }
        
        (k_state, v_state)
    }

    /// Get block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

/// Statistics for continuous batching.
#[derive(Debug, Clone, Default)]
pub struct BatchingStats {
    /// Total sequences processed
    pub sequences_completed: usize,
    /// Total tokens generated
    pub tokens_generated: usize,
    /// Total batched iterations
    pub iterations: usize,
    /// Number of preemptions
    pub preemptions: usize,
    /// Average batch size
    pub avg_batch_size: f32,
    /// Average tokens per second
    pub tokens_per_second: f32,
}

impl BatchingStats {
    /// Update average batch size after an iteration.
    pub fn update_avg_batch_size(&mut self, batch_size: usize) {
        let n = self.iterations as f32;
        self.avg_batch_size = (self.avg_batch_size * n + batch_size as f32) / (n + 1.0);
    }
}

/// Scheduler for managing sequences in the batch.
pub struct BatchScheduler {
    /// Configuration
    config: BatchingConfig,
    /// Pending sequences waiting to be processed
    pending_queue: VecDeque<Sequence>,
    /// Currently running sequences
    running: HashMap<SequenceId, Sequence>,
    /// Completed sequences
    finished: Vec<Sequence>,
    /// Next sequence ID
    next_id: SequenceId,
    /// KV cache pool
    cache_pool: Option<KVCachePool>,
    /// Statistics
    stats: BatchingStats,
}

impl BatchScheduler {
    /// Create a new batch scheduler.
    pub fn new(config: BatchingConfig) -> Self {
        Self {
            config,
            pending_queue: VecDeque::new(),
            running: HashMap::new(),
            finished: Vec::new(),
            next_id: 0,
            cache_pool: None,
            stats: BatchingStats::default(),
        }
    }

    /// Initialize the KV cache pool.
    pub fn init_cache(&mut self, model_config: &Config) {
        self.cache_pool = Some(KVCachePool::new(model_config, &self.config));
    }

    /// Add a new sequence to the batch.
    pub fn add_sequence(
        &mut self,
        input_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
    ) -> SequenceId {
        let id = self.next_id;
        self.next_id += 1;

        let seq = Sequence::new(id, input_tokens, max_tokens, temperature);
        self.pending_queue.push_back(seq);

        id
    }

    /// Add a sequence with priority.
    pub fn add_sequence_with_priority(
        &mut self,
        input_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        priority: i32,
    ) -> SequenceId {
        let id = self.next_id;
        self.next_id += 1;

        let mut seq = Sequence::new(id, input_tokens, max_tokens, temperature);
        seq.priority = priority;
        
        // Insert in priority order
        let insert_pos = self.pending_queue
            .iter()
            .position(|s| s.priority < priority)
            .unwrap_or(self.pending_queue.len());
        self.pending_queue.insert(insert_pos, seq);

        id
    }

    /// Cancel a sequence.
    pub fn cancel_sequence(&mut self, id: SequenceId) -> bool {
        // Check pending queue
        if let Some(pos) = self.pending_queue.iter().position(|s| s.id == id) {
            let mut seq = self.pending_queue.remove(pos).unwrap();
            seq.status = SequenceStatus::Cancelled;
            self.finished.push(seq);
            return true;
        }

        // Check running
        if let Some(mut seq) = self.running.remove(&id) {
            seq.status = SequenceStatus::Cancelled;
            self.free_sequence_cache(&seq);
            self.finished.push(seq);
            return true;
        }

        false
    }

    /// Get the status of a sequence.
    pub fn get_status(&self, id: SequenceId) -> Option<SequenceStatus> {
        if let Some(seq) = self.pending_queue.iter().find(|s| s.id == id) {
            return Some(seq.status);
        }
        if let Some(seq) = self.running.get(&id) {
            return Some(seq.status);
        }
        if let Some(seq) = self.finished.iter().find(|s| s.id == id) {
            return Some(seq.status);
        }
        None
    }

    /// Get the output tokens for a finished sequence.
    pub fn get_output(&self, id: SequenceId) -> Option<&[u32]> {
        self.finished.iter()
            .find(|s| s.id == id)
            .map(|s| s.output_tokens.as_slice())
    }

    /// Schedule the next batch of sequences to process.
    pub fn schedule(&mut self) -> Vec<SequenceId> {
        let mut batch = Vec::new();
        let mut batch_tokens = 0;

        // First, include running sequences that aren't finished
        for (id, seq) in &self.running {
            if !seq.is_finished() && batch.len() < self.config.max_batch_size {
                batch.push(*id);
                batch_tokens += 1; // One token per running sequence
            }
        }

        // Try to add pending sequences
        while batch.len() < self.config.max_batch_size && !self.pending_queue.is_empty() {
            if let Some(seq) = self.pending_queue.front() {
                let tokens_needed = seq.input_tokens.len();
                
                // Check if we have space
                if batch_tokens + tokens_needed > self.config.max_batch_tokens {
                    break;
                }

                // Check if we can allocate cache
                if let Some(pool) = &self.cache_pool {
                    if !pool.can_allocate(seq.total_length() + seq.max_tokens) {
                        // Try preemption if enabled
                        if self.config.enable_preemption && !self.running.is_empty() {
                            self.preempt_one();
                        } else {
                            break;
                        }
                    }
                }

                let mut seq = self.pending_queue.pop_front().unwrap();
                seq.status = SequenceStatus::Running;
                batch.push(seq.id);
                batch_tokens += tokens_needed;
                self.running.insert(seq.id, seq);
            } else {
                break;
            }
        }

        self.stats.iterations += 1;
        self.stats.update_avg_batch_size(batch.len());

        batch
    }

    /// Preempt the oldest running sequence to make room.
    fn preempt_one(&mut self) {
        // Find the oldest non-priority sequence
        let oldest_id = self.running.iter()
            .filter(|(_, s)| s.priority <= 0)
            .min_by_key(|(_, s)| s.created_at)
            .map(|(id, _)| *id);

        if let Some(id) = oldest_id {
            if let Some(mut seq) = self.running.remove(&id) {
                seq.status = SequenceStatus::Preempted;
                self.free_sequence_cache(&seq);
                self.pending_queue.push_front(seq);
                self.stats.preemptions += 1;
            }
        }
    }

    /// Free cache blocks for a sequence.
    fn free_sequence_cache(&mut self, seq: &Sequence) {
        if let Some(pool) = &mut self.cache_pool {
            for &block_idx in &seq.cache_blocks {
                pool.free_block(block_idx);
            }
        }
    }

    /// Process generated tokens for a batch.
    /// Returns (sequence_id, generated_token) pairs.
    pub fn process_outputs(&mut self, outputs: Vec<(SequenceId, u32)>, eos_token: u32) {
        for (id, token) in outputs {
            if let Some(seq) = self.running.get_mut(&id) {
                seq.output_tokens.push(token);
                seq.position += 1;
                self.stats.tokens_generated += 1;

                // Check for completion
                if token == eos_token || seq.output_tokens.len() >= seq.max_tokens {
                    seq.status = SequenceStatus::Finished;
                }
            }
        }

        // Move finished sequences
        let finished_ids: Vec<SequenceId> = self.running.iter()
            .filter(|(_, s)| s.is_finished())
            .map(|(id, _)| *id)
            .collect();

        for id in finished_ids {
            if let Some(seq) = self.running.remove(&id) {
                self.free_sequence_cache(&seq);
                self.stats.sequences_completed += 1;
                self.finished.push(seq);
            }
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> &BatchingStats {
        &self.stats
    }

    /// Get number of running sequences.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Get number of pending sequences.
    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }

    /// Get number of finished sequences.
    pub fn finished_count(&self) -> usize {
        self.finished.len()
    }

    /// Clear finished sequences from memory.
    pub fn clear_finished(&mut self) -> Vec<Sequence> {
        std::mem::take(&mut self.finished)
    }
}

/// Batched inference state for multiple sequences.
pub struct BatchedInferenceState {
    /// Model configuration
    pub config: Config,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Hidden states for each sequence in batch: [batch, hidden_size]
    pub hidden_states: Array2<f32>,
    /// Positions for each sequence
    pub positions: Vec<usize>,
    /// Temperature for each sequence
    pub temperatures: Vec<f32>,
    /// Sequence IDs in current batch
    pub sequence_ids: Vec<SequenceId>,
    /// Output logits: [batch, vocab_size]
    pub logits: Array2<f32>,
}

impl BatchedInferenceState {
    /// Create a new batched inference state.
    pub fn new(config: Config, max_batch_size: usize) -> Self {
        Self {
            hidden_states: Array2::zeros((max_batch_size, config.hidden_size)),
            positions: vec![0; max_batch_size],
            temperatures: vec![1.0; max_batch_size],
            sequence_ids: Vec::with_capacity(max_batch_size),
            logits: Array2::zeros((max_batch_size, config.vocab_size)),
            config,
            max_batch_size,
        }
    }

    /// Set up the batch for processing.
    pub fn setup_batch(&mut self, sequences: &[&Sequence]) {
        self.sequence_ids.clear();
        
        for (i, seq) in sequences.iter().enumerate().take(self.max_batch_size) {
            self.positions[i] = seq.position;
            self.temperatures[i] = seq.temperature;
            self.sequence_ids.push(seq.id);
        }
    }

    /// Get the current batch size.
    pub fn batch_size(&self) -> usize {
        self.sequence_ids.len()
    }

    /// Get logits for a sequence in the batch.
    pub fn get_logits(&self, batch_idx: usize) -> Option<Array1<f32>> {
        if batch_idx < self.batch_size() {
            Some(self.logits.row(batch_idx).to_owned())
        } else {
            None
        }
    }
}

/// Result of a batched generation step.
#[derive(Debug)]
pub struct BatchStepResult {
    /// Generated tokens: (sequence_id, token)
    pub tokens: Vec<(SequenceId, u32)>,
    /// Sequences that finished in this step
    pub finished: Vec<SequenceId>,
    /// Number of sequences processed
    pub batch_size: usize,
}

/// High-level continuous batching engine.
pub struct ContinuousBatchingEngine {
    /// Scheduler
    scheduler: BatchScheduler,
    /// Batched state
    state: Option<BatchedInferenceState>,
    /// EOS token ID
    eos_token: u32,
}

impl ContinuousBatchingEngine {
    /// Create a new continuous batching engine.
    pub fn new(config: BatchingConfig, eos_token: u32) -> Self {
        Self {
            scheduler: BatchScheduler::new(config),
            state: None,
            eos_token,
        }
    }

    /// Initialize with model configuration.
    pub fn init(&mut self, model_config: &Config) {
        let max_batch_size = self.scheduler.config.max_batch_size;
        self.scheduler.init_cache(model_config);
        self.state = Some(BatchedInferenceState::new(model_config.clone(), max_batch_size));
    }

    /// Add a request to the engine.
    pub fn add_request(
        &mut self,
        input_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
    ) -> SequenceId {
        self.scheduler.add_sequence(input_tokens, max_tokens, temperature)
    }

    /// Add a priority request to the engine.
    pub fn add_priority_request(
        &mut self,
        input_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        priority: i32,
    ) -> SequenceId {
        self.scheduler.add_sequence_with_priority(input_tokens, max_tokens, temperature, priority)
    }

    /// Cancel a request.
    pub fn cancel(&mut self, id: SequenceId) -> bool {
        self.scheduler.cancel_sequence(id)
    }

    /// Check if a request is finished.
    pub fn is_finished(&self, id: SequenceId) -> bool {
        matches!(self.scheduler.get_status(id), Some(SequenceStatus::Finished) | Some(SequenceStatus::Cancelled))
    }

    /// Get output tokens for a finished request.
    pub fn get_output(&self, id: SequenceId) -> Option<&[u32]> {
        self.scheduler.get_output(id)
    }

    /// Get engine statistics.
    pub fn stats(&self) -> &BatchingStats {
        self.scheduler.stats()
    }

    /// Get number of active sequences.
    pub fn active_count(&self) -> usize {
        self.scheduler.running_count() + self.scheduler.pending_count()
    }

    /// Check if the engine has work to do.
    pub fn has_work(&self) -> bool {
        self.scheduler.running_count() > 0 || self.scheduler.pending_count() > 0
    }

    /// Process outputs after a batch step.
    pub fn process_outputs(&mut self, outputs: Vec<(SequenceId, u32)>) {
        self.scheduler.process_outputs(outputs, self.eos_token);
    }

    /// Get the next batch to process.
    pub fn next_batch(&mut self) -> Vec<SequenceId> {
        self.scheduler.schedule()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config() -> Config {
        Config {
            hidden_size: 256,
            intermediate_size: 512,
            n_layers: 4,
            n_heads: 8,
            n_kv_heads: 4,
            vocab_size: 1000,
            max_position_embeddings: 2048,
            sliding_window: 4096,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            act_type: "silu".to_string(),
            quant: "f32".to_string(),
        }
    }

    #[test]
    fn test_sequence_creation() {
        let seq = Sequence::new(0, vec![1, 2, 3], 10, 0.7);
        
        assert_eq!(seq.id, 0);
        assert_eq!(seq.status, SequenceStatus::Pending);
        assert_eq!(seq.total_length(), 3);
        assert!(!seq.is_finished());
    }

    #[test]
    fn test_scheduler_add_sequence() {
        let config = BatchingConfig::default();
        let mut scheduler = BatchScheduler::new(config);

        let id1 = scheduler.add_sequence(vec![1, 2, 3], 10, 0.7);
        let id2 = scheduler.add_sequence(vec![4, 5], 20, 0.5);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(scheduler.pending_count(), 2);
    }

    #[test]
    fn test_scheduler_schedule() {
        let config = BatchingConfig::default();
        let mut scheduler = BatchScheduler::new(config);
        scheduler.init_cache(&make_test_config());

        scheduler.add_sequence(vec![1, 2, 3], 10, 0.7);
        scheduler.add_sequence(vec![4, 5], 10, 0.5);

        let batch = scheduler.schedule();
        
        assert_eq!(batch.len(), 2);
        assert_eq!(scheduler.running_count(), 2);
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_scheduler_process_outputs() {
        let config = BatchingConfig::default();
        let mut scheduler = BatchScheduler::new(config);
        scheduler.init_cache(&make_test_config());

        let id = scheduler.add_sequence(vec![1, 2, 3], 10, 0.7);
        scheduler.schedule();

        // Process some outputs
        scheduler.process_outputs(vec![(id, 100)], 0);
        
        assert_eq!(scheduler.stats().tokens_generated, 1);

        // Process EOS
        scheduler.process_outputs(vec![(id, 0)], 0);
        
        assert_eq!(scheduler.finished_count(), 1);
        assert_eq!(scheduler.running_count(), 0);
    }

    #[test]
    fn test_scheduler_priority() {
        let config = BatchingConfig::default();
        let mut scheduler = BatchScheduler::new(config);

        scheduler.add_sequence_with_priority(vec![1], 10, 0.7, 0);
        scheduler.add_sequence_with_priority(vec![2], 10, 0.7, 10); // Higher priority
        scheduler.add_sequence_with_priority(vec![3], 10, 0.7, 5);

        // High priority should be first in queue
        let first = scheduler.pending_queue.front().unwrap();
        assert_eq!(first.input_tokens[0], 2);
    }

    #[test]
    fn test_kv_cache_pool() {
        let model_config = make_test_config();
        let batching_config = BatchingConfig::default();
        
        let mut pool = KVCachePool::new(&model_config, &batching_config);

        // Test allocation
        let block = pool.allocate_block();
        assert!(block.is_some());
        assert!(pool.allocated_block_count() > 0);

        // Test freeing
        pool.free_block(block.unwrap());
        assert_eq!(pool.allocated_block_count(), 0);

        // Test can_allocate
        assert!(pool.can_allocate(100));
    }

    #[test]
    fn test_batched_inference_state() {
        let config = make_test_config();
        let state = BatchedInferenceState::new(config, 8);

        assert_eq!(state.max_batch_size, 8);
        assert_eq!(state.batch_size(), 0);
    }

    #[test]
    fn test_continuous_batching_engine() {
        let config = BatchingConfig::default();
        let mut engine = ContinuousBatchingEngine::new(config, 0);
        engine.init(&make_test_config());

        let id1 = engine.add_request(vec![1, 2, 3], 10, 0.7);
        let id2 = engine.add_request(vec![4, 5], 10, 0.5);

        assert_eq!(engine.active_count(), 2);
        assert!(!engine.is_finished(id1));
        assert!(!engine.is_finished(id2));
        assert!(engine.has_work());

        let batch = engine.next_batch();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_batching_stats() {
        let mut stats = BatchingStats::default();
        
        stats.update_avg_batch_size(4);
        stats.iterations = 1;
        assert!((stats.avg_batch_size - 4.0).abs() < 1e-6);
        
        stats.update_avg_batch_size(8);
        assert!((stats.avg_batch_size - 6.0).abs() < 1e-6);
    }
}
