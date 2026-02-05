# Phase 4: Algorithmic Optimizations

This document describes the Phase 4 algorithmic optimizations implemented in torchless-rs.

## Overview

Phase 4 focuses on algorithmic improvements that provide significant performance gains:

| Feature | Impact | Status |
|---------|--------|--------|
| Flash Attention | ~50% memory reduction for long sequences | ✅ Complete |
| Speculative Decoding | 2-3x generation speedup | ✅ Complete |
| Continuous Batching | 2-5x throughput for server workloads | ✅ Complete |

---

## 1. Flash Attention

**File:** `src/model/modules/flash_attention.rs`

### Description

Flash Attention is a memory-efficient attention algorithm that achieves O(N) memory complexity instead of the standard O(N²). It computes attention scores in tiles using online softmax normalization, avoiding the need to materialize the full attention matrix.

### Key Components

#### `FlashAttentionConfig`

Configuration for flash attention behavior:

```rust
use torchless::FlashAttentionConfig;

// Default configuration
let config = FlashAttentionConfig::default();

// Memory-optimized (smaller tiles, sequential processing)
let config = FlashAttentionConfig::memory_optimized();

// Speed-optimized (larger tiles, parallel processing)
let config = FlashAttentionConfig::speed_optimized();
```

Configuration options:
- `tile_size` - Size of tiles for tiled computation (default: 64)
- `threshold` - Minimum sequence length to use flash attention (default: 128)
- `parallel` - Whether to use parallel head computation

#### Core Functions

```rust
use torchless::{flash_attention_single_head, flash_attention_multi_head, flash_attention_into};

// Single head attention
let output = flash_attention_single_head(
    &query,           // [head_dim]
    k_cache.view(),   // [seq_len, head_dim]
    v_cache.view(),   // [seq_len, head_dim]
    seq_len,
    scale,
    &config,
);

// Multi-head attention with GQA support
let output = flash_attention_multi_head(
    &q_state,         // [n_heads, head_dim]
    k_cache.view(),   // [seq_len, head_dim]
    v_cache.view(),   // [seq_len, head_dim]
    n_heads,
    n_kv_heads,
    seq_len,
    &config,
);

// Zero-allocation version (writes to pre-allocated buffers)
flash_attention_into(
    &query,
    k_cache.view(),
    v_cache.view(),
    &mut output,
    seq_len,
    scale,
    &config,
    &mut scores_buffer,
    &mut values_buffer,
);
```

#### Memory Estimation

```rust
use torchless::flash_attention_estimate_memory;

let (flash_bytes, standard_bytes) = flash_attention_estimate_memory(
    seq_len,    // 1024
    n_heads,    // 32
    head_dim,   // 128
    tile_size,  // 64
);

println!("Flash: {} bytes, Standard: {} bytes", flash_bytes, standard_bytes);
// Flash uses significantly less memory for long sequences
```

### Algorithm

Flash Attention uses the online softmax algorithm:

1. **Tiled Processing**: Process K/V cache in fixed-size tiles
2. **Running Statistics**: Maintain running max and sum for softmax
3. **Rescaling**: Rescale previous contributions when max changes
4. **Accumulation**: Accumulate weighted values incrementally
5. **Normalization**: Final division by total sum

This avoids materializing the full `[n_heads, seq_len]` attention score matrix.

### Performance Characteristics

- **Memory**: O(tile_size × head_dim) per head instead of O(seq_len × n_heads)
- **Speed**: Comparable to standard attention, often faster due to better cache locality
- **Accuracy**: Mathematically equivalent to standard attention (within floating point precision)

---

## 2. Speculative Decoding

**File:** `src/model/speculative.rs`

### Description

Speculative decoding accelerates text generation by using a smaller, faster "draft" model to propose multiple candidate tokens, then verifying them in parallel with the main model. This can achieve 2-3x speedup for well-matched model pairs.

### Key Components

#### `SpeculativeConfig`

```rust
use torchless::SpeculativeConfig;

// Default configuration
let config = SpeculativeConfig::default();

// Quality-focused (conservative speculation)
let config = SpeculativeConfig::quality();

// Speed-focused (aggressive speculation)
let config = SpeculativeConfig::speed();
```

Configuration options:
- `speculation_length` - Number of tokens to speculate per iteration (default: 5)
- `temperature` - Sampling temperature (default: 0.7)
- `adaptive` - Enable adaptive speculation length (default: true)
- `min_speculation` / `max_speculation` - Bounds for adaptive mode

#### `SpeculativeDecoder`

Two-model speculative decoding:

```rust
use torchless::{SpeculativeDecoder, SpeculativeConfig, InferenceState};

let config = SpeculativeConfig::default();

let mut decoder = SpeculativeDecoder::new(
    |state, token| main_model.forward(state, token, false),
    |state, token| draft_model.forward(state, token, false),
    config,
);

// Generate tokens
let accepted_tokens = decoder.generate_step(
    &mut main_state,
    &mut draft_state,
    context_token,
);

// Check statistics
let stats = decoder.stats();
println!("Acceptance rate: {:.1}%", stats.acceptance_rate() * 100.0);
println!("Tokens per iteration: {:.2}", stats.tokens_per_iteration());
```

#### `SelfSpeculativeDecoder`

Single-model speculation using temperature difference:

```rust
use torchless::SelfSpeculativeDecoder;

let mut decoder = SelfSpeculativeDecoder::new(
    |state, token| model.forward(state, token, false),
    config,
    1.5,  // Draft temperature (higher = faster, less accurate)
    0.7,  // Main temperature
);

let tokens = decoder.generate_step(&mut state, context_token);
```

#### `LookaheadDecoder`

Generate multiple candidate tokens:

```rust
use torchless::LookaheadDecoder;

let decoder = LookaheadDecoder::new(4, 1.0);
let candidates = decoder.generate_candidates(&logits, 5); // Top 5 candidates
```

#### `TokenBuffer`

Manage committed and pending tokens with rollback support:

```rust
use torchless::TokenBuffer;

let mut buffer = TokenBuffer::new();
buffer.commit(token1);
buffer.add_pending(&[token2, token3, token4]);
buffer.accept(2);  // Accept first 2 pending
buffer.reject_all();  // Or reject all pending
```

### Algorithm

1. **Draft Phase**: Draft model generates K candidate tokens autoregressively
2. **Verify Phase**: Main model processes all K+1 positions
3. **Accept/Reject**: Use rejection sampling to accept tokens where `P_main(x) / P_draft(x) > random()`
4. **Correction**: On rejection, sample from adjusted distribution `max(0, P_main - P_draft)`
5. **Bonus Token**: If all accepted, sample one more from main model

### Performance Characteristics

- **Speedup**: 2-3x for well-matched draft/main model pairs
- **Quality**: Mathematically equivalent to main model alone
- **Overhead**: Requires running draft model (should be much smaller than main)

---

## 3. Continuous Batching

**File:** `src/model/batching.rs`

### Description

Continuous batching enables efficient processing of multiple sequences simultaneously. Unlike static batching, sequences can dynamically join and leave the batch as they complete or new requests arrive.

### Key Components

#### `BatchingConfig`

```rust
use torchless::BatchingConfig;

// Default configuration
let config = BatchingConfig::default();

// Low memory systems
let config = BatchingConfig::low_memory();

// High throughput servers
let config = BatchingConfig::high_throughput();
```

Configuration options:
- `max_batch_size` - Maximum sequences in a batch (default: 32)
- `max_batch_tokens` - Maximum total tokens across all sequences (default: 4096)
- `max_seq_len` - Maximum sequence length (default: 2048)
- `num_cache_blocks` - Number of KV cache blocks (default: 256)
- `block_size` - Tokens per cache block (default: 16)
- `enable_preemption` - Enable fair scheduling (default: true)

#### `ContinuousBatchingEngine`

High-level API for production serving:

```rust
use torchless::{ContinuousBatchingEngine, BatchingConfig, Config};

let config = BatchingConfig::default();
let eos_token = 2;  // End-of-sequence token ID

let mut engine = ContinuousBatchingEngine::new(config, eos_token);
engine.init(&model_config);

// Add requests
let id1 = engine.add_request(vec![1, 2, 3], 100, 0.7);
let id2 = engine.add_priority_request(vec![4, 5], 50, 0.5, 10);

// Processing loop
while engine.has_work() {
    let batch = engine.next_batch();
    
    // Run model forward pass for batch...
    let outputs: Vec<(SequenceId, u32)> = /* model outputs */;
    
    engine.process_outputs(outputs);
}

// Get results
if engine.is_finished(id1) {
    let output = engine.get_output(id1).unwrap();
}
```

#### `BatchScheduler`

Lower-level scheduler for custom implementations:

```rust
use torchless::{BatchScheduler, BatchingConfig};

let mut scheduler = BatchScheduler::new(config);
scheduler.init_cache(&model_config);

// Add sequences
let id = scheduler.add_sequence(tokens, max_tokens, temperature);

// Schedule batch
let batch_ids = scheduler.schedule();

// Process outputs
scheduler.process_outputs(outputs, eos_token);

// Get statistics
let stats = scheduler.stats();
println!("Completed: {}", stats.sequences_completed);
println!("Avg batch size: {:.1}", stats.avg_batch_size);
```

#### `KVCachePool`

Block-based KV cache management (PagedAttention-style):

```rust
use torchless::KVCachePool;

let mut pool = KVCachePool::new(&model_config, &batching_config);

// Allocate blocks
if pool.can_allocate(100) {
    let block_idx = pool.allocate_block().unwrap();
    
    // Write KV state
    pool.write_kv(block_idx, layer_idx, position_in_block, &k_state, &v_state);
    
    // Read KV state
    let (k, v) = pool.read_kv(block_idx, layer_idx, position_in_block);
    
    // Free when done
    pool.free_block(block_idx);
}
```

#### `Sequence` and `SequenceStatus`

```rust
use torchless::{Sequence, SequenceStatus};

let seq = Sequence::new(id, input_tokens, max_tokens, temperature);

match seq.status {
    SequenceStatus::Pending => { /* waiting */ }
    SequenceStatus::Running => { /* actively generating */ }
    SequenceStatus::Finished => { /* completed */ }
    SequenceStatus::Preempted => { /* paused for fairness */ }
    SequenceStatus::Cancelled => { /* user cancelled */ }
}
```

### Features

- **Dynamic batching**: Sequences join/leave at any time
- **Priority scheduling**: Higher priority sequences processed first
- **Preemption**: Long-running sequences can be paused for fairness
- **Memory efficiency**: Block-based KV cache allocation
- **Statistics**: Throughput, batch size, completion metrics

### Performance Characteristics

- **Throughput**: 2-5x improvement for server workloads with varying sequence lengths
- **Latency**: Lower average latency due to better scheduling
- **Memory**: More efficient utilization through dynamic allocation

---

## Public API Summary

All Phase 4 components are exported from the library root:

```rust
// Flash Attention
use torchless::{
    FlashAttentionConfig,
    flash_attention_single_head,
    flash_attention_multi_head,
    flash_attention_into,
    flash_attention_estimate_memory,
    FLASH_TILE_SIZE,
    FLASH_ATTENTION_THRESHOLD,
};

#[cfg(feature = "parallel")]
use torchless::flash_attention_parallel;

// Speculative Decoding
use torchless::{
    SpeculativeConfig,
    SpeculativeStats,
    SpeculativeDecoder,
    SelfSpeculativeDecoder,
    LookaheadDecoder,
    TokenBuffer,
    CacheState,
    SpeculativeModel,
};

// Continuous Batching
use torchless::{
    BatchingConfig,
    BatchingStats,
    BatchScheduler,
    BatchedInferenceState,
    BatchStepResult,
    ContinuousBatchingEngine,
    KVCachePool,
    Sequence,
    SequenceId,
    SequenceStatus,
};
```

---

## Testing

All components include comprehensive unit tests:

```bash
# Run all Phase 4 tests
cargo test --lib flash_attention
cargo test --lib speculative
cargo test --lib batching

# Run all library tests
cargo test --lib
```

Test coverage includes:
- Flash attention correctness vs standard attention
- Online softmax state management
- Memory estimation accuracy
- Speculative decoding acceptance/rejection
- Softmax temperature scaling
- Token buffer rollback
- Batch scheduler priority ordering
- KV cache pool allocation
- Continuous batching engine workflow
