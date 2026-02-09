# Future Parameters

This document catalogs library-level configuration that is **not yet exposed** via the CLI. Each entry describes what the configuration controls, whether the library fully implements it or only defines the config struct, and a suggested CLI flag for future exposure.

For parameters that **are** exposed via the CLI today, see [Parameter Reference](params.md).

---

## Flash Attention

**Library type:** `FlashAttentionConfig` (in `src/model/modules/flash_attention.rs`)
**Status:** Fully implemented. Used automatically when sequence length exceeds the threshold.

Flash attention reduces memory usage from O(N^2) to O(N) for the attention mechanism by processing attention in tiles rather than materializing the full attention matrix.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tile_size` | `usize` | 64 | Number of tokens processed per tile. Larger tiles are faster but use more memory. |
| `threshold` | `usize` | 128 | Minimum sequence length to activate flash attention. Below this, standard attention is used. |
| `parallel` | `bool` | `true` (if `parallel` feature) | Whether to process tiles in parallel using Rayon. |

**Suggested CLI flags:**

- `--flash-tile-size <N>` -- Override tile size (default: 64).
- `--no-flash-attention` -- Disable flash attention entirely.

**Presets:** The library defines `FLASH_TILE_SIZE` (64) and `FLASH_ATTENTION_THRESHOLD` (128) as constants. A `--flash-preset` flag could select from small/default/large tile configurations.

---

## Speculative Decoding (Two-Model)

**Library type:** `SpeculativeDecoder` (in `src/model/speculative.rs`)
**Status:** Fully implemented. Requires a separate draft model.

The two-model speculative decoder uses a smaller, faster draft model to propose tokens and the main model to verify them. This is more effective than self-speculative decoding when a good draft model is available (e.g., a 1B model drafting for a 7B model).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `speculation_length` | `usize` | 5 | Tokens proposed per iteration. |
| `temperature` | `f32` | 0.7 | Sampling temperature for both models. |
| `min_p_ratio` | `f32` | 0.0 | Minimum probability ratio for acceptance (rejection sampling threshold). |
| `adaptive` | `bool` | `true` | Dynamically adjust speculation length based on acceptance rate. |
| `min_speculation` | `usize` | 2 | Minimum speculation length when adaptive. |
| `max_speculation` | `usize` | 8 | Maximum speculation length when adaptive. |

The CLI currently exposes `--speculative` for **self-speculative** decoding only (same model, different temperatures). Two-model speculative decoding is not exposed.

**Suggested CLI flags:**

- `--draft-model <PATH>` -- Path to the draft model. Implies speculative decoding.
- `--speculation-length <N>` -- Override the number of draft tokens per iteration.
- `--no-adaptive-speculation` -- Disable adaptive speculation length.

---

## Continuous Batching

**Library type:** `BatchingConfig` (in `src/model/batching.rs`)
**Status:** Fully implemented. Designed for server/batch workloads.

Continuous batching processes multiple sequences simultaneously with dynamic scheduling, preemption, and paged KV cache management. This is primarily useful for serving (handling multiple concurrent requests), not interactive single-user CLI usage.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_batch_size` | `usize` | 32 | Maximum sequences in a batch. |
| `max_batch_tokens` | `usize` | 4096 | Maximum total tokens across all sequences. |
| `max_seq_len` | `usize` | 2048 | Maximum sequence length supported. |
| `num_cache_blocks` | `usize` | 256 | Number of KV cache blocks in the pool. |
| `block_size` | `usize` | 16 | Tokens per cache block. |
| `enable_preemption` | `bool` | `true` | Allow preempting long-running sequences for fairness. |
| `preemption_threshold` | `f64` | 30.0 | Seconds before a sequence is eligible for preemption. |

**Suggested CLI flags:** Batching is server-oriented and unlikely to be exposed in the single-user CLI. A future `torchless serve` subcommand could expose:

- `--batch-size <N>` -- Maximum concurrent sequences.
- `--batch-tokens <N>` -- Maximum total tokens per batch step.
- `--cache-blocks <N>` -- Number of KV cache blocks.

---

## Parallelization

The library provides several parallelization strategies, all gated behind the `parallel` Cargo feature. None are currently exposed via CLI flags.

### Work Distribution

**Library type:** `WorkDistributionConfig` (in `src/kernels/parallel.rs`)
**Status:** Fully implemented. Controls how matrix operations are split across CPU threads.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_chunk_size` | `usize` | 64 | Minimum elements per task (prevents excessive overhead). |
| `max_chunk_size` | `usize` | 4096 | Maximum elements per task (ensures load balancing). |
| `tasks_per_thread` | `usize` | 4 | Target tasks per thread for work stealing. |
| `numa_aware` | `bool` | `false` | Enable NUMA-aware allocation hints. |
| `cache_line_size` | `usize` | 64 | Cache line size in bytes for alignment. |

### Pipeline Parallelism

**Library type:** `PipelineConfig` (in `src/kernels/parallel.rs`)
**Status:** Fully implemented. Splits model layers across pipeline stages.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_micro_batches` | `usize` | 4 | Number of micro-batches in the pipeline. |
| `async_execution` | `bool` | `false` | Whether to use asynchronous stage execution. |
| `memory_budget` | `usize` | 512 MB | Memory budget for pipeline buffers. |

### Tensor Parallelism

**Library type:** `TensorParallelConfig` (in `src/kernels/parallel.rs`)
**Status:** Fully implemented. Shards weight matrices across devices/threads.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `world_size` | `usize` | 1 | Number of parallel partitions. |
| `rank` | `usize` | 0 | This device's rank (0..world_size). |
| `attention_qkv_strategy` | `TensorParallelStrategy` | `None` | Strategy for attention QKV projections. |
| `attention_output_strategy` | `TensorParallelStrategy` | `None` | Strategy for attention output projection. |
| `mlp_gate_up_strategy` | `TensorParallelStrategy` | `None` | Strategy for MLP gate/up projections. |
| `mlp_down_strategy` | `TensorParallelStrategy` | `None` | Strategy for MLP down projection. |

### Unified Parallel Config

**Library type:** `ParallelConfig` (in `src/model/parallel.rs`)
**Status:** Fully implemented. Combines all parallelization options.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pipeline` | `Option<PipelineConfig>` | `None` | Pipeline parallelism configuration. |
| `tensor_parallel` | `Option<TensorParallelConfig>` | `None` | Tensor parallelism configuration. |
| `work_distribution` | `WorkDistributionConfig` | (see above) | Work distribution for matmul/attention. |
| `adaptive_matmul` | `bool` | `true` | Use adaptive chunking for matrix operations. |
| `parallel_heads` | `bool` | `true` | Parallelize across attention heads. |
| `layers_per_stage` | `usize` | 4 | Layers grouped per pipeline stage. |

**Suggested CLI flags:**

- `--parallel-mode <MODE>` -- Select parallelization mode: `auto`, `pipeline`, `tensor`, `none`.
- `--num-threads <N>` -- Override the number of worker threads (currently auto-detected).
- `--layers-per-stage <N>` -- Override pipeline stage size.
- `--numa` -- Enable NUMA-aware thread placement.

---

## Mixed Precision

**Library type:** `MixedPrecisionConfig` (in `src/tensor/storage.rs`)
**Status:** Config struct defined with presets. Not yet wired into the compute path -- the model always uses the precision of its stored weights. This is aspirational infrastructure for future optimization.

| Field | Type | Default (`balanced`) | Description |
|-------|------|----------------------|-------------|
| `weights_dtype` | `Dtype` | `Int8` | Data type for dense layer weights. |
| `embedding_dtype` | `Dtype` | `F16` | Data type for embedding weights. |
| `activation_dtype` | `Dtype` | `F16` | Data type for intermediate activations. |
| `kv_cache_dtype` | `Dtype` | `F16` | Data type for KV cache storage. |
| `attention_dtype` | `Dtype` | `F32` | Data type for attention score computation. |
| `output_dtype` | `Dtype` | `F32` | Data type for output logits. |

The library provides named presets: `full_precision()` (all F32), `balanced()` (default), `memory_optimized()` (aggressive quantization), and `gpu_optimized()` (F16/INT4).

**Suggested CLI flags:**

- `--precision <PRESET>` -- Select a mixed-precision preset: `full`, `balanced`, `memory`, `gpu`.
- `--kv-cache-dtype <DTYPE>` -- Override KV cache precision (e.g., `f16`, `f32`).

---

## GPU Memory Management

**Library type:** `MemoryConfig` (in `src/kernels/gpu_memory.rs`)
**Status:** Fully implemented. Controls GPU memory allocation, pooling, and CPU fallback.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_memory_fraction` | `f64` | 0.9 | Fraction of GPU memory to use (0.0-1.0). |
| `cpu_fallback_enabled` | `bool` | `true` | Fall back to CPU when GPU memory is exhausted. |
| `min_free_bytes` | `usize` | 512 MB | Minimum free GPU memory to maintain. |
| `track_allocations` | `bool` | `false` (release) | Track individual buffer allocations (debug overhead). |
| `max_pooled_per_bucket` | `usize` | 8 | Maximum buffers per size class in the memory pool. |
| `pre_allocate` | `bool` | `false` | Pre-allocate common buffer sizes at startup. |
| `pre_allocate_sizes` | `Vec<usize>` | `[]` | Sizes to pre-allocate if enabled. |

**Suggested CLI flags:**

- `--gpu-memory-fraction <F>` -- Fraction of GPU memory to use (default: 0.9).
- `--gpu-min-free <BYTES>` -- Minimum free GPU memory to maintain.
- `--no-cpu-fallback` -- Disable CPU fallback when GPU memory is exhausted.

---

## Architecture Override

**Library API:** `DynamicModel::load_with_arch(params, architecture)`
**Status:** Fully implemented. The CLI auto-detects architecture from model metadata.

The library's `DynamicModel` auto-detects the model architecture (Mistral, LLaMA, Phi, Gemma, Qwen) from config metadata and tensor naming patterns. `load_with_arch()` allows overriding this detection to force a specific architecture, which is useful when metadata is missing or incorrect.

**Suggested CLI flag:**

- `--arch <ARCH>` -- Force architecture: `mistral`, `llama`, `phi`, `gemma`, `qwen`. Overrides auto-detection.

---

## Model Format Override

**Library API:** `detect_format()`, `load_model_auto()`
**Status:** Fully implemented. The CLI auto-detects format from file extension and magic bytes.

The loader supports GGUF (`.gguf`), Safetensors (`.safetensors`), and native binary (`.bin`) formats. Format detection is automatic. The library API allows forcing a specific format via direct loader calls (`GGUFLoader`, `SafetensorsLoader`), but the CLI does not expose this.

**Suggested CLI flag:**

- `--format <FORMAT>` -- Force model format: `gguf`, `safetensors`, `bin`. Overrides auto-detection.

---

## Arena Inference State

**Library type:** `ArenaInferenceState` (in `src/model/mod.rs`)
**Status:** Fully implemented. Provides faster allocation for inference temporaries.

`ArenaInferenceState` wraps the standard `InferenceState` with an arena allocator and pre-allocated scratch buffers for MLP and attention computations. This avoids repeated heap allocations during the forward pass, improving throughput by 5-15% on CPU.

| Field | Type | Description |
|-------|------|-------------|
| `inner` | `InferenceState` | The underlying inference state (KV cache, logits, etc.). |
| `arena` | `InferenceArena` | Arena allocator for temporary forward-pass allocations. |
| `mlp_scratch` | `AlignedBuffer<f32>` | Cache-aligned scratch buffer for MLP intermediates. |
| `attention_scratch` | `AlignedBuffer<f32>` | Cache-aligned scratch buffer for attention scores. |

**Suggested CLI flag:**

- `--arena` -- Use arena-backed inference state for faster allocation. Default could be auto-detected based on model size.
