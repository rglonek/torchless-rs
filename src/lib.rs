pub mod kernels;
pub mod loader;
pub mod memory;
pub mod model;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

// Core loader types
pub use loader::{Config, Parameters, TensorDtype, TensorView};

// Quantization support (Phase 2)
pub use loader::{
    QuantFormat, QuantizedTensor,
    Q4_0Block, Q8_0Block, Q4KMBlock, Q4KSBlock,
    QK4_0, QK8_0, QK_K,
};

// Model types
pub use model::{ArenaInferenceState, InferenceState, LazyMistral, Mistral};

// Sampler functions
pub use sampler::{generate, generate_lazy, sample_greedy, sample_multinomial};

// Backend abstraction (Phase 1 foundation)
pub use kernels::backend::{
    Backend, BackendPreference, CpuBackend, KernelBackend, 
    init_backend, default_backend,
};

// CUDA backend (Phase 5)
#[cfg(feature = "cuda")]
pub use kernels::cuda::{
    CudaBackend, CudaTensor, CudaMemoryPool,
    memory::{MemoryPoolStats, PinnedBuffer, estimate_model_memory_mb, estimate_kv_cache_memory_mb},
};

// ROCm backend (Phase 5) - AMD GPU support
#[cfg(feature = "rocm")]
pub use kernels::rocm::{
    RocmBackend, RocmTensor, RocmMemoryPool,
    memory::{
        MemoryPoolStats as RocmMemoryPoolStats,
        PinnedBuffer as RocmPinnedBuffer,
        estimate_model_memory_mb as rocm_estimate_model_memory_mb,
        estimate_kv_cache_memory_mb as rocm_estimate_kv_cache_memory_mb,
    },
};

// Metal backend (Phase 5) - Apple Silicon GPU support
#[cfg(feature = "metal-gpu")]
pub use kernels::metal::{
    MetalBackend, MetalTensor, MetalMemoryPool,
    memory::{
        MetalMemoryPoolStats,
        estimate_model_memory_mb as metal_estimate_model_memory_mb,
        estimate_kv_cache_memory_mb as metal_estimate_kv_cache_memory_mb,
        get_device_memory_size as metal_get_device_memory_size,
    },
};

// Tensor storage abstraction (Phase 1 foundation)
pub use tensor::{
    Device, Dtype, TensorStorage, UnifiedTensor, DeviceTransfer,
    MixedPrecisionConfig, ModelSizeParams,
};

// Memory optimizations (Phase 3)
pub use memory::{
    AlignedBuffer, InferenceArena, 
    CACHE_LINE_SIZE, CACHE_LINE_F32S, SIMD_ALIGNMENT,
    prefetch_read, prefetch_write, prefetch_sequential,
    unchecked as memory_unchecked,
    sum_squares_unchecked, rmsnorm_unchecked, max_unchecked,
    softmax_unchecked, silu_unchecked,
};

// =============================================================================
// Phase 4: Algorithmic Optimizations
// =============================================================================

// Flash Attention - Memory-efficient attention with O(N) memory complexity
pub use model::{
    FlashAttentionConfig,
    flash_attention_single_head, flash_attention_multi_head, flash_attention_into,
    flash_attention_estimate_memory,
    FLASH_TILE_SIZE, FLASH_ATTENTION_THRESHOLD,
};

#[cfg(feature = "parallel")]
pub use model::flash_attention_parallel;

// Speculative Decoding - Accelerated generation using draft model speculation
pub use model::{
    SpeculativeConfig, SpeculativeStats, SpeculativeDecoder,
    SelfSpeculativeDecoder, LookaheadDecoder, TokenBuffer,
    CacheState, SpeculativeModel,
};

// Continuous Batching - Efficient multi-sequence processing
pub use model::{
    BatchingConfig, BatchingStats, BatchScheduler,
    BatchedInferenceState, BatchStepResult, ContinuousBatchingEngine,
    KVCachePool, Sequence, SequenceId, SequenceStatus,
};
