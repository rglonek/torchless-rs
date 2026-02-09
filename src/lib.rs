pub mod chat;
pub mod coding;
pub mod kernels;
pub mod loader;
pub mod memory;
pub mod model;
pub mod sampler;
pub mod tensor;
pub mod tokenizer;

// Chat template support
pub use chat::{
    display_thinking_token, display_thinking_token_to, strip_thinking, ANSI_DIM, ANSI_RESET,
};
pub use chat::{ChatMessage, ChatRole, ChatTemplate, ThinkingState, TokenAction};

// Coding mode support
pub use coding::{
    apply_edit, check_path_blocked, coding_system_prompt, expand_file_references, format_edit_diff,
    parse_edit_blocks, parse_file_references, FileReference, PendingEdit,
};

// Core loader types
pub use loader::{Config, Parameters, TensorDtype, TensorView};

// Quantization support (Phase 2)
pub use loader::{
    Q4KMBlock, Q4KSBlock, Q4_0Block, Q8_0Block, QuantFormat, QuantizedTensor, QK4_0, QK8_0, QK_K,
};

// =============================================================================
// Phase 7: Multi-Format Support
// =============================================================================

// Format detection and unified loading
pub use loader::{detect_format, load_model_auto, ModelFormat, UnifiedConfig, UnifiedModelData};

// GGUF format (llama.cpp compatible)
pub use loader::{GGMLType, GGUFLoader, GGUFMetadata, GGUFTensorInfo};

// Safetensors format (HuggingFace compatible)
pub use loader::{SafetensorsLoader, SafetensorsTensorInfo};

// Additional Safetensors utilities
pub use loader::formats::safetensors::{
    load_with_config as load_safetensors_with_config, parse_hf_config,
};

// Model types
pub use model::{ArenaInferenceState, InferenceState, LazyMistral, Mistral};

// =============================================================================
// Phase 8: Multi-Architecture Support
// =============================================================================

// Architecture detection and configuration
pub use model::{
    detect_architecture, detect_architecture_from_config, detect_architecture_from_tensors,
    ActivationType, ArchitectureConfig, Model, ModelArchitecture, NormType, RopeScaling,
    TensorNamePattern,
};

// Additional model architectures
pub use model::{
    // Dynamic model for runtime architecture selection
    DynamicModel,
    // Gemma (Google) - Gemma 1/2 support
    Gemma,
    // LLaMA (Meta) - LLaMA 1/2/3 support
    LLaMA,
    LazyGemma,
    LazyLLaMA,
    LazyPhi,
    LazyQwen,
    ModelLoader,
    // Phi (Microsoft) - Phi-2, Phi-3 support
    Phi,
    // Qwen (Alibaba) - Qwen 1/2 support
    Qwen,
};

// Sampler functions
pub use sampler::{
    generate, generate_lazy, generate_lazy_until_eos, generate_until_eos, sample_greedy,
    sample_multinomial, sample_top_k, sample_top_p, sample_with_config, GenerationResult,
    SamplingConfig,
};

// Backend abstraction (Phase 1 foundation)
pub use kernels::backend::{
    default_backend, init_backend, Backend, BackendPreference, CpuBackend, KernelBackend,
};

// Backend discovery and device enumeration (Phase 5)
pub use kernels::backend::{
    best_available_backend, discover_backends, init_backend_with_memory_check,
    print_backend_summary, select_backend_for_model, BackendInfo, BackendType, DeviceInfo,
};

// Unified GPU memory management (Phase 5)
pub use kernels::gpu_memory::{
    compute_f16_bytes,
    compute_f32_bytes,
    estimate_inference_memory,
    estimate_kv_cache_memory,
    estimate_model_memory,
    round_up_power_of_2,
    size_class,
    AllocationTracker,
    // Buffer tracking
    BufferId,
    BufferLocation,
    BufferMetadata,
    // Device capabilities
    DeviceCapabilities,
    // Fallback buffer for GPU/CPU hybrid operation
    FallbackBuffer,
    GenericMemoryPool,
    // Memory pool trait and generic implementation
    GpuMemoryPool,
    // Memory estimation utilities
    InferenceMemoryEstimate,
    MemoryConfig,
    MemoryPressure,
    // Memory statistics and configuration
    MemoryStats as GpuMemoryStats,
};

// CUDA backend (Phase 5)
#[cfg(feature = "cuda")]
pub use kernels::cuda::{
    memory::{
        estimate_kv_cache_memory_mb, estimate_model_memory_mb, MemoryPoolStats, PinnedBuffer,
    },
    CudaBackend, CudaMemoryPool, CudaTensor,
};

// ROCm backend (Phase 5) - AMD GPU support
#[cfg(feature = "rocm")]
pub use kernels::rocm::{
    memory::{
        estimate_kv_cache_memory_mb as rocm_estimate_kv_cache_memory_mb,
        estimate_model_memory_mb as rocm_estimate_model_memory_mb,
        MemoryPoolStats as RocmMemoryPoolStats, PinnedBuffer as RocmPinnedBuffer,
    },
    RocmBackend, RocmMemoryPool, RocmTensor,
};

// Metal backend (Phase 5) - Apple Silicon GPU support
#[cfg(feature = "metal-gpu")]
pub use kernels::metal::{
    memory::{
        estimate_kv_cache_memory_mb as metal_estimate_kv_cache_memory_mb,
        estimate_model_memory_mb as metal_estimate_model_memory_mb,
        get_device_memory_size as metal_get_device_memory_size, MetalMemoryPoolStats,
    },
    MetalBackend, MetalMemoryPool, MetalTensor,
};

// WebGPU backend - Cross-platform GPU via wgpu
#[cfg(feature = "webgpu")]
pub use kernels::webgpu::{
    memory::{
        estimate_kv_cache_memory_mb as webgpu_estimate_kv_cache_memory_mb,
        estimate_model_memory_mb as webgpu_estimate_model_memory_mb, WebGPUMemoryPoolStats,
    },
    WebGPUBackend, WebGPUMemoryPool, WebGPUTensor,
};

// Tensor storage abstraction (Phase 1 foundation)
pub use tensor::{
    Device, DeviceTransfer, Dtype, MixedPrecisionConfig, ModelSizeParams, TensorStorage,
    UnifiedTensor,
};

// Memory optimizations (Phase 3)
pub use memory::{
    max_unchecked, prefetch_read, prefetch_sequential, prefetch_write, rmsnorm_unchecked,
    silu_unchecked, softmax_unchecked, sum_squares_unchecked, unchecked as memory_unchecked,
    AlignedBuffer, InferenceArena, CACHE_LINE_F32S, CACHE_LINE_SIZE, SIMD_ALIGNMENT,
};

// =============================================================================
// Phase 4: Algorithmic Optimizations
// =============================================================================

// Flash Attention - Memory-efficient attention with O(N) memory complexity
pub use model::{
    flash_attention_estimate_memory, flash_attention_into, flash_attention_multi_head,
    flash_attention_single_head, FlashAttentionConfig, FLASH_ATTENTION_THRESHOLD, FLASH_TILE_SIZE,
};

#[cfg(feature = "parallel")]
pub use model::flash_attention_parallel;

// Speculative Decoding - Accelerated generation using draft model speculation
pub use model::{
    CacheState, LookaheadDecoder, SelfSpeculativeDecoder, SpeculativeConfig, SpeculativeDecoder,
    SpeculativeModel, SpeculativeStats, TokenBuffer,
};

// Continuous Batching - Efficient multi-sequence processing
pub use model::{
    BatchScheduler, BatchStepResult, BatchedInferenceState, BatchingConfig, BatchingStats,
    ContinuousBatchingEngine, KVCachePool, Sequence, SequenceId, SequenceStatus,
};

// =============================================================================
// Phase 6: Parallelization Improvements
// =============================================================================

// Advanced parallelization utilities (work distribution, pipeline, tensor parallelism)
#[cfg(feature = "parallel")]
pub use kernels::parallel::{
    all_reduce_sum,
    all_reduce_sum_inplace,

    // Parallel Attention and MLP with adaptive work distribution
    attention_parallel_adaptive,
    column_parallel_linear,
    matmul_vec_adaptive,
    matmul_vec_adaptive_into,

    mlp_tensor_parallel,
    num_cpus,
    row_parallel_linear,
    NumaHint,
    PipelineConfig,

    // Pipeline Parallelism (6.2)
    PipelineState,
    TensorParallelConfig,
    // Tensor Parallelism (6.3)
    TensorParallelStrategy,
    // Work Distribution (6.1)
    WorkDistributionConfig,
    WorkStealingStats,
};

// Parallel model implementations
#[cfg(feature = "parallel")]
pub use model::{
    ParallelAttention, ParallelConfig, ParallelInferenceState, ParallelLayer, ParallelMLP,
    PipelineParallelMistral, TensorParallelMistral,
};
