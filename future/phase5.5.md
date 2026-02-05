# Phase 5.5: Unified GPU Memory Management & Backend Selection

**Status:** ✅ Implemented  
**Impact:** Foundation for all GPU backends, enables CPU fallback, improves debugging  
**Platform:** All (cross-platform)

## Overview

This phase implements a unified GPU memory management system and enhanced backend selection that works across all GPU backends (CUDA, ROCm, Metal, OpenCL). It provides:

- **Unified Memory Pool Interface**: Common trait for all GPU memory pools
- **Buffer Tracking**: Debug allocations with unique IDs and metadata
- **Memory Pressure Monitoring**: Detect and respond to low memory conditions
- **Automatic CPU Fallback**: Gracefully fall back to CPU when GPU memory is exhausted
- **Backend Discovery**: Enumerate available backends and their devices
- **Memory-Aware Initialization**: Select backends based on available memory

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GpuMemoryManager                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ MemoryConfig │  │MemoryStats   │  │MemoryPressure│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│          │                │                 │                   │
│          └────────────────┼─────────────────┘                   │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              GpuMemoryPool Trait                         │   │
│  │  - backend_name()     - stats()        - pressure()     │   │
│  │  - total_memory()     - free_memory()  - clear_pool()   │   │
│  └─────────────────────────────────────────────────────────┘   │
│          │                │                │                │   │
│          ▼                ▼                ▼                ▼   │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────┐ │
│  │ CudaPool  │    │ RocmPool  │    │MetalPool  │    │OpenCL │ │
│  └───────────┘    └───────────┘    └───────────┘    └───────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Files Created

### `src/kernels/gpu_memory.rs`

Unified GPU memory management module containing:

#### Buffer Tracking

```rust
/// Unique identifier for buffer allocations
pub struct BufferId(pub u64);

/// Metadata for tracked allocations
pub struct BufferMetadata {
    pub id: BufferId,
    pub size_bytes: usize,
    pub num_elements: usize,
    pub location: BufferLocation,
    pub allocated_at: std::time::Instant,
    pub tag: Option<String>,
}

/// Where a buffer is located
pub enum BufferLocation {
    Gpu,
    Cpu,  // For fallback scenarios
}

/// Thread-safe allocation tracker
pub struct AllocationTracker {
    allocations: RwLock<HashMap<BufferId, BufferMetadata>>,
    total_bytes: AtomicUsize,
    peak_bytes: AtomicUsize,
}
```

#### Memory Statistics

```rust
/// Unified stats across all backends
pub struct MemoryStats {
    pub total_allocated_bytes: usize,
    pub in_use_bytes: usize,
    pub pooled_bytes: usize,
    pub peak_bytes: usize,
    pub pool_hits: usize,
    pub pool_misses: usize,
    pub cpu_fallbacks: usize,
    pub active_buffers: usize,
}
```

#### Memory Configuration

```rust
/// Configuration for GPU memory management
pub struct MemoryConfig {
    pub gpu_memory_fraction: f64,      // Default: 0.9
    pub cpu_fallback_enabled: bool,     // Default: true
    pub min_free_bytes: usize,          // Default: 512 MB
    pub track_allocations: bool,        // Default: debug only
    pub max_pooled_per_bucket: usize,   // Default: 8
    pub pre_allocate: bool,
    pub pre_allocate_sizes: Vec<usize>,
}
```

**Preset Configurations:**
- `MemoryConfig::for_inference()` - Maximum memory usage, no tracking
- `MemoryConfig::for_training()` - More conservative, tracking in debug
- `MemoryConfig::for_constrained()` - For limited memory systems

#### Memory Pressure Levels

```rust
pub enum MemoryPressure {
    Low,      // < 60% utilization
    Medium,   // 60-80% utilization
    High,     // 80-95% utilization
    Critical, // > 95% utilization
}
```

#### Memory Pool Trait

```rust
/// Common trait for GPU memory pools
pub trait GpuMemoryPool: Send + Sync {
    fn backend_name(&self) -> &'static str;
    fn stats(&self) -> MemoryStats;
    fn total_memory(&self) -> usize;
    fn free_memory(&self) -> usize;
    fn pressure(&self) -> MemoryPressure;
    fn clear_pool(&mut self);
    fn shrink(&mut self) -> usize;
}
```

#### Generic Memory Pool

```rust
/// Generic implementation for any backend
pub struct GenericMemoryPool<B> {
    free_buffers: BTreeMap<usize, Vec<(B, usize)>>,
    config: MemoryConfig,
    stats: MemoryStats,
    tracker: Option<AllocationTracker>,
}
```

#### Fallback Buffer

```rust
/// Buffer that can be on GPU or CPU
pub struct FallbackBuffer<G> {
    gpu_buffer: Option<G>,
    cpu_buffer: Option<Vec<f32>>,
    location: BufferLocation,
    len: usize,
    id: BufferId,
}
```

#### Memory Estimation

```rust
/// Estimate model memory requirements
pub fn estimate_model_memory(
    hidden_size: usize,
    intermediate_size: usize,
    n_layers: usize,
    vocab_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    dtype_bytes: usize,
) -> usize;

/// Estimate KV cache memory
pub fn estimate_kv_cache_memory(
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    dtype_bytes: usize,
) -> usize;

/// Full inference memory breakdown
pub fn estimate_inference_memory(...) -> InferenceMemoryEstimate;

pub struct InferenceMemoryEstimate {
    pub model_bytes: usize,
    pub kv_cache_bytes: usize,
    pub activation_bytes: usize,
    pub total_bytes: usize,
}
```

## Files Modified

### `src/kernels/backend.rs`

Added backend discovery and device enumeration:

#### Backend Information

```rust
/// Information about an available backend
pub struct BackendInfo {
    pub backend_type: BackendType,
    pub name: String,
    pub available: bool,
    pub devices: Vec<DeviceInfo>,
    pub error: Option<String>,
}

pub enum BackendType {
    Cpu, Cuda, Rocm, Metal, OpenCL,
}

/// Information about a specific device
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub total_memory: usize,
    pub free_memory: Option<usize>,
    pub compute_capability: Option<String>,
    pub unified_memory: bool,
}
```

#### Discovery Functions

```rust
/// Discover all available backends and devices
pub fn discover_backends() -> Vec<BackendInfo>;

/// Get the best available backend
pub fn best_available_backend() -> BackendType;

/// Select backend that can fit a model
pub fn select_backend_for_model(required_bytes: usize) -> Result<(BackendType, usize)>;

/// Print summary of all backends
pub fn print_backend_summary();
```

#### Memory-Aware Initialization

```rust
/// Initialize backend with memory checks
pub fn init_backend_with_memory_check(
    preference: BackendPreference,
    required_bytes: usize,
) -> Result<Backend>;
```

#### System Information

Helper functions for cross-platform system info:
- `get_cpu_name()` - CPU model detection (Linux/macOS/Windows)
- `get_system_memory()` - Total RAM
- `get_available_memory()` - Free RAM

### `src/kernels/mod.rs`

Added exports:

```rust
pub mod gpu_memory;

// Re-export GPU memory management
pub use gpu_memory::{
    BufferId, BufferMetadata, BufferLocation, AllocationTracker,
    MemoryStats, MemoryConfig, MemoryPressure,
    GpuMemoryPool, GenericMemoryPool,
    FallbackBuffer, DeviceCapabilities,
    InferenceMemoryEstimate,
    estimate_model_memory, estimate_kv_cache_memory, estimate_inference_memory,
    round_up_power_of_2, size_class, compute_f32_bytes, compute_f16_bytes,
};

// Re-export backend discovery
pub use backend::{
    BackendInfo, BackendType, DeviceInfo,
    discover_backends, best_available_backend, select_backend_for_model,
    print_backend_summary, init_backend_with_memory_check,
};
```

### `src/lib.rs`

Added public exports for all new types and functions.

## Usage

### Discovering Available Backends

```rust
use torchless::{discover_backends, print_backend_summary, best_available_backend};

// Print all available backends
print_backend_summary();
// Output:
// === Available Backends ===
// ✓ CPU (CPU)
//     [0] Apple M2 Pro: 32.00 GB (28.50 GB free)
// ✓ Metal (Metal)
//     [0] Apple M2 Pro: 21.33 GB (unified memory)
// ✗ CUDA (CUDA)
//     Error: CUDA feature not enabled at compile time
// 
// Recommended backend: Metal

// Get the best backend programmatically
let best = best_available_backend();
println!("Best backend: {}", best);
```

### Selecting Backend for Model Size

```rust
use torchless::{select_backend_for_model, estimate_inference_memory};

// Estimate memory for Mistral 7B
let estimate = estimate_inference_memory(
    4096,   // hidden_size
    14336,  // intermediate_size
    32,     // n_layers
    32000,  // vocab_size
    32,     // n_heads
    8,      // n_kv_heads
    4096,   // max_seq_len
    4,      // dtype_bytes (f32)
);

println!("Model requires: {:.2} GB", estimate.total_gb());

// Find a backend with enough memory
match select_backend_for_model(estimate.total_bytes) {
    Ok((backend_type, device_idx)) => {
        println!("Use {} on device {}", backend_type, device_idx);
    }
    Err(e) => {
        println!("No backend has enough memory: {}", e);
    }
}
```

### Memory-Aware Backend Initialization

```rust
use torchless::{init_backend_with_memory_check, BackendPreference};

let required_bytes = 28 * 1024 * 1024 * 1024; // 28 GB

// Initialize with memory check
match init_backend_with_memory_check(BackendPreference::Auto, required_bytes) {
    Ok(backend) => {
        println!("Initialized {} backend", backend.name());
    }
    Err(e) => {
        // Helpful error message about which backends were tried
        // and why they failed
        eprintln!("Failed: {}", e);
    }
}
```

### Configuring Memory Management

```rust
use torchless::{MemoryConfig, MemoryPressure};

// Default configuration
let config = MemoryConfig::default();

// Inference-optimized (maximum memory usage)
let config = MemoryConfig::for_inference();

// Memory-constrained systems
let config = MemoryConfig::for_constrained();

// Custom configuration with builder
let config = MemoryConfig::new()
    .with_gpu_memory_fraction(0.8)   // Use 80% of GPU memory
    .with_cpu_fallback(true)          // Enable CPU fallback
    .with_min_free_bytes(1024 * 1024 * 1024)  // Keep 1GB free
    .with_tracking(true);             // Enable allocation tracking
```

### Tracking Allocations (Debug)

```rust
use torchless::{AllocationTracker, BufferMetadata, BufferId, BufferLocation};

let tracker = AllocationTracker::new();

// Track an allocation
let meta = BufferMetadata::new(
    BufferId::new(),
    1024 * 1024,  // 1 MB
    256 * 1024,   // 256K elements
    BufferLocation::Gpu,
).with_tag("attention_scores");

tracker.track(meta);

// Find potentially leaked buffers
let old = tracker.find_old_allocations(std::time::Duration::from_secs(60));
for alloc in old {
    eprintln!("Potential leak: {} ({})", alloc.id, alloc.tag.unwrap_or_default());
}

// Get stats
println!("Tracked: {} buffers, {} bytes", tracker.count(), tracker.total_bytes());
```

### Memory Estimation

```rust
use torchless::{estimate_inference_memory, InferenceMemoryEstimate};

// Estimate for Mistral 7B in FP32
let fp32 = estimate_inference_memory(4096, 14336, 32, 32000, 32, 8, 4096, 4);
println!("FP32: {:.2} GB", fp32.total_gb());

// Estimate for FP16
let fp16 = estimate_inference_memory(4096, 14336, 32, 32000, 32, 8, 4096, 2);
println!("FP16: {:.2} GB", fp16.total_gb());

// Detailed breakdown
println!("  Model: {:.2} GB", fp32.model_mb() / 1024.0);
println!("  KV Cache: {:.2} GB", fp32.kv_cache_mb() / 1024.0);
println!("  Activations: {:.2} MB", fp32.activation_mb());
```

## Memory Pool Integration

Each GPU backend can use the unified memory management:

```rust
use torchless::kernels::gpu_memory::{GenericMemoryPool, MemoryConfig, GpuMemoryPool};

// Create a pool for any backend
let config = MemoryConfig::for_inference();
let mut pool: GenericMemoryPool<Vec<u8>> = GenericMemoryPool::new(config);

// Record allocation
pool.record_allocation(1024 * 1024);

// Try to get from pool
if let Some((buffer, size)) = pool.try_get(512 * 1024) {
    // Reused from pool
} else {
    // Need to allocate new
}

// Return to pool
pool.return_buffer(buffer, size);

// Get stats
let stats = pool.stats();
println!("Hit rate: {:.1}%", stats.hit_rate() * 100.0);
```

## Tests

All new functionality is covered by tests:

```bash
# Run GPU memory tests
cargo test gpu_memory

# Run backend discovery tests
cargo test backend

# All tests
cargo test
```

Test coverage:
- `test_buffer_id_uniqueness` - Unique ID generation
- `test_round_up_power_of_2` - Bucket size calculation
- `test_memory_pressure` - Pressure level detection
- `test_memory_config_builder` - Configuration builder
- `test_allocation_tracker` - Allocation tracking
- `test_generic_memory_pool` - Pool operations
- `test_inference_memory_estimate` - Memory estimation
- `test_memory_stats_display` - Stats formatting
- `test_fallback_buffer` - GPU/CPU buffer abstraction

## Dependencies

No new dependencies required - uses existing standard library types.

## Performance Considerations

1. **Buffer Pooling**: Power-of-2 bucket sizing for efficient reuse
2. **Hit Rate Tracking**: Monitor pool effectiveness
3. **Lazy Allocation Tracking**: Only tracks when enabled (debug builds)
4. **Lock-Free Statistics**: Atomic counters for hot paths

## Future Improvements

1. **Per-Backend Pool Integration**: Connect unified pools to each backend
2. **Async Memory Operations**: Overlap transfers with computation
3. **Memory Defragmentation**: Compact pool to reduce fragmentation
4. **Multi-GPU Memory**: Coordinate memory across multiple GPUs
5. **Memory Pressure Callbacks**: Notify when pressure changes

## Related Phases

- **Phase 5.1 (CUDA)**: Uses CudaMemoryPool which can implement GpuMemoryPool
- **Phase 5.2 (ROCm)**: Uses RocmMemoryPool which can implement GpuMemoryPool
- **Phase 5.3 (Metal)**: Uses MetalMemoryPool which can implement GpuMemoryPool
- **Phase 5.4 (OpenCL)**: Uses OpenCLMemoryPool which can implement GpuMemoryPool
