# GPU Memory Management

Unified GPU memory management system working across all GPU backends.

## Overview

Provides:
- **Unified Memory Pool Interface**: Common trait for all GPU memory pools
- **Buffer Tracking**: Debug allocations with unique IDs and metadata
- **Memory Pressure Monitoring**: Detect and respond to low memory conditions
- **Automatic CPU Fallback**: Gracefully fall back to CPU when GPU memory exhausted
- **Memory-Aware Initialization**: Select backends based on available memory

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GpuMemoryManager                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ MemoryConfig │  │MemoryStats   │  │MemoryPressure│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              GpuMemoryPool Trait                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│          │                │                │                │   │
│          ▼                ▼                ▼                ▼   │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────┐ │
│  │ CudaPool  │    │ RocmPool  │    │MetalPool  │    │OpenCL │ │
│  └───────────┘    └───────────┘    └───────────┘    └───────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Statistics

```rust
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

## Memory Configuration

```rust
use torchless::MemoryConfig;

// Default configuration
let config = MemoryConfig::default();

// Inference-optimized (maximum memory usage)
let config = MemoryConfig::for_inference();

// Memory-constrained systems
let config = MemoryConfig::for_constrained();

// Custom configuration
let config = MemoryConfig::new()
    .with_gpu_memory_fraction(0.8)   // Use 80% of GPU memory
    .with_cpu_fallback(true)          // Enable CPU fallback
    .with_min_free_bytes(1024 * 1024 * 1024)  // Keep 1GB free
    .with_tracking(true);             // Enable allocation tracking
```

Configuration options:
- `gpu_memory_fraction` - Fraction of GPU memory to use (default: 0.9)
- `cpu_fallback_enabled` - Enable CPU fallback (default: true)
- `min_free_bytes` - Minimum free bytes to maintain (default: 512 MB)
- `track_allocations` - Enable allocation tracking (default: debug only)
- `max_pooled_per_bucket` - Max buffers per size bucket (default: 8)

## Memory Pressure Levels

```rust
pub enum MemoryPressure {
    Low,      // < 60% utilization
    Medium,   // 60-80% utilization
    High,     // 80-95% utilization
    Critical, // > 95% utilization
}
```

## GpuMemoryPool Trait

```rust
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

## Memory Estimation

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

## Selecting Backend for Model Size

```rust
use torchless::select_backend_for_model;

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

## Buffer Tracking (Debug)

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

## Fallback Buffer

Buffer that can be on GPU or CPU:

```rust
pub struct FallbackBuffer<G> {
    gpu_buffer: Option<G>,
    cpu_buffer: Option<Vec<f32>>,
    location: BufferLocation,
    len: usize,
    id: BufferId,
}
```

Automatically falls back to CPU when GPU memory is exhausted.

## Memory Pool Integration

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

## Performance Considerations

1. **Buffer Pooling**: Power-of-2 bucket sizing for efficient reuse
2. **Hit Rate Tracking**: Monitor pool effectiveness
3. **Lazy Allocation Tracking**: Only tracks when enabled (debug builds)
4. **Lock-Free Statistics**: Atomic counters for hot paths
