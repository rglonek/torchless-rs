//! Unified GPU Memory Management
//!
//! This module provides a unified interface for GPU memory management across all backends
//! (CUDA, ROCm, Metal, OpenCL). It includes:
//!
//! - **Common Memory Pool Trait**: Abstract interface for buffer pooling
//! - **Memory Pressure Monitoring**: Track and respond to memory constraints
//! - **Automatic CPU Fallback**: Fall back to CPU when GPU memory is exhausted
//! - **Buffer ID Tracking**: Debug and profile buffer allocations
//! - **Multi-Dtype Support**: Handle different data types (f32, f16, etc.)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    GpuMemoryManager                          │
//! │  (Unified interface with fallback logic)                    │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!        ┌─────────────────────┼─────────────────────┐
//!        ▼                     ▼                     ▼
//! ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
//! │ CudaPool    │       │ MetalPool   │       │ OpenCLPool  │
//! └─────────────┘       └─────────────┘       └─────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use torchless::kernels::gpu_memory::{GpuMemoryManager, MemoryConfig};
//!
//! // Create manager with 80% GPU memory limit and CPU fallback enabled
//! let config = MemoryConfig::default()
//!     .with_gpu_memory_fraction(0.8)
//!     .with_cpu_fallback(true);
//!
//! let manager = GpuMemoryManager::new(config)?;
//!
//! // Allocate buffer - automatically falls back to CPU if needed
//! let buffer = manager.alloc_f32(1024)?;
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::RwLock;

// =============================================================================
// Buffer ID Tracking
// =============================================================================

/// Unique identifier for a buffer allocation.
/// Used for debugging and tracking buffer lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(pub u64);

impl BufferId {
    /// Generate a new unique buffer ID.
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Buffer#{}", self.0)
    }
}

// =============================================================================
// Memory Statistics
// =============================================================================

/// Unified memory pool statistics across all backends.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total bytes allocated from the device/system
    pub total_allocated_bytes: usize,
    /// Bytes currently in use by active tensors
    pub in_use_bytes: usize,
    /// Bytes sitting in pool waiting for reuse
    pub pooled_bytes: usize,
    /// Peak memory usage (high water mark)
    pub peak_bytes: usize,
    /// Number of allocations served from pool (cache hits)
    pub pool_hits: usize,
    /// Number of allocations requiring new allocation (cache misses)
    pub pool_misses: usize,
    /// Number of CPU fallback allocations
    pub cpu_fallbacks: usize,
    /// Number of active buffer IDs being tracked
    pub active_buffers: usize,
}

impl MemoryStats {
    /// Calculate pool hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.pool_hits + self.pool_misses;
        if total > 0 {
            self.pool_hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Calculate memory utilization (in_use / total).
    pub fn utilization(&self) -> f64 {
        if self.total_allocated_bytes > 0 {
            self.in_use_bytes as f64 / self.total_allocated_bytes as f64
        } else {
            0.0
        }
    }

    /// Get memory efficiency (1.0 = all allocated memory is in use).
    pub fn efficiency(&self) -> f64 {
        if self.total_allocated_bytes > 0 {
            self.in_use_bytes as f64 / self.total_allocated_bytes as f64
        } else {
            1.0
        }
    }

    /// Merge stats from another source (useful for aggregating).
    pub fn merge(&mut self, other: &MemoryStats) {
        self.total_allocated_bytes += other.total_allocated_bytes;
        self.in_use_bytes += other.in_use_bytes;
        self.pooled_bytes += other.pooled_bytes;
        self.peak_bytes = self.peak_bytes.max(other.peak_bytes);
        self.pool_hits += other.pool_hits;
        self.pool_misses += other.pool_misses;
        self.cpu_fallbacks += other.cpu_fallbacks;
        self.active_buffers += other.active_buffers;
    }
}

impl fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Memory: {:.2} MB allocated ({:.2} MB in use, {:.2} MB pooled), \
             peak {:.2} MB, {:.1}% hit rate ({} hits, {} misses), \
             {} CPU fallbacks, {} active buffers",
            self.total_allocated_bytes as f64 / 1024.0 / 1024.0,
            self.in_use_bytes as f64 / 1024.0 / 1024.0,
            self.pooled_bytes as f64 / 1024.0 / 1024.0,
            self.peak_bytes as f64 / 1024.0 / 1024.0,
            self.hit_rate() * 100.0,
            self.pool_hits,
            self.pool_misses,
            self.cpu_fallbacks,
            self.active_buffers,
        )
    }
}

// =============================================================================
// Memory Configuration
// =============================================================================

/// Configuration for GPU memory management.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Fraction of GPU memory to use (0.0 to 1.0).
    /// Default: 0.9 (90%)
    pub gpu_memory_fraction: f64,

    /// Whether to fall back to CPU when GPU memory is exhausted.
    /// Default: true
    pub cpu_fallback_enabled: bool,

    /// Minimum free GPU memory to maintain (in bytes).
    /// Default: 512 MB
    pub min_free_bytes: usize,

    /// Whether to track individual buffer allocations.
    /// Useful for debugging but adds overhead.
    /// Default: false in release, true in debug builds
    pub track_allocations: bool,

    /// Maximum number of buffers to keep in pool per size bucket.
    /// Default: 8
    pub max_pooled_per_bucket: usize,

    /// Whether to pre-allocate common buffer sizes.
    /// Default: false
    pub pre_allocate: bool,

    /// Common sizes to pre-allocate if pre_allocate is true.
    pub pre_allocate_sizes: Vec<usize>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            gpu_memory_fraction: 0.9,
            cpu_fallback_enabled: true,
            min_free_bytes: 512 * 1024 * 1024, // 512 MB
            track_allocations: cfg!(debug_assertions),
            max_pooled_per_bucket: 8,
            pre_allocate: false,
            pre_allocate_sizes: vec![],
        }
    }
}

impl MemoryConfig {
    /// Create a new memory config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set GPU memory fraction (builder pattern).
    pub fn with_gpu_memory_fraction(mut self, fraction: f64) -> Self {
        self.gpu_memory_fraction = fraction.clamp(0.1, 1.0);
        self
    }

    /// Enable or disable CPU fallback (builder pattern).
    pub fn with_cpu_fallback(mut self, enabled: bool) -> Self {
        self.cpu_fallback_enabled = enabled;
        self
    }

    /// Set minimum free bytes (builder pattern).
    pub fn with_min_free_bytes(mut self, bytes: usize) -> Self {
        self.min_free_bytes = bytes;
        self
    }

    /// Enable or disable allocation tracking (builder pattern).
    pub fn with_tracking(mut self, enabled: bool) -> Self {
        self.track_allocations = enabled;
        self
    }

    /// Set max pooled buffers per bucket (builder pattern).
    pub fn with_max_pooled(mut self, max: usize) -> Self {
        self.max_pooled_per_bucket = max;
        self
    }

    /// Enable pre-allocation with specified sizes (builder pattern).
    pub fn with_pre_allocation(mut self, sizes: Vec<usize>) -> Self {
        self.pre_allocate = true;
        self.pre_allocate_sizes = sizes;
        self
    }

    /// Configuration optimized for inference (maximum memory usage, no tracking).
    pub fn for_inference() -> Self {
        Self {
            gpu_memory_fraction: 0.95,
            cpu_fallback_enabled: true,
            min_free_bytes: 256 * 1024 * 1024,
            track_allocations: false,
            max_pooled_per_bucket: 4,
            pre_allocate: false,
            pre_allocate_sizes: vec![],
        }
    }

    /// Configuration optimized for training (more conservative).
    pub fn for_training() -> Self {
        Self {
            gpu_memory_fraction: 0.8,
            cpu_fallback_enabled: false,
            min_free_bytes: 1024 * 1024 * 1024,
            track_allocations: cfg!(debug_assertions),
            max_pooled_per_bucket: 16,
            pre_allocate: false,
            pre_allocate_sizes: vec![],
        }
    }

    /// Configuration for memory-constrained systems.
    pub fn for_constrained() -> Self {
        Self {
            gpu_memory_fraction: 0.7,
            cpu_fallback_enabled: true,
            min_free_bytes: 128 * 1024 * 1024,
            track_allocations: false,
            max_pooled_per_bucket: 2,
            pre_allocate: false,
            pre_allocate_sizes: vec![],
        }
    }
}

// =============================================================================
// Memory Pressure Levels
// =============================================================================

/// Current memory pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Memory usage is low, plenty of headroom
    Low,
    /// Memory usage is moderate
    Medium,
    /// Memory usage is high, consider reducing allocations
    High,
    /// Memory is critically low, immediate action needed
    Critical,
}

impl MemoryPressure {
    /// Determine pressure level based on utilization percentage.
    pub fn from_utilization(utilization: f64) -> Self {
        if utilization < 0.6 {
            MemoryPressure::Low
        } else if utilization < 0.8 {
            MemoryPressure::Medium
        } else if utilization < 0.95 {
            MemoryPressure::High
        } else {
            MemoryPressure::Critical
        }
    }

    /// Check if we should try to free memory.
    pub fn should_free(&self) -> bool {
        matches!(self, MemoryPressure::High | MemoryPressure::Critical)
    }

    /// Check if we should deny new allocations.
    pub fn should_deny(&self) -> bool {
        matches!(self, MemoryPressure::Critical)
    }
}

impl fmt::Display for MemoryPressure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryPressure::Low => write!(f, "Low"),
            MemoryPressure::Medium => write!(f, "Medium"),
            MemoryPressure::High => write!(f, "High"),
            MemoryPressure::Critical => write!(f, "Critical"),
        }
    }
}

// =============================================================================
// Buffer Location
// =============================================================================

/// Where a buffer is located.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferLocation {
    /// Buffer is on GPU
    Gpu,
    /// Buffer fell back to CPU
    Cpu,
}

impl fmt::Display for BufferLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BufferLocation::Gpu => write!(f, "GPU"),
            BufferLocation::Cpu => write!(f, "CPU"),
        }
    }
}

// =============================================================================
// Buffer Metadata
// =============================================================================

/// Metadata for a tracked buffer allocation.
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    /// Unique buffer ID
    pub id: BufferId,
    /// Size in bytes
    pub size_bytes: usize,
    /// Number of elements
    pub num_elements: usize,
    /// Where the buffer is located
    pub location: BufferLocation,
    /// Timestamp when allocated (monotonic)
    pub allocated_at: std::time::Instant,
    /// Optional name/tag for debugging
    pub tag: Option<String>,
}

impl BufferMetadata {
    /// Create new metadata for a buffer.
    pub fn new(
        id: BufferId,
        size_bytes: usize,
        num_elements: usize,
        location: BufferLocation,
    ) -> Self {
        Self {
            id,
            size_bytes,
            num_elements,
            location,
            allocated_at: std::time::Instant::now(),
            tag: None,
        }
    }

    /// Set a debug tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }

    /// Get age of this allocation.
    pub fn age(&self) -> std::time::Duration {
        self.allocated_at.elapsed()
    }
}

// =============================================================================
// Allocation Tracker
// =============================================================================

/// Tracks active buffer allocations for debugging.
#[derive(Debug, Default)]
pub struct AllocationTracker {
    /// Map of buffer ID to metadata
    allocations: RwLock<HashMap<BufferId, BufferMetadata>>,
    /// Total bytes tracked
    total_bytes: AtomicUsize,
    /// Peak bytes tracked
    peak_bytes: AtomicUsize,
}

impl AllocationTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Track a new allocation.
    pub fn track(&self, metadata: BufferMetadata) {
        let size = metadata.size_bytes;
        let id = metadata.id;

        let mut allocs = self.allocations.write().unwrap();
        allocs.insert(id, metadata);

        let new_total = self.total_bytes.fetch_add(size, Ordering::Relaxed) + size;
        self.peak_bytes.fetch_max(new_total, Ordering::Relaxed);
    }

    /// Untrack an allocation.
    pub fn untrack(&self, id: BufferId) -> Option<BufferMetadata> {
        let mut allocs = self.allocations.write().unwrap();
        if let Some(meta) = allocs.remove(&id) {
            self.total_bytes
                .fetch_sub(meta.size_bytes, Ordering::Relaxed);
            Some(meta)
        } else {
            None
        }
    }

    /// Get metadata for a buffer.
    pub fn get(&self, id: BufferId) -> Option<BufferMetadata> {
        self.allocations.read().unwrap().get(&id).cloned()
    }

    /// Get all tracked allocations.
    pub fn all_allocations(&self) -> Vec<BufferMetadata> {
        self.allocations.read().unwrap().values().cloned().collect()
    }

    /// Get number of tracked allocations.
    pub fn count(&self) -> usize {
        self.allocations.read().unwrap().len()
    }

    /// Get total tracked bytes.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Get peak tracked bytes.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Find potentially leaked buffers (older than threshold).
    pub fn find_old_allocations(&self, threshold: std::time::Duration) -> Vec<BufferMetadata> {
        self.allocations
            .read()
            .unwrap()
            .values()
            .filter(|m| m.age() > threshold)
            .cloned()
            .collect()
    }

    /// Clear all tracked allocations.
    pub fn clear(&self) {
        self.allocations.write().unwrap().clear();
        self.total_bytes.store(0, Ordering::Relaxed);
    }
}

// =============================================================================
// Memory Pool Trait
// =============================================================================

/// Common trait for GPU memory pools across all backends.
///
/// This trait defines the interface that all backend-specific memory pools
/// must implement for unified memory management.
pub trait GpuMemoryPool: Send + Sync {
    /// Get the name of this memory pool backend.
    fn backend_name(&self) -> &'static str;

    /// Get current memory statistics.
    fn stats(&self) -> MemoryStats;

    /// Get total available GPU memory in bytes.
    fn total_memory(&self) -> usize;

    /// Get currently free GPU memory in bytes.
    fn free_memory(&self) -> usize;

    /// Get current memory pressure level.
    fn pressure(&self) -> MemoryPressure {
        let utilization = 1.0 - (self.free_memory() as f64 / self.total_memory() as f64);
        MemoryPressure::from_utilization(utilization)
    }

    /// Clear all pooled (unused) buffers to free memory.
    fn clear_pool(&mut self);

    /// Shrink the pool to release unused memory to the system.
    /// Returns the number of bytes freed.
    fn shrink(&mut self) -> usize;
}

// =============================================================================
// CPU Fallback Buffer
// =============================================================================

/// A buffer that can be on GPU or CPU (for fallback scenarios).
#[derive(Debug)]
pub struct FallbackBuffer<G> {
    /// GPU buffer if allocated there
    gpu_buffer: Option<G>,
    /// CPU fallback buffer
    cpu_buffer: Option<Vec<f32>>,
    /// Where the data currently resides
    location: BufferLocation,
    /// Number of elements
    len: usize,
    /// Unique ID for tracking
    id: BufferId,
}

impl<G> FallbackBuffer<G> {
    /// Create a new GPU-backed buffer.
    pub fn gpu(buffer: G, len: usize) -> Self {
        Self {
            gpu_buffer: Some(buffer),
            cpu_buffer: None,
            location: BufferLocation::Gpu,
            len,
            id: BufferId::new(),
        }
    }

    /// Create a new CPU-backed buffer (fallback).
    pub fn cpu(data: Vec<f32>) -> Self {
        let len = data.len();
        Self {
            gpu_buffer: None,
            cpu_buffer: Some(data),
            location: BufferLocation::Cpu,
            len,
            id: BufferId::new(),
        }
    }

    /// Get the buffer location.
    pub fn location(&self) -> BufferLocation {
        self.location
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if this is a GPU buffer.
    pub fn is_gpu(&self) -> bool {
        self.location == BufferLocation::Gpu
    }

    /// Check if this is a CPU fallback buffer.
    pub fn is_cpu_fallback(&self) -> bool {
        self.location == BufferLocation::Cpu
    }

    /// Get the unique buffer ID.
    pub fn id(&self) -> BufferId {
        self.id
    }

    /// Get reference to GPU buffer (if present).
    pub fn as_gpu(&self) -> Option<&G> {
        self.gpu_buffer.as_ref()
    }

    /// Get mutable reference to GPU buffer (if present).
    pub fn as_gpu_mut(&mut self) -> Option<&mut G> {
        self.gpu_buffer.as_mut()
    }

    /// Get reference to CPU buffer (if present).
    pub fn as_cpu(&self) -> Option<&[f32]> {
        self.cpu_buffer.as_deref()
    }

    /// Get mutable reference to CPU buffer (if present).
    pub fn as_cpu_mut(&mut self) -> Option<&mut [f32]> {
        self.cpu_buffer.as_deref_mut()
    }

    /// Take the GPU buffer out.
    pub fn take_gpu(self) -> Option<G> {
        self.gpu_buffer
    }

    /// Take the CPU buffer out.
    pub fn take_cpu(self) -> Option<Vec<f32>> {
        self.cpu_buffer
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Round up to the nearest power of 2.
/// Used for bucket sizing in memory pools.
pub fn round_up_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

/// Calculate the size class (bucket) for a given size.
/// Returns the power of 2 bucket.
pub fn size_class(size: usize) -> usize {
    round_up_power_of_2(size)
}

/// Compute tensor size in bytes for f32 elements.
pub fn compute_f32_bytes(elements: usize) -> usize {
    elements * std::mem::size_of::<f32>()
}

/// Compute tensor size in bytes for f16 elements.
pub fn compute_f16_bytes(elements: usize) -> usize {
    elements * 2
}

/// Estimate GPU memory needed for a model.
///
/// # Arguments
/// * `hidden_size` - Model hidden dimension
/// * `intermediate_size` - MLP intermediate dimension
/// * `n_layers` - Number of transformer layers
/// * `vocab_size` - Vocabulary size
/// * `n_heads` - Number of attention heads
/// * `n_kv_heads` - Number of KV heads (for GQA)
/// * `dtype_bytes` - Bytes per element (4 for f32, 2 for f16)
pub fn estimate_model_memory(
    hidden_size: usize,
    intermediate_size: usize,
    n_layers: usize,
    vocab_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    dtype_bytes: usize,
) -> usize {
    let head_dim = hidden_size / n_heads;

    // Per-layer weights
    let q_proj = hidden_size * hidden_size;
    let k_proj = hidden_size * (n_kv_heads * head_dim);
    let v_proj = hidden_size * (n_kv_heads * head_dim);
    let o_proj = hidden_size * hidden_size;
    let gate_proj = hidden_size * intermediate_size;
    let up_proj = hidden_size * intermediate_size;
    let down_proj = intermediate_size * hidden_size;
    let norm = hidden_size * 2;

    let layer_params = q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj + norm;
    let total_layer_params = layer_params * n_layers;

    // Embedding and output
    let embedding = vocab_size * hidden_size;
    let output = vocab_size * hidden_size;
    let final_norm = hidden_size;

    let total_params = total_layer_params + embedding + output + final_norm;
    total_params * dtype_bytes
}

/// Estimate KV cache memory for a model.
///
/// # Arguments
/// * `n_layers` - Number of transformer layers
/// * `n_kv_heads` - Number of KV heads
/// * `head_dim` - Head dimension
/// * `max_seq_len` - Maximum sequence length
/// * `dtype_bytes` - Bytes per element
pub fn estimate_kv_cache_memory(
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    dtype_bytes: usize,
) -> usize {
    // K and V each: [n_layers, max_seq_len, n_kv_heads, head_dim]
    let elements_per_cache = n_layers * max_seq_len * n_kv_heads * head_dim;
    elements_per_cache * 2 * dtype_bytes
}

/// Estimate total inference memory requirements.
#[allow(clippy::too_many_arguments)]
pub fn estimate_inference_memory(
    hidden_size: usize,
    intermediate_size: usize,
    n_layers: usize,
    vocab_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    dtype_bytes: usize,
) -> InferenceMemoryEstimate {
    let head_dim = hidden_size / n_heads;

    let model_bytes = estimate_model_memory(
        hidden_size,
        intermediate_size,
        n_layers,
        vocab_size,
        n_heads,
        n_kv_heads,
        dtype_bytes,
    );

    let kv_cache_bytes =
        estimate_kv_cache_memory(n_layers, n_kv_heads, head_dim, max_seq_len, dtype_bytes);

    // Activation memory (rough estimate)
    let activation_bytes = (hidden_size + intermediate_size * 2) * dtype_bytes * 2;

    InferenceMemoryEstimate {
        model_bytes,
        kv_cache_bytes,
        activation_bytes,
        total_bytes: model_bytes + kv_cache_bytes + activation_bytes,
    }
}

/// Estimate GPU memory needed for a Mixture-of-Experts model.
///
/// MoE models have two types of layers:
/// - Dense layers (first_moe_layer layers): standard attention + MLP
/// - MoE layers (remaining layers): attention + router + N experts + optional shared experts
///
/// This returns the TOTAL parameter memory (all experts loaded).
/// For lazy/memory-mapped inference, actual RAM usage is much lower since
/// only the top-k experts are accessed per token.
///
/// # Arguments
/// * `hidden_size` - Model hidden dimension
/// * `intermediate_size` - Dense MLP intermediate dimension
/// * `moe_intermediate_size` - Expert FFN intermediate dimension
/// * `n_layers` - Total number of transformer layers
/// * `first_moe_layer` - Index of first MoE layer (layers before are dense)
/// * `n_routed_experts` - Number of routed experts per MoE layer
/// * `n_shared_experts` - Number of shared experts per MoE layer
/// * `vocab_size` - Vocabulary size
/// * `n_heads` - Number of attention heads
/// * `n_kv_heads` - Number of KV heads (for GQA)
/// * `dtype_bytes` - Bytes per element (4 for f32, 2 for f16)
#[allow(clippy::too_many_arguments)]
pub fn estimate_moe_model_memory(
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    n_layers: usize,
    first_moe_layer: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    vocab_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    dtype_bytes: usize,
) -> MoEMemoryEstimate {
    let head_dim = hidden_size / n_heads;

    // Attention weights (same for all layers)
    let q_proj = hidden_size * hidden_size;
    let k_proj = hidden_size * (n_kv_heads * head_dim);
    let v_proj = hidden_size * (n_kv_heads * head_dim);
    let o_proj = hidden_size * hidden_size;
    let attn_params = q_proj + k_proj + v_proj + o_proj;
    let norm_params = hidden_size * 2; // input_norm + post_attention_norm

    // Dense layer MLP weights
    let dense_mlp_params = hidden_size * intermediate_size * 3; // gate + up + down

    // MoE layer weights
    let router_params = n_routed_experts * hidden_size; // gate weight
    let expert_params = moe_intermediate_size * hidden_size * 3; // per expert: gate + up + down
    let total_expert_params = expert_params * n_routed_experts;
    let shared_expert_params = if n_shared_experts > 0 {
        moe_intermediate_size * hidden_size * 3 * n_shared_experts
    } else {
        0
    };
    let moe_layer_params = router_params + total_expert_params + shared_expert_params;

    // Per-layer totals
    let dense_layer_params = attn_params + dense_mlp_params + norm_params;
    let moe_layer_total_params = attn_params + moe_layer_params + norm_params;

    let n_dense_layers = first_moe_layer;
    let n_moe_layers = n_layers - first_moe_layer;

    let total_dense_params = dense_layer_params * n_dense_layers;
    let total_moe_params = moe_layer_total_params * n_moe_layers;

    // Embedding and output
    let embedding = vocab_size * hidden_size;
    let output = vocab_size * hidden_size;
    let final_norm = hidden_size;

    let total_params = total_dense_params + total_moe_params + embedding + output + final_norm;

    // Active parameters per token (only top-k experts per MoE layer)
    // This is useful for understanding computational cost
    let n_experts_per_token = if n_routed_experts > 0 { 8 } else { 0 }; // default top-k
    let active_expert_params = expert_params * n_experts_per_token + shared_expert_params;
    let active_moe_layer_params = attn_params + router_params + active_expert_params + norm_params;
    let active_params = total_dense_params
        + active_moe_layer_params * n_moe_layers
        + embedding
        + output
        + final_norm;

    MoEMemoryEstimate {
        total_bytes: total_params * dtype_bytes,
        active_bytes: active_params * dtype_bytes,
        total_params,
        active_params,
        n_dense_layers,
        n_moe_layers,
    }
}

/// Memory estimate breakdown for MoE models.
#[derive(Debug, Clone)]
pub struct MoEMemoryEstimate {
    /// Total memory for ALL model weights (all experts)
    pub total_bytes: usize,
    /// Memory for active parameters per token (only top-k experts)
    pub active_bytes: usize,
    /// Total parameter count
    pub total_params: usize,
    /// Active parameter count per token
    pub active_params: usize,
    /// Number of dense layers
    pub n_dense_layers: usize,
    /// Number of MoE layers
    pub n_moe_layers: usize,
}

/// Memory estimate breakdown for inference.
#[derive(Debug, Clone)]
pub struct InferenceMemoryEstimate {
    /// Memory for model weights
    pub model_bytes: usize,
    /// Memory for KV cache
    pub kv_cache_bytes: usize,
    /// Memory for activations
    pub activation_bytes: usize,
    /// Total memory required
    pub total_bytes: usize,
}

impl InferenceMemoryEstimate {
    /// Get model memory in MB.
    pub fn model_mb(&self) -> f64 {
        self.model_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get KV cache memory in MB.
    pub fn kv_cache_mb(&self) -> f64 {
        self.kv_cache_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get activation memory in MB.
    pub fn activation_mb(&self) -> f64 {
        self.activation_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get total memory in MB.
    pub fn total_mb(&self) -> f64 {
        self.total_bytes as f64 / 1024.0 / 1024.0
    }

    /// Get total memory in GB.
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / 1024.0 / 1024.0 / 1024.0
    }
}

impl fmt::Display for InferenceMemoryEstimate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Inference Memory: {:.2} GB total ({:.2} GB model, {:.2} GB KV cache, {:.2} MB activations)",
            self.total_gb(),
            self.model_mb() / 1024.0,
            self.kv_cache_mb() / 1024.0,
            self.activation_mb(),
        )
    }
}

// =============================================================================
// Common Memory Pool Implementation
// =============================================================================

/// A generic memory pool implementation that can be used by any backend.
/// This provides the common logic for buffer pooling.
#[derive(Debug)]
pub struct GenericMemoryPool<B> {
    /// Free buffers organized by size bucket
    free_buffers: BTreeMap<usize, Vec<(B, usize)>>,
    /// Configuration
    config: MemoryConfig,
    /// Statistics
    stats: MemoryStats,
    /// Allocation tracker (optional)
    tracker: Option<AllocationTracker>,
}

impl<B> GenericMemoryPool<B> {
    /// Create a new generic memory pool.
    pub fn new(config: MemoryConfig) -> Self {
        let tracker = if config.track_allocations {
            Some(AllocationTracker::new())
        } else {
            None
        };

        Self {
            free_buffers: BTreeMap::new(),
            config,
            stats: MemoryStats::default(),
            tracker,
        }
    }

    /// Try to get a buffer from the pool.
    /// Returns None if no suitable buffer is available.
    pub fn try_get(&mut self, min_size: usize) -> Option<(B, usize)> {
        let bucket_size = round_up_power_of_2(min_size);

        // Try exact bucket first
        if let Some(buffers) = self.free_buffers.get_mut(&bucket_size) {
            if let Some(buffer) = buffers.pop() {
                self.stats.in_use_bytes += buffer.1;
                self.stats.pooled_bytes = self.stats.pooled_bytes.saturating_sub(buffer.1);
                self.stats.pool_hits += 1;
                return Some(buffer);
            }
        }

        // Try larger buckets
        for (_size, buffers) in self.free_buffers.range_mut(bucket_size..) {
            if let Some(buffer) = buffers.pop() {
                self.stats.in_use_bytes += buffer.1;
                self.stats.pooled_bytes = self.stats.pooled_bytes.saturating_sub(buffer.1);
                self.stats.pool_hits += 1;
                return Some(buffer);
            }
        }

        self.stats.pool_misses += 1;
        None
    }

    /// Record a new allocation that wasn't from the pool.
    pub fn record_allocation(&mut self, size_bytes: usize) {
        self.stats.total_allocated_bytes += size_bytes;
        self.stats.in_use_bytes += size_bytes;
        self.stats.peak_bytes = self.stats.peak_bytes.max(self.stats.in_use_bytes);
    }

    /// Return a buffer to the pool.
    pub fn return_buffer(&mut self, buffer: B, size_bytes: usize) {
        let bucket_size = round_up_power_of_2(size_bytes);

        self.stats.in_use_bytes = self.stats.in_use_bytes.saturating_sub(size_bytes);

        // Check if we should keep this buffer
        let buffers = self.free_buffers.entry(bucket_size).or_default();
        if buffers.len() < self.config.max_pooled_per_bucket {
            self.stats.pooled_bytes += size_bytes;
            buffers.push((buffer, size_bytes));
        }
        // If pool is full, buffer is dropped
    }

    /// Record a CPU fallback allocation.
    pub fn record_cpu_fallback(&mut self) {
        self.stats.cpu_fallbacks += 1;
    }

    /// Clear all pooled buffers.
    pub fn clear(&mut self) {
        let freed = self.stats.pooled_bytes;
        self.free_buffers.clear();
        self.stats.pooled_bytes = 0;
        self.stats.total_allocated_bytes = self.stats.total_allocated_bytes.saturating_sub(freed);
    }

    /// Get current statistics.
    pub fn stats(&self) -> MemoryStats {
        let mut stats = self.stats.clone();
        if let Some(tracker) = &self.tracker {
            stats.active_buffers = tracker.count();
        }
        stats
    }

    /// Get the allocation tracker.
    pub fn tracker(&self) -> Option<&AllocationTracker> {
        self.tracker.as_ref()
    }

    /// Get mutable allocation tracker.
    pub fn tracker_mut(&mut self) -> Option<&mut AllocationTracker> {
        self.tracker.as_mut()
    }

    /// Shrink the pool, returning estimated bytes freed.
    pub fn shrink(&mut self) -> usize {
        let freed = self.stats.pooled_bytes;
        self.clear();
        freed
    }

    /// Get number of pooled buffers.
    pub fn pooled_count(&self) -> usize {
        self.free_buffers.values().map(|v| v.len()).sum()
    }
}

// =============================================================================
// Device Capabilities
// =============================================================================

/// Information about a GPU device's capabilities.
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device name
    pub name: String,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Compute capability or equivalent
    pub compute_capability: String,
    /// Whether unified memory is supported (e.g., Apple Silicon)
    pub unified_memory: bool,
    /// Maximum allocation size
    pub max_allocation_size: usize,
    /// Memory alignment requirement
    pub alignment: usize,
}

impl DeviceCapabilities {
    /// Check if the device has enough memory for a model.
    pub fn can_fit(&self, required_bytes: usize) -> bool {
        required_bytes <= self.total_memory
    }

    /// Get recommended memory fraction based on device type.
    pub fn recommended_memory_fraction(&self) -> f64 {
        if self.unified_memory {
            // For unified memory (Apple Silicon), be more conservative
            // as it's shared with the system
            0.7
        } else {
            // For discrete GPUs, can use more
            0.9
        }
    }
}

impl fmt::Display for DeviceCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {:.2} GB memory, compute {}{}",
            self.name,
            self.total_memory as f64 / 1024.0 / 1024.0 / 1024.0,
            self.compute_capability,
            if self.unified_memory {
                " (unified)"
            } else {
                ""
            }
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_id_uniqueness() {
        let id1 = BufferId::new();
        let id2 = BufferId::new();
        let id3 = BufferId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert!(id2.0 > id1.0);
        assert!(id3.0 > id2.0);
    }

    #[test]
    fn test_round_up_power_of_2() {
        assert_eq!(round_up_power_of_2(0), 1);
        assert_eq!(round_up_power_of_2(1), 1);
        assert_eq!(round_up_power_of_2(2), 2);
        assert_eq!(round_up_power_of_2(3), 4);
        assert_eq!(round_up_power_of_2(4), 4);
        assert_eq!(round_up_power_of_2(5), 8);
        assert_eq!(round_up_power_of_2(1000), 1024);
    }

    #[test]
    fn test_memory_pressure() {
        assert_eq!(MemoryPressure::from_utilization(0.3), MemoryPressure::Low);
        assert_eq!(
            MemoryPressure::from_utilization(0.7),
            MemoryPressure::Medium
        );
        assert_eq!(MemoryPressure::from_utilization(0.9), MemoryPressure::High);
        assert_eq!(
            MemoryPressure::from_utilization(0.98),
            MemoryPressure::Critical
        );
    }

    #[test]
    fn test_memory_config_builder() {
        let config = MemoryConfig::new()
            .with_gpu_memory_fraction(0.8)
            .with_cpu_fallback(false)
            .with_min_free_bytes(1024 * 1024 * 1024);

        assert!((config.gpu_memory_fraction - 0.8).abs() < 0.001);
        assert!(!config.cpu_fallback_enabled);
        assert_eq!(config.min_free_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_allocation_tracker() {
        let tracker = AllocationTracker::new();

        let meta1 = BufferMetadata::new(BufferId::new(), 1024, 256, BufferLocation::Gpu);
        let id1 = meta1.id;
        tracker.track(meta1);

        let meta2 = BufferMetadata::new(BufferId::new(), 2048, 512, BufferLocation::Gpu);
        let id2 = meta2.id;
        tracker.track(meta2);

        assert_eq!(tracker.count(), 2);
        assert_eq!(tracker.total_bytes(), 3072);

        tracker.untrack(id1);
        assert_eq!(tracker.count(), 1);
        assert_eq!(tracker.total_bytes(), 2048);

        tracker.untrack(id2);
        assert_eq!(tracker.count(), 0);
        assert_eq!(tracker.total_bytes(), 0);
    }

    #[test]
    fn test_generic_memory_pool() {
        let config = MemoryConfig::default();
        let mut pool: GenericMemoryPool<Vec<u8>> = GenericMemoryPool::new(config);

        // Simulate allocations
        pool.record_allocation(1024);
        pool.record_allocation(2048);

        let stats = pool.stats();
        assert_eq!(stats.total_allocated_bytes, 3072);
        assert_eq!(stats.in_use_bytes, 3072);

        // Return a buffer
        pool.return_buffer(vec![0u8; 1024], 1024);

        let stats = pool.stats();
        assert_eq!(stats.in_use_bytes, 2048);
        assert_eq!(stats.pooled_bytes, 1024);

        // Try to get from pool
        let result = pool.try_get(1024);
        assert!(result.is_some());

        let stats = pool.stats();
        assert_eq!(stats.pool_hits, 1);
    }

    #[test]
    fn test_inference_memory_estimate() {
        let estimate = estimate_inference_memory(
            4096,  // hidden_size
            14336, // intermediate_size
            32,    // n_layers
            32000, // vocab_size
            32,    // n_heads
            8,     // n_kv_heads
            4096,  // max_seq_len
            4,     // dtype_bytes (f32)
        );

        // Sanity check - should be in reasonable range for 7B model
        assert!(estimate.total_gb() > 20.0);
        assert!(estimate.total_gb() < 40.0);
    }

    #[test]
    fn test_memory_stats_display() {
        let stats = MemoryStats {
            total_allocated_bytes: 1024 * 1024 * 1024, // 1 GB
            in_use_bytes: 512 * 1024 * 1024,           // 512 MB
            pooled_bytes: 256 * 1024 * 1024,           // 256 MB
            peak_bytes: 768 * 1024 * 1024,             // 768 MB
            pool_hits: 100,
            pool_misses: 10,
            cpu_fallbacks: 2,
            active_buffers: 50,
        };

        let display = format!("{}", stats);
        assert!(display.contains("1024.00 MB"));
        assert!(display.contains("90.9%")); // hit rate
    }

    #[test]
    fn test_fallback_buffer() {
        // GPU buffer
        let gpu_buf: FallbackBuffer<Vec<f32>> = FallbackBuffer::gpu(vec![1.0, 2.0, 3.0], 3);
        assert!(gpu_buf.is_gpu());
        assert!(!gpu_buf.is_cpu_fallback());
        assert_eq!(gpu_buf.len(), 3);

        // CPU fallback buffer
        let cpu_buf: FallbackBuffer<Vec<f32>> = FallbackBuffer::cpu(vec![4.0, 5.0, 6.0, 7.0]);
        assert!(!cpu_buf.is_gpu());
        assert!(cpu_buf.is_cpu_fallback());
        assert_eq!(cpu_buf.len(), 4);
    }
}
