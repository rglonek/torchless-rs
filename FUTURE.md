# Future Optimizations Roadmap

This document tracks remaining optimizations and improvements that have not yet been implemented.

## Priority Legend

- 🔴 **High Impact** - Significant performance or usability improvement
- 🟡 **Medium Impact** - Moderate improvement, good ROI
- 🟢 **Low Impact** - Nice to have, incremental improvement

---

## 1. GPU Unified Tensor Transfer

### 🔴 DeviceTransfer Implementation for UnifiedTensor

The GPU backends (CUDA, Metal, ROCm, OpenCL) have full kernel implementations, but the `DeviceTransfer` trait for `UnifiedTensor` is not yet complete. Currently, GPU operations require explicit CPU-GPU data copies rather than automatic tensor transfer.

**Current state:**
- ✅ GPU kernel implementations (matmul, rmsnorm, softmax, silu, rope, attention)
- ✅ GPU memory pools and buffer management
- ❌ `UnifiedTensor::to_device()` for GPU targets (returns "not yet implemented")
- ❌ Automatic data movement for quantized tensor types on GPU

**Implementation needed:**
- [ ] Implement `DeviceTransfer::to_device()` for CUDA device
- [ ] Implement `DeviceTransfer::to_device()` for Metal device
- [ ] Implement `DeviceTransfer::to_device()` for ROCm device
- [ ] Implement `DeviceTransfer::to_device()` for OpenCL device
- [ ] Add GPU-side storage for quantized types (F16, BF16, Int8, Int4)
- [ ] Implement automatic tensor synchronization between CPU and GPU

**Location:** `src/tensor/storage.rs` - `DeviceTransfer` trait implementation

**Impact:** Enables seamless model loading to GPU without manual data management

---

## 2. ROCm Device Enum Integration

### 🟡 Add ROCm to Device Enum

The `Device` enum in `src/tensor/storage.rs` is missing the `Rocm` variant, despite the ROCm backend being implemented.

**Current state:**
```rust
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal(usize),
    OpenCL(usize),
    // Missing: Rocm(usize)
}
```

**Implementation needed:**
- [ ] Add `Rocm(usize)` variant to `Device` enum
- [ ] Update `DeviceTransfer` trait to handle `Rocm` device
- [ ] Update any match statements that use `Device`

**Location:** `src/tensor/storage.rs`

**Impact:** Completes AMD GPU support integration

---

## 3. WebGPU Backend

### 🟢 Cross-Platform GPU via wgpu

Add WebGPU support for cross-platform GPU acceleration.

**Benefits:**
- Works across all platforms (Windows, Linux, macOS)
- Potential future browser deployment
- Standardized modern GPU API

**Implementation needed:**
- [ ] Add `wgpu` crate dependency
- [ ] Implement `WebGPUBackend` struct
- [ ] Write WGSL compute shaders for core operations:
  - [ ] matmul_vec kernel
  - [ ] rmsnorm kernel
  - [ ] softmax kernel
  - [ ] silu kernel
  - [ ] rope kernel
  - [ ] attention kernels
- [ ] Add buffer management
- [ ] Integrate with `init_backend()` selection

**Reference implementations:** See `src/kernels/cuda/mod.rs` and `src/kernels/metal/mod.rs`

**Estimated effort:** Medium - similar structure to existing GPU backends

---

## 4. Profile-Guided Optimization (PGO)

### 🟡 Complete PGO Workflow

The `Cargo.toml` has PGO-friendly settings, but the full workflow requires documentation and scripting.

**Current state:**
- ✅ Release profile optimized (`lto = "fat"`, `codegen-units = 1`)
- ❌ No automated PGO build script
- ❌ No documented PGO workflow

**Implementation needed:**
- [ ] Create `scripts/build_pgo.sh` automation script
- [ ] Document PGO workflow in `docs/optimization/build.md`
- [ ] Add representative profiling workload
- [ ] Add CI job for PGO builds (optional)

**PGO workflow:**
```bash
# Step 1: Build with profiling instrumentation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release

# Step 2: Run representative workload
./target/release/torchless model.bin "test prompt" --max-tokens 100

# Step 3: Merge profile data
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# Step 4: Build with profile data
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release
```

**Expected speedup:** 5-15%

---

## Summary

| Item | Priority | Status |
|------|----------|--------|
| GPU DeviceTransfer for UnifiedTensor | 🔴 High | Framework ready, transfer pending |
| ROCm Device enum | 🟡 Medium | Backend ready, enum missing |
| WebGPU Backend | 🟢 Low | Not started |
| PGO Workflow Automation | 🟡 Medium | Config ready, scripts pending |

---

## Contributing

If you're interested in implementing any of these:

1. Open an issue to discuss the approach
2. Reference existing implementations as guides
3. Add tests to verify correctness
4. Submit PR with documentation updates

See existing GPU backends (`src/kernels/cuda/`, `src/kernels/metal/`) as reference implementations.
