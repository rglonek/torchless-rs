# Future Optimizations Roadmap

This document tracks remaining optimizations and improvements that have not yet been implemented.

## Priority Legend

- 🔴 **High Impact** - Significant performance or usability improvement
- 🟡 **Medium Impact** - Moderate improvement, good ROI
- 🟢 **Low Impact** - Nice to have, incremental improvement

---

## WebGPU Backend

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

## Summary

| Item | Priority | Status |
|------|----------|--------|
| WebGPU Backend | 🟢 Low | Not started |

---

## Contributing

If you're interested in implementing any of these:

1. Open an issue to discuss the approach
2. Reference existing implementations as guides
3. Add tests to verify correctness
4. Submit PR with documentation updates

See existing GPU backends (`src/kernels/cuda/`, `src/kernels/metal/`) as reference implementations.
