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

## Mixture-of-Experts (MoE) Architecture

### 🔴 MoE Model Support for Full DeepSeek-R1 / V3

Add Mixture-of-Experts architecture support to unlock full-size reasoning models like DeepSeek-R1 (671B), DeepSeek-V3, and Mistral Large 3.

Currently, only **distilled** thinking models are supported (e.g., DeepSeek-R1-Distill-Qwen-7B) because they use standard dense architectures. The full MoE models require a fundamentally different forward pass with expert routing.

**Benefits:**
- Run DeepSeek-R1 (671B) -- one of the strongest open reasoning models
- Run DeepSeek-V3 -- frontier-class open model
- Run Mistral Large 3 (675B) -- strong multilingual reasoning
- Only a fraction of parameters are active per token (e.g., 37B of 671B), making MoE models more efficient per-FLOP than equivalent dense models

**Implementation needed:**
- [ ] Add `MoE` variant to `ModelArchitecture` enum
- [ ] Implement expert gating / router module (top-k routing with learned gate weights)
- [ ] Implement sparse FFN layer (select top-k experts per token, run only those FFNs)
- [ ] Add load balancing loss tracking (for monitoring routing quality)
- [ ] Support shared experts (DeepSeek uses 1 shared + N routed experts)
- [ ] Add MoE-aware memory estimation (active params vs total params)
- [ ] Handle MoE tensor naming patterns in architecture detection
- [ ] Add lazy loading support for MoE (critical -- loading all experts eagerly would require >1TB RAM)
- [ ] Add tests with small synthetic MoE model

**Key models this would unlock:**

| Model | Active / Total Params | Experts | Context | Notes |
|-------|----------------------|---------|---------|-------|
| DeepSeek-R1 | ~37B / ~671B | 8 of 256 | 128,000 | Thinking model with MoE |
| DeepSeek-V3 | ~37B / ~671B | 8 of 256 | 128,000 | General-purpose MoE |
| Mistral Large 3 | 41B / 675B | top-2 of 16 | 256,000 | Strong reasoning |
| LLaMA 4 Scout | 17B / 109B | top-1 of 16 | 10,000,000 | Largest context window |
| LLaMA 4 Maverick | 17B / 400B | top-1 of 128 | 1,000,000 | Multimodal |

**Reference:** DeepSeek-V3 technical report describes the MoE routing mechanism in detail.

**Estimated effort:** Large - new architecture type with significant changes to forward pass, memory management, and tensor loading.

---

## Deferred Chat Session Commands

These in-session commands require significant architectural changes and are deferred for future implementation.

### 🟡 `/lazy` -- Toggle Lazy/Eager Loading Mid-Session

Toggle between lazy (mmap) and eager (full load) model loading at runtime.

**Why it's hard:** The model is currently loaded as either `Mistral` (eager) or `LazyMistral<'a>` (lazy), which are distinct types with different lifetimes. Switching at runtime would require a trait-object or enum-based model abstraction that unifies both types behind a common interface. The closures passed to `run_chat_repl` capture the model by reference with type-specific lifetimes, so changing the model type would invalidate all existing closure captures and the KV cache.

**Implementation needed:**
- [ ] Create a `ModelDispatch` enum or trait object wrapping `Mistral` / `LazyMistral`
- [ ] Update `run_chat_repl` to accept the abstraction instead of model-specific closures
- [ ] Handle full model reload (free old weights, load new, reset KV cache)

### 🟡 `/model <path>` -- Hot-Swap Model File

Load a different model file without restarting the session.

**Why it's hard:** Requires a full model reload (new parameters, new tokenizer, new config), which means all existing KV cache state is invalidated. It has the same type-level challenges as `/lazy` -- the new model may have a different architecture, vocabulary size, or embedding dimension, making the existing `InferenceState` incompatible. The tokenizer change means the entire conversation history must be re-tokenized.

**Implementation needed:**
- [ ] All items from `/lazy` above (model abstraction)
- [ ] Architecture detection and config validation for the new model
- [ ] Re-tokenize conversation history with new tokenizer
- [ ] Allocate new `InferenceState` matching new model dimensions
- [ ] Graceful error handling if new model fails to load (keep old model)

### 🟡 `/template <name>` -- Change Chat Template

Switch between chat templates (Mistral/LLaMA/Phi/Gemma/Qwen) mid-session.

**Why it's hard:** Each template formats the conversation differently (different special tokens, role markers, BOS/EOS sequences). Changing the template requires re-tokenizing the entire conversation history under the new format, which may produce a different token count that exceeds the context window. The EOS token IDs also change per template, and the KV cache must be fully reset since the token sequence changes.

**Implementation needed:**
- [ ] Re-tokenize all messages in `history` using the new template format
- [ ] Check that re-tokenized conversation fits within `max_seq_len`
- [ ] Update `eos_ids` for the new template
- [ ] Reset KV cache and `processed_tokens`
- [ ] Reprocess entire conversation through the model

---

## Summary

| Item | Priority | Status |
|------|----------|--------|
| MoE Architecture | 🔴 High | Not started |
| WebGPU Backend | 🟢 Low | Not started |
| `/lazy` command | 🟡 Medium | Not started |
| `/model` command | 🟡 Medium | Not started |
| `/template` command | 🟡 Medium | Not started |

---

## Contributing

If you're interested in implementing any of these:

1. Open an issue to discuss the approach
2. Reference existing implementations as guides
3. Add tests to verify correctness
4. Submit PR with documentation updates

See existing GPU backends (`src/kernels/cuda/`, `src/kernels/metal/`) as reference implementations.
