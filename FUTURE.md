# Future Optimizations Roadmap

This document tracks remaining optimizations and improvements that have not yet been implemented.

## Priority Legend

- 🔴 **High Impact** - Significant performance or usability improvement
- 🟡 **Medium Impact** - Moderate improvement, good ROI
- 🟢 **Low Impact** - Nice to have, incremental improvement

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
