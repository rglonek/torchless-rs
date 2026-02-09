# Parameter Reference

This document explains the CLI flags, in-session chat commands, and coding mode for torchless.

## Parameters

### `--max-seq-len <N>` -- Context Window Size

**Default:** The model's trained context length (`max_position_embeddings` from model metadata).

The maximum number of tokens the model can "see" at once -- prompt, conversation history, and generated output combined. This directly controls memory allocation: the KV cache is sized to hold this many positions.

**Typical values by model:**

| Model | Trained context (`max_position_embeddings`) |
|-------|----------------------------------------------|
| Mistral 7B | 32,768 |
| LLaMA 2 7B | 4,096 |
| LLaMA 3 8B | 8,192 |
| Phi-3 mini | 4,096 (or 128,000 for the long variant) |
| Qwen 2 7B | 32,768 |

Setting `--max-seq-len` higher than the model's trained limit will produce a warning at startup. The model's RoPE positional embeddings were not trained beyond that length, so output quality degrades for positions past the trained limit. When in doubt, leave this at the default.

Setting it lower than the model's limit is fine and saves memory. For example, `--max-seq-len 4096` on a Mistral 7B model uses less RAM but limits your conversation length.

### `--max-tokens <N>` -- Maximum Response Length

**Default:** `max_seq_len / 4`, clamped to a minimum of 2,048 (or 1,024 if `max_seq_len < 4096`) and a maximum of 32,768.

The maximum number of tokens the model will generate **per response**. In chat mode, generation also stops early if the model produces an EOS (end-of-sequence) token, so this acts as an upper bound rather than a target.

**Examples of what fits in different budgets:**

| `--max-tokens` | Approximate output | Good for |
|----------------|-------------------|----------|
| 1,024 | ~750 words | Short answers, quick lookups |
| 2,048 | ~1,500 words | Explanations, short code |
| 4,096 | ~3,000 words | Detailed code, longer reasoning |
| 8,192 | ~6,000 words | Full files, comprehensive analysis |

### `--temperature <T>` -- Sampling Randomness

**Default:** `0.7`

Controls how "random" the model's token choices are. The temperature value scales the model's raw prediction scores (logits) before sampling:

| Temperature | Behavior | Use case |
|-------------|----------|----------|
| 0.0 | **Greedy/deterministic.** Always picks the most likely token. Same input produces the same output every time. | Factual tasks, reproducible output |
| 0.1 - 0.3 | Low randomness. Mostly follows the highest-probability path with occasional variation. | Code generation, structured output |
| 0.5 - 0.7 | Balanced. Varied but coherent. | General chat, explanations |
| 0.8 - 1.0 | Creative. More diverse word choices, occasionally surprising. | Brainstorming, creative writing |
| > 1.0 | Increasingly chaotic. Rarely useful. | Experimental only |

Mathematically: each logit is divided by the temperature before softmax. Lower temperature sharpens the distribution (concentrating probability on top choices); higher temperature flattens it (spreading probability more evenly).

### `--top-k <K>` -- Top-K Sampling

**Default:** Disabled (all tokens considered).

Limits the model to choosing only from the `K` most likely tokens at each step. All other tokens are masked out before sampling. This prevents the model from occasionally picking very unlikely (and often nonsensical) tokens, which can happen at higher temperatures.

| `--top-k` | Behavior |
|-----------|----------|
| _disabled_ | All tokens in the vocabulary are candidates (default) |
| 10 | Very restrictive -- only the 10 most likely tokens. Repetitive output likely. |
| 40 | Moderate -- good balance for focused generation |
| 50-100 | Typical sweet spot -- diverse but coherent |
| 200+ | Very permissive -- rarely has practical effect |

Top-k filtering is applied **before** softmax and before top-p. When used with `--top-p`, both filters are applied in sequence: top-k narrows the candidate set first, then top-p further trims by cumulative probability.

### `--top-p <P>` -- Nucleus (Top-P) Sampling

**Default:** Disabled (full distribution used).

Keeps only the smallest set of tokens whose cumulative probability is at least `P`, then re-normalizes and samples from that set. Unlike top-k, this adapts to the model's confidence: when the model is very sure, only a few tokens are kept; when it's uncertain, more tokens remain.

| `--top-p` | Behavior |
|-----------|----------|
| _disabled_ | Full probability distribution used (default) |
| 0.5 | Restrictive -- only the most confident half of the distribution |
| 0.9 | Standard -- widely used default in many inference frameworks |
| 0.95 | Permissive -- retains most of the probability mass |
| 1.0 | No effect (all tokens kept, equivalent to disabled) |

Top-p filtering is applied **after** softmax and after top-k (if enabled). The typical recommendation is to use top-p **or** top-k, not both, but combining them is supported.

**Combining temperature with top-k/top-p:**

Temperature, top-k, and top-p are applied in this order:

1. **Temperature** scales the raw logits (broadening or sharpening the distribution).
2. **Top-k** masks out all but the top K tokens by logit value.
3. **Softmax** converts logits to probabilities.
4. **Top-p** trims the distribution to the smallest set with cumulative probability >= P.
5. **Sample** a token from the resulting distribution.

A common recipe for high-quality generation:

```bash
--temperature 0.7 --top-p 0.9
```

### `--lazy` -- Lazy Model Loading

**Default:** Off (eager loading).

Controls whether the model weights are loaded entirely into memory at startup (eager) or loaded on demand during inference (lazy).

| Mode | RAM usage (7B) | Speed | Best for |
|------|----------------|-------|----------|
| Eager (default) | ~15 GB (FP16) / ~5 GB (Q4) | ~1 tok/s | Machines with plenty of RAM; fastest inference |
| Lazy (`--lazy`) | <2 GB base + KV cache | ~0.5 tok/s | Memory-constrained machines; exploring models |

**How it works:** In lazy mode, the model weights remain memory-mapped from disk. Each layer's tensors are loaded into working memory only when needed for the forward pass, then released. This dramatically reduces peak RAM usage at the cost of repeated I/O — roughly 2x slower on CPU.

Use `--lazy` when:
- Your machine doesn't have enough RAM to hold the full model (e.g., running a 7B model on a 16 GB laptop).
- You want to quickly test a model without waiting for a full load.
- You're running multiple models and need to minimize per-model memory footprint.

### `--kv-dtype <TYPE>` -- KV Cache Precision

**Default:** `f16` (half precision).

Controls the data type used for the attention key-value cache. FP16 halves KV cache memory compared to FP32 with negligible quality impact.

| `--kv-dtype` | Bytes/element | Memory (Mistral 7B, 32K ctx) | Quality |
|--------------|---------------|------------------------------|---------|
| `f16` (default) | 2 | ~4 GB | Excellent -- negligible difference from f32 |
| `f32` | 4 | ~8 GB | Baseline |

The KV cache precision is independent of the model's weight format. A Q4_K_M model still benefits from FP16 KV cache.

**When to use `--kv-dtype f32`:**
- Debugging quality issues in very long sequences
- Comparing against a reference implementation
- Models that are particularly sensitive to KV precision (rare)

### `--speculative` -- Self-Speculative Decoding

**Default:** Off.

Enables self-speculative decoding, which can accelerate generation by 1.5-3x on CPU. The same model is used at two different temperatures: a higher temperature for fast "draft" proposals, and the normal temperature for verification. Multiple tokens are proposed per iteration, and only those that pass verification are accepted.

| Setting | Behavior |
|---------|----------|
| Off (default) | Standard autoregressive generation (1 forward pass per token) |
| `--speculative` | Draft-then-verify: generates multiple tokens per iteration |

The decoder uses adaptive speculation length (starting at 5 tokens, ranging 2-8) and adjusts based on acceptance rate. When `--debug` is enabled, speculative decoding statistics are printed after generation:

```
[speculative: acceptance_rate=72.3%, tokens/iteration=3.6]
```

**When to use:**
- CPU inference where generation speed is the bottleneck.
- Longer responses (the overhead is amortized over many tokens).
- Models where the probability distribution is relatively peaky (higher acceptance rates).

**Limitations:**
- `--top-k` and `--top-p` are ignored in speculative mode. Only temperature-based sampling is used for draft/verify.
- Not yet compatible with two-model speculative decoding (separate draft model). The library supports this, but the CLI only exposes the self-speculative variant. See [Future Parameters](params-future.md) for details.

### `--show-thinking` -- Show Reasoning Traces

**Default:** Off (thinking traces hidden).

For thinking models (DeepSeek-R1 distilled variants, QwQ, full DeepSeek-R1), this flag shows the `<think>`...`</think>` reasoning traces inline during generation. Traces appear dimmed in the terminal.

Thinking is **auto-detected** from the model's vocabulary -- no flag is needed to *enable* thinking, only to *display* it. You can also toggle visibility at runtime with the `/thinking` chat command.

### `--chat` -- Interactive Chat Mode

Enters an interactive multi-turn conversation REPL. Supports `/commands` for changing settings at runtime, saving/loading sessions, coding mode, and more. See [Chat Commands](#chat-commands) below.

### `--system <MSG>` -- System Prompt

**Default:** None.

Sets the initial system prompt for chat mode. Can be changed at runtime with `/system`.

### `--debug` -- Debug Output

**Default:** Off.

Enables verbose debug output: token processing progress, KV cache reuse stats, speculative decoding acceptance rates, EOS detection, and more. Toggleable at runtime with `/debug`.

### `--socket <PATH>` -- Unix Socket Server (Unix only)

Starts a multi-user chat server listening on a Unix domain socket at `PATH`. Requires `--chat-save-root`.

### `--chat-save-root <DIR>` -- Per-User Save Directory

Required with `--socket`. Sets the root directory for per-user chat session saves.

## How `max-seq-len` and `max-tokens` Interact

The context window must hold both the conversation history **and** the model's response:

```
max_seq_len = conversation_history + max_tokens
```

In chat mode, `trim_to_fit` reserves `max_tokens` worth of space for generation. If the conversation grows beyond the remaining budget, the oldest messages are dropped (the system prompt is preserved). This means:

- If `max_tokens` is too close to `max_seq_len`, most of the context window is reserved for generation, leaving little room for conversation history.
- If `max_tokens == max_seq_len`, the history budget is **zero** -- the conversation would be trimmed to nothing every turn.

### Choosing good values

| `max-seq-len` | `max-tokens` | History budget | Assessment |
|---------------|-------------|----------------|------------|
| 32,768 | 32,768 | 0 | **Bad** -- no room for any conversation history |
| 32,768 | 16,384 | 16,384 | Borderline -- half the context wasted on generation reserve |
| 32,768 | 8,192 | 24,576 | Good -- generous responses, plenty of history |
| 32,768 | 4,096 | 28,672 | Good -- large history, room for detailed responses |
| 32,768 | 2,048 | 30,720 | Good -- maximizes conversation memory |
| 4,096 | 1,024 | 3,072 | Reasonable for small models |

**Rule of thumb:** Set `max-tokens` to the longest single response you'd want. For most chat, 2,048-4,096 is plenty. For coding tasks where the model may produce full files, 4,096-8,192 is safer. The default (`max_seq_len / 4`) provides a good balance automatically.

## Examples

```bash
# Use all defaults (context from model, max-tokens = context/4, temperature = 0.7)
./torchless --chat model.bin

# Explicit coding setup: full context, generous response room
./torchless --chat --max-seq-len 32768 --max-tokens 8192 --temperature 0.3 model.bin

# Deterministic single-shot completion
./torchless --temperature 0.0 --max-tokens 200 model.bin "The capital of France is"

# Creative writing with large response budget
./torchless --chat --max-tokens 4096 --temperature 0.9 model.bin

# Memory-constrained: small context window
./torchless --chat --max-seq-len 4096 model.bin

# Full-precision KV cache (debugging / reference comparison)
./torchless --kv-dtype f32 --chat model.bin

# Lazy loading on a memory-constrained machine
./torchless --lazy --chat model.bin

# Top-k + temperature for focused creative writing
./torchless --chat --temperature 0.8 --top-k 50 model.bin

# Nucleus sampling for balanced generation
./torchless --chat --temperature 0.7 --top-p 0.9 model.bin

# Combine top-k and top-p
./torchless --lazy --top-k 50 --top-p 0.9 model.bin "Tell me a story"

# Speculative decoding for faster CPU inference
./torchless --speculative model.bin "Explain quantum computing"

# Speculative + debug to see acceptance statistics
./torchless --speculative --debug --chat model.bin

# Full setup: lazy loading, nucleus sampling, chat with system prompt
./torchless --lazy --top-p 0.95 --temperature 0.6 --chat --system "You are a coding assistant." model.bin

# Thinking model with visible reasoning
./torchless --chat --show-thinking deepseek-r1-distill-qwen-7b.gguf
```

---

## Chat Commands

In `--chat` mode, type `/help` to see all commands. Commands start with `/` and are processed locally (not sent to the model). Anything else is sent as a message.

### Session Control

| Command | Description |
|---------|-------------|
| `/quit`, `/exit`, `/q` | Exit chat |
| `/clear` | Clear conversation history and reset KV cache |
| `/retry` | Remove the last assistant response and regenerate |
| `/context` | Show token usage, context headroom, and conversation stats |
| `/settings` | Show all current runtime settings |

### Sampling & Generation

These modify settings for the current session. Changes take effect on the next response.

| Command | Description |
|---------|-------------|
| `/temperature <T>` | Set sampling temperature (0.0 = greedy, 1.0+ = creative) |
| `/top-k <K\|off>` | Set or disable top-k sampling |
| `/top-p <P\|off>` | Set or disable nucleus (top-p) sampling |
| `/speculative` | Toggle self-speculative decoding on/off |
| `/max-tokens <N>` | Set max tokens per response |
| `/debug` | Toggle debug output |
| `/system <MSG\|off>` | Set, change, or remove the system prompt (resets KV cache) |
| `/thinking [on\|off]` | Toggle or set thinking trace visibility (thinking models only) |

### Save / Load

Conversations can be saved to and loaded from JSON files.

| Command | Description |
|---------|-------------|
| `/save <file>` | Save conversation history to a JSON file |
| `/fullsave <file>` | Save conversation history **and** current settings (temperature, top-k, etc.) |
| `/load <file>` | Load a conversation (and optional settings) from a JSON file |

### Coding Mode

Coding mode enables structured file-edit proposals from the model. When active, the model uses SEARCH/REPLACE blocks to propose changes, and you review/apply them interactively.

| Command | Description |
|---------|-------------|
| `/code [on\|off]` | Toggle coding mode (appends coding instructions to system prompt, resets KV cache) |
| `/diff` | Show all pending file edits proposed by the model |
| `/apply [all\|N]` | Apply pending edits to files (prompts for confirmation per edit) |
| `/discard [all\|N]` | Discard pending edits without applying |

**How coding mode works:**

1. Enable with `/code on` (or `/code` to toggle).
2. Reference files in your messages using `@filepath` syntax (e.g., `@src/main.rs` or `@src/main.rs:10-50` for a line range). The file contents are expanded inline before sending to the model.
3. The model proposes changes as SEARCH/REPLACE blocks.
4. Review with `/diff`, then apply with `/apply` (each edit requires `y/N` confirmation) or discard with `/discard`.

```
> /code on
Coding mode: on. Use @file references and the model will propose edits. KV cache reset.

> Fix the off-by-one error in @src/parser.rs:42-60
[1 edit(s) proposed. Use /diff to review, /apply to apply.]

> /diff
Edit 1: src/parser.rs
--- SEARCH ---
    if index >= items.len() {
--- REPLACE ---
    if index > items.len() {

> /apply
Apply edit 1/1? [y/N]: y
Applied.
```

### `@file` References

You can use `@filepath` in any message (not just coding mode) to include file contents in your prompt. Supported formats:

| Syntax | Effect |
|--------|--------|
| `@path/to/file` | Include the entire file |
| `@path/to/file:10-50` | Include lines 10 through 50 |

File paths are resolved relative to the current working directory.
