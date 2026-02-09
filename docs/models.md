# Supported Models & Memory Requirements

Torchless-rs supports transformer models in the **Mistral**, **LLaMA**, **Phi**, **Gemma**, **Qwen**, and **DeepSeek** families -- including both dense and **Mixture-of-Experts (MoE)** architectures. This page lists tested models, their specifications, and the memory needed to run them in each mode.

---

## Small Models (Consumer Hardware)

These models can run on a typical desktop or laptop.

### Specifications

| Model | Family | Parameters | Trained Context | GQA | Vocab |
|-------|--------|------------|-----------------|-----|-------|
| Gemma 2B | Gemma | 2.5B | 8,192 | 1 KV head | 256,128 |
| Phi-3 Mini | Phi | 3.8B | 4,096 | No (32 KV) | 32,064 |
| Phi-3 Mini Long | Phi | 3.8B | 128,000 | No (32 KV) | 32,064 |
| LLaMA 2 7B | LLaMA | 6.7B | 4,096 | No (32 KV) | 32,000 |
| Mistral 7B | Mistral | 7.2B | 32,768 | 8 KV heads | 32,000 |
| Qwen 2 7B | Qwen | 7.6B | 32,768 | 4 KV heads | 151,936 |
| Qwen 2.5 7B | Qwen | 7.6B | 131,072 | 4 KV heads | 152,064 |
| LLaMA 3 8B | LLaMA | 8.0B | 8,192 | 8 KV heads | 128,256 |
| LLaMA 3.1 8B | LLaMA | 8.0B | 131,072 | 8 KV heads | 128,256 |
| Gemma 7B | Gemma | 8.2B | 8,192 | 16 KV heads | 256,128 |

### Model Weight Memory

Memory required to store model weights at each quantization level (weights only, excludes KV cache and activations).

| Model | FP32 | FP16 | Q8_0 | Q4_K_M |
|-------|------|------|------|--------|
| Gemma 2B | ~10 GB | ~5 GB | ~2.7 GB | ~1.4 GB |
| Phi-3 Mini | ~15 GB | ~7.6 GB | ~4.1 GB | ~2.1 GB |
| LLaMA 2 7B | ~27 GB | ~13 GB | ~7.1 GB | ~3.8 GB |
| Mistral 7B | ~29 GB | ~14.5 GB | ~7.7 GB | ~4.1 GB |
| Qwen 2 / 2.5 7B | ~30 GB | ~15 GB | ~8.1 GB | ~4.3 GB |
| LLaMA 3 / 3.1 8B | ~32 GB | ~16 GB | ~8.5 GB | ~4.5 GB |
| Gemma 7B | ~33 GB | ~16.4 GB | ~8.7 GB | ~4.6 GB |

### KV Cache Memory (FP32)

The KV cache stores attention keys and values for all layers. It is always allocated in FP32 and scales linearly with sequence length. Using `--max-seq-len` controls this.

| Model | Context | KV Heads | KV Cache |
|-------|---------|----------|----------|
| Gemma 2B | 8,192 | 1 | 0.3 GB |
| LLaMA 3 / 3.1 8B | 8,192 | 8 | 2.0 GB |
| Phi-3 Mini | 4,096 | 32 | 3.0 GB |
| Qwen 2 7B | 32,768 | 4 | 3.5 GB |
| LLaMA 2 7B | 4,096 | 32 | 4.0 GB |
| Gemma 7B | 8,192 | 16 | 5.3 GB |
| Mistral 7B | 32,768 | 8 | 8.0 GB |

> **Note:** Models with GQA (fewer KV heads) use dramatically less KV cache memory. Mistral 7B and LLaMA 3 use 8 KV heads; Qwen 2 uses 4. LLaMA 2 and Phi-3 use full multi-head attention (32 KV heads) which costs more.

For large-context models at full context:

| Model | Context | KV Cache (FP32) |
|-------|---------|-----------------|
| Qwen 2.5 7B | 131,072 | 14 GB |
| LLaMA 3.1 8B | 131,072 | 32 GB |
| Phi-3 Mini Long | 128,000 | 94 GB |

> **Tip:** Use `--max-seq-len 4096` to limit KV cache allocation when you don't need the full trained context. This saves significant memory, especially for large-context models.

---

## Total Memory by Run Mode

These tables show the **total memory** required (model weights + KV cache + ~0.5 GB activations/overhead) for each run mode.

### At Default Context (2,048 tokens)

This is with `--max-seq-len 2048` or the model default if smaller.

| Model | Eager CPU (FP32) | Lazy CPU | GPU FP16 | GPU Q4_K_M |
|-------|-------------------|----------|----------|------------|
| Gemma 2B | ~11 GB | ~2 GB | ~6 GB | ~2 GB |
| Phi-3 Mini | ~17 GB | ~3 GB | ~10 GB | ~4 GB |
| LLaMA 2 7B | ~30 GB | ~4 GB | ~16 GB | ~6 GB |
| Mistral 7B | ~30 GB | ~2 GB | ~16 GB | ~5 GB |
| Qwen 2 / 2.5 7B | ~31 GB | ~2 GB | ~16 GB | ~5 GB |
| LLaMA 3 / 3.1 8B | ~33 GB | ~2 GB | ~17 GB | ~5 GB |
| Gemma 7B | ~35 GB | ~3 GB | ~18 GB | ~6 GB |

### At Full Trained Context

Using the model's full `max_position_embeddings` context window.

| Model | Context | Eager CPU (FP32) | Lazy CPU | GPU FP16 | GPU Q4_K_M |
|-------|---------|-------------------|----------|----------|------------|
| Gemma 2B | 8,192 | ~11 GB | ~2 GB | ~6 GB | ~2 GB |
| Phi-3 Mini | 4,096 | ~19 GB | ~5 GB | ~11 GB | ~6 GB |
| LLaMA 2 7B | 4,096 | ~32 GB | ~6 GB | ~18 GB | ~9 GB |
| LLaMA 3 8B | 8,192 | ~35 GB | ~4 GB | ~19 GB | ~7 GB |
| Gemma 7B | 8,192 | ~39 GB | ~7 GB | ~22 GB | ~11 GB |
| Mistral 7B | 32,768 | ~38 GB | ~10 GB | ~23 GB | ~13 GB |
| Qwen 2 7B | 32,768 | ~34 GB | ~5 GB | ~19 GB | ~8 GB |
| Qwen 2.5 7B | 131,072 | ~45 GB | ~16 GB | ~30 GB | ~19 GB |
| LLaMA 3.1 8B | 131,072 | ~65 GB | ~34 GB | ~49 GB | ~37 GB |

### Run Mode Descriptions

| Mode | Description | Best For |
|------|-------------|----------|
| **Eager CPU (FP32)** | All weights loaded into RAM as FP32. Fastest CPU inference. | High-memory systems, best throughput on CPU |
| **Lazy CPU** | Weights stay memory-mapped on disk; loaded on-demand. ~1-1.5 GB base overhead. | Memory-constrained systems, laptops |
| **GPU FP16** | Weights in VRAM as FP16. KV cache still FP32. | NVIDIA/AMD/Apple GPUs with ≥16 GB VRAM |
| **GPU Q4_K_M** | Weights quantized to 4-bit in VRAM. KV cache still FP32. | GPUs with 4-8 GB VRAM |

---

## Large Models (Server Hardware)

These models require server-grade hardware. Torchless-rs supports them (they use the same dense architectures) but they need significantly more resources.

| Model | Family | Parameters | Trained Context | FP16 Weights | Q4_K_M Weights |
|-------|--------|------------|-----------------|--------------|----------------|
| Qwen 2.5 14B | Qwen | ~14.2B | 131,072 | ~28 GB | ~8 GB |
| Qwen 2.5 32B | Qwen | ~32.5B | 131,072 | ~65 GB | ~18 GB |
| LLaMA 3.1 70B | LLaMA | ~70.6B | 131,072 | ~141 GB | ~40 GB |
| Qwen 2.5 72B | Qwen | ~72.7B | 131,072 | ~145 GB | ~41 GB |
| LLaMA 2 70B | LLaMA | ~68.9B | 4,096 | ~138 GB | ~39 GB |

> **KV cache for 70B-class models at 131K context:** ~160-250 GB (FP32). Use `--max-seq-len` to limit this or run with a shorter context.

---

## Thinking / Reasoning Models

Torchless-rs supports thinking models that use `<think>`...`</think>` token pairs to emit chain-of-thought reasoning before their final answer. Both **distilled dense** variants (Qwen/LLaMA-based) and the **full MoE** models (DeepSeek-R1, DeepSeek-V3) are supported.

### Distilled Thinking Models (Dense)

These use the same dense architectures as their base families and run without any special configuration.

| Model | Base Architecture | Parameters | Context | Notes |
|-------|-------------------|------------|---------|-------|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen | 1.5B | 131,072 | Smallest distilled reasoning model |
| DeepSeek-R1-Distill-Qwen-7B | Qwen | 7.6B | 131,072 | Good balance of speed and reasoning |
| DeepSeek-R1-Distill-Qwen-14B | Qwen | ~14.2B | 131,072 | Strong reasoning quality |
| DeepSeek-R1-Distill-Qwen-32B | Qwen | ~32.5B | 131,072 | Near-frontier reasoning |
| DeepSeek-R1-Distill-LLaMA-8B | LLaMA | 8.0B | 131,072 | LLaMA-based distillation |
| DeepSeek-R1-Distill-LLaMA-70B | LLaMA | ~70.6B | 131,072 | Largest distilled variant |
| QwQ-32B | Qwen | ~32.5B | 131,072 | Alibaba's reasoning model |

### MoE Thinking Models

These use the full DeepSeek MoE architecture with top-k expert routing and shared experts. Lazy loading is strongly recommended -- eager loading would require >1 TB RAM.

| Model | Active / Total Params | Experts | Context | Notes |
|-------|----------------------|---------|---------|-------|
| DeepSeek-R1 | ~37B / ~671B | 256 routed + 1 shared, top-8 | 128,000 | Full reasoning model with MoE |
| DeepSeek-V3 | ~37B / ~671B | 256 routed + 1 shared, top-8 | 128,000 | Base MoE model (non-thinking) |

### Thinking Model Behavior

When you load a thinking model, torchless-rs **auto-detects** the `<think>` and `</think>` tokens in the vocabulary. No flags are needed to enable thinking -- the model will reason naturally.

By default, the thinking trace is **hidden** and only the final answer is shown. To see the model's reasoning:

```bash
# Show thinking traces from the start
torchless --chat --show-thinking model.gguf

# Or toggle at runtime in chat mode
> /thinking on
Thinking display: on (traces shown dimmed)
> /thinking off
Thinking display: off (traces hidden)
```

When shown, thinking traces appear dimmed in the terminal to visually distinguish them from the final answer.

---

## Large-Context Reference

A reference table of notable open-source models with large context windows. Not all of these are supported by torchless-rs.

### Dense Models (Supported)

| Model | Parameters | Context | License | Notes |
|-------|------------|---------|---------|-------|
| LLaMA 3.1 8B | 8.0B | 131,072 | Meta | GQA (8 KV heads), good context efficiency |
| Qwen 2.5 7B | 7.6B | 131,072 | Apache 2.0 | GQA (4 KV heads), best small-model context ratio |
| Qwen 2.5 7B-1M | 7.6B | 1,000,000 | Apache 2.0 | Extended context variant |
| Qwen 2.5 14B | ~14.2B | 131,072 | Apache 2.0 | Sweet spot for quality + context |
| Qwen 2.5 14B-1M | ~14.2B | 1,000,000 | Apache 2.0 | Extended context variant |
| Qwen 2.5 32B | ~32.5B | 131,072 | Apache 2.0 | Strong reasoning |
| Qwen 2.5 72B | ~72.7B | 131,072 | Apache 2.0 | Frontier-class open model |
| LLaMA 3.1 70B | ~70.6B | 131,072 | Meta | Excellent for long-form tasks |

### MoE Models

| Model | Active / Total Params | Context | License | Supported | Notes |
|-------|----------------------|---------|---------|-----------|-------|
| DeepSeek V3 | ~37B / ~671B | 128,000 | DeepSeek | Yes | 256 experts, top-8 routing |
| DeepSeek R1 | ~37B / ~671B | 128,000 | DeepSeek | Yes | MoE + thinking |
| Mistral Large 3 | 41B / 675B | 256,000 | Apache 2.0 | No | Different MoE architecture |
| LLaMA 4 Scout | 17B / 109B | 10,000,000 | Meta | No | Different MoE architecture |
| LLaMA 4 Maverick | 17B / 400B | 1,000,000 | Meta | No | Different MoE architecture |
| Command R | 35B | 180,000 | Apache 2.0 | No | Different architecture entirely |

> **Note:** MoE support currently covers the DeepSeek architecture (V3/R1). Other MoE architectures (Mistral Large, LLaMA 4) use different routing/expert layouts and are not yet supported.

---

## Understanding Run Modes

Memory usage depends on two independent choices: the **model file you download** and the **runtime options you select**.

### What the Model File Determines

The **quantization level** (FP32, FP16, Q8_0, Q4_K_M, etc.) is baked into the model file you download. A GGUF file quantized to Q4_K_M stores weights in 4-bit format on disk. You cannot change this after download — you need to download a different variant for a different quantization.

### What the App Determines at Runtime

**Eager vs. Lazy loading** — a loading strategy chosen at runtime. Both work on the same model file:

- **Eager** (`Mistral::load()`) reads all weights into RAM as f32 arrays upfront. This gives the fastest CPU inference but uses the most memory.
- **Lazy** (`LazyMistral::load()`) memory-maps the file and reads weights on-demand during each forward pass. This uses far less memory but is slower.

**CPU vs. GPU** — chosen at runtime via `--backend`:

```bash
--backend cpu       # CPU inference (default)
--backend cuda      # NVIDIA GPU
--backend rocm      # AMD GPU
--backend metal     # Apple Silicon GPU
--backend opencl    # Cross-platform GPU
--backend auto      # Auto-select best available
```

### How They Interact

| You download... | You run with... | Result |
|-----------------|-----------------|--------|
| FP32 model | `--backend cpu` (eager) | ~29 GB RAM for a 7B model |
| FP32 model | `--backend cpu` (lazy) | ~2 GB RAM for a 7B model |
| FP16 model | `--backend cuda` | ~14.5 GB VRAM |
| Q4_K_M GGUF | `--backend cuda` | ~4 GB VRAM |
| Q4_K_M GGUF | `--backend cpu` (lazy) | ~2 GB RAM |
| Q4_K_M GGUF | `--backend cpu` (eager) | **~29 GB RAM** (dequantized to f32!) |

> **Important:** Eager mode always dequantizes weights to f32 on load (`params.get_tensor()` returns `Vec<f32>`). This means a Q4_K_M model loaded eagerly uses the **same** amount of RAM as an FP32 model. The on-disk quantization only saves memory in **lazy** and **GPU** modes, where weights are accessed in their native format.

**In short:** download the quantization you want (smaller = less disk space and less VRAM), then pick eager/lazy and CPU/GPU at runtime. The biggest memory savings come from combining a **Q4_K_M model** with **lazy** or **GPU** mode.

---

## How Memory Is Calculated

Torchless-rs estimates memory using these formulas (see `src/kernels/gpu_memory.rs`):

### Model Weights

Per transformer layer:
- Attention: `q_proj + k_proj + v_proj + o_proj` (Q/K/V/O projection matrices)
- MLP: `gate_proj + up_proj + down_proj` (SwiGLU feed-forward)
- Norms: `hidden_size × 2` (input norm + post-attention norm)

Plus embedding table (`vocab_size × hidden_size`) and LM head (`vocab_size × hidden_size`).

**Bytes per parameter by format:**

| Format | Bytes/Param | Description |
|--------|-------------|-------------|
| FP32 | 4.0 | Full precision |
| FP16 / BF16 | 2.0 | Half precision |
| Q8_0 | ~1.06 | 8-bit quantized (32-element blocks) |
| Q4_K_M | ~0.56 | 4-bit quantized (256-element super-blocks) |
| Q4_0 | ~0.56 | 4-bit quantized (32-element blocks) |

### KV Cache

```
kv_cache_bytes = n_layers × n_kv_heads × head_dim × max_seq_len × 2 × 4
```

The `× 2` is for K and V caches; `× 4` is because the cache is always FP32.

### Lazy Mode Savings

In lazy mode (`LazyMistral`, `LazyLLaMA`, etc.), model weights are memory-mapped from the file on disk. Only norm weights (~a few KB per layer) are loaded eagerly. The OS pages in weight data on demand, so actual resident memory is typically **1-1.5 GB** plus the KV cache.

### Reducing Memory Usage

- **`--max-seq-len N`** — Limit the context window. The KV cache is allocated upfront for the full sequence length, so setting this lower saves memory proportionally.
- **Quantized models (GGUF Q4_K_M)** — 4-bit models use ~7× less memory for weights than FP32.
- **Lazy loading** — Use `LazyMistral` / `LazyLLaMA` etc. via the library API, or the engine will auto-select lazy mode when memory is constrained.

---

## Recommended Models

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Minimal hardware** (8 GB RAM) | Gemma 2B (Q4) | Fits in ~2 GB total |
| **Laptop** (16 GB RAM) | Mistral 7B (Q4, lazy, 4K ctx) | ~5 GB total, good quality |
| **Desktop** (32 GB RAM) | LLaMA 3 8B (Q8, lazy) or Mistral 7B (eager) | Best quality at 7-8B scale |
| **GPU** (8 GB VRAM) | Mistral 7B (Q4) | Fits in ~5 GB with 2K context |
| **GPU** (16 GB VRAM) | LLaMA 3 8B (FP16) | Full precision, fast inference |
| **GPU** (24 GB VRAM) | Mistral 7B (FP16, full 32K ctx) | Full context window |
| **Large context** (budget) | Qwen 2.5 7B (Q4, lazy) | 128K context, only 4 KV heads |
| **Large context** (quality) | Qwen 2.5 14B (Q4) | Best quality-to-context ratio |
