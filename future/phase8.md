# Phase 8: Multi-Architecture Support

This document describes the implementation of Phase 8 from the optimization plan - support for multiple model architectures beyond Mistral.

## Overview

Phase 8 adds support for loading and running inference on models from different architecture families:

- **LLaMA** (Meta) - LLaMA 1, LLaMA 2, LLaMA 3
- **Phi** (Microsoft) - Phi-2, Phi-3
- **Gemma** (Google) - Gemma 1, Gemma 2
- **Qwen** (Alibaba) - Qwen 1, Qwen 2

## Architecture Detection

The system automatically detects model architecture from:

1. **Config metadata** - If the model specifies `architecture` field
2. **Tensor naming patterns** - Each architecture has distinctive tensor names

```rust
use torchless::{detect_architecture, ModelArchitecture};

// Automatic detection
let arch = detect_architecture(&config, &tensor_names);

// Or detect from tensor names only
let arch = detect_architecture_from_tensors(&tensor_names);
```

### Detection Patterns

| Architecture | Distinctive Pattern |
|-------------|---------------------|
| Mistral | `model.layers.*.self_attn.*` (default) |
| LLaMA | Same as Mistral (detected via config) |
| Phi | `model.final_layernorm.*` |
| Gemma | `*.post_feedforward_layernorm.*` |
| Qwen | `transformer.h.*.attn.c_attn.*` |

## New Types

### ModelArchitecture Enum

```rust
pub enum ModelArchitecture {
    Mistral,
    LLaMA,
    Phi,
    Gemma,
    Qwen,
    Unknown,
}
```

### Model Trait

A unified interface for all model architectures:

```rust
pub trait Model: Send + Sync {
    fn architecture(&self) -> ModelArchitecture;
    fn config(&self) -> &Config;
    fn forward(&self, state: &mut InferenceState, token: u32, debug: bool);
    fn fast_forward(&self, state: &mut InferenceState, token: u32, debug: bool);
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
}
```

### DynamicModel Enum

Runtime polymorphism for model loading:

```rust
pub enum DynamicModel<'a> {
    Mistral(Mistral),
    LLaMA(LLaMA),
    Phi(Phi),
    Gemma(Gemma),
    Qwen(Qwen),
    // Lazy variants...
}
```

## Architecture Configurations

### ArchitectureConfig

Stores architecture-specific settings:

```rust
pub struct ArchitectureConfig {
    pub architecture: ModelArchitecture,
    pub tensor_names: TensorNamePattern,
    pub tie_embeddings: bool,      // Gemma uses tied embeddings
    pub fused_qkv: bool,           // Qwen fuses Q/K/V
    pub fused_gate_up: bool,       // Phi fuses gate+up projection
    pub rope_scaling: RopeScaling, // LLaMA 3 uses dynamic NTK
    pub activation: ActivationType,
    pub norm_type: NormType,
    pub parallel_residual: bool,   // Phi uses parallel residuals
}
```

### Architecture Comparison

| Feature | Mistral | LLaMA | Phi | Gemma | Qwen |
|---------|---------|-------|-----|-------|------|
| **Activation** | SwiGLU | SwiGLU | GELUTanh | GeGLU | SwiGLU |
| **Normalization** | RMSNorm | RMSNorm | LayerNorm | RMSNorm(+1) | RMSNorm |
| **Parallel Residuals** | No | No | Yes | No | No |
| **Fused QKV** | No | No | No | No | Yes |
| **Tied Embeddings** | No | No | No | Yes | No |
| **RoPE Scaling** | None | Optional | None | None | None |

## Usage Examples

### Automatic Loading

```rust
use torchless::{Parameters, DynamicModel, InferenceState};

// Load with automatic architecture detection
let params = Parameters::load("model.bin")?;
let model = DynamicModel::load_auto(params)?;

println!("Detected architecture: {}", model.architecture());

// Create inference state
let mut state = InferenceState::new(model.config().clone());

// Run inference
model.forward(&mut state, token, false);
```

### Specific Architecture

```rust
use torchless::{Parameters, LLaMA, InferenceState};

// Load specifically as LLaMA
let params = Parameters::load("llama-7b.bin")?;
let model = LLaMA::load(params)?;

let mut state = InferenceState::new(model.config.clone());
model.fast_forward(&mut state, token, false);
```

### Lazy Loading (Memory Efficient)

```rust
use torchless::{Parameters, LazyGemma, InferenceState};

// Lazy loading keeps weights memory-mapped
let params = Parameters::load("gemma-2b.bin")?;
let model = LazyGemma::load(&params)?;

let mut state = InferenceState::new(model.config.clone());
model.forward(&mut state, token, false);
```

### Using ModelLoader

```rust
use torchless::{ModelLoader, ModelArchitecture};

// Auto-detect and load
let model = ModelLoader::load_auto("model.bin")?;

// Or force specific architecture
let model = ModelLoader::load_as("model.bin", ModelArchitecture::Phi)?;
```

## Architecture-Specific Features

### LLaMA

- Supports LLaMA 1, 2, and 3 models
- Automatic version detection based on:
  - `rope_theta > 100000` → LLaMA 3
  - `vocab_size > 100000` → LLaMA 3
- RoPE scaling options for extended context:
  - `RopeScaling::Linear` - Simple position scaling
  - `RopeScaling::DynamicNTK` - NTK-aware scaling (LLaMA 3)
  - `RopeScaling::YaRN` - Yet another RoPE extensioN

### Phi

- Parallel residual connections:
  ```
  x = x + attention(norm(x)) + mlp(norm(x))
  ```
- Fused gate+up projection in MLP
- LayerNorm instead of RMSNorm
- GELU-tanh activation

### Gemma

- Tied embeddings: LM head uses embedding weights
- GeGLU activation (GELU-gated GLU)
- RMSNorm with +1 offset: `x * (1 + weight) / rms`
- Different naming: `post_feedforward_layernorm`

### Qwen

- Fused Q/K/V projection (single `c_attn` tensor)
- Different tensor naming:
  - `transformer.h.*` instead of `model.layers.*`
  - `transformer.wte` instead of `model.embed_tokens`
  - `transformer.ln_f` instead of `model.norm`

## File Structure

```
src/model/
├── architecture.rs      # Architecture detection & config
├── models/
│   ├── mod.rs          # DynamicModel, ModelLoader
│   ├── llama.rs        # LLaMA, LazyLLaMA
│   ├── phi.rs          # Phi, LazyPhi, LayerNorm, PhiMLP
│   ├── gemma.rs        # Gemma, LazyGemma, GemmaRMSNorm, GemmaMLP
│   └── qwen.rs         # Qwen, LazyQwen, QwenAttention
└── mod.rs              # Updated exports
```

## Exports

All new types are exported from the library root:

```rust
// Architecture detection
pub use model::{
    ModelArchitecture, Model, TensorNamePattern, ArchitectureConfig,
    RopeScaling, ActivationType, NormType,
    detect_architecture, detect_architecture_from_tensors,
};

// Model implementations
pub use model::{
    LLaMA, LazyLLaMA,
    Phi, LazyPhi,
    Gemma, LazyGemma,
    Qwen, LazyQwen,
    DynamicModel, ModelLoader,
};
```

## Testing

All architecture implementations include unit tests:

```rust
#[test]
fn test_architecture_from_str() {
    assert_eq!(ModelArchitecture::from_str("llama"), ModelArchitecture::LLaMA);
    assert_eq!(ModelArchitecture::from_str("phi-3"), ModelArchitecture::Phi);
}

#[test]
fn test_architecture_detection() {
    let gemma_tensors = vec![
        "model.layers.0.post_feedforward_layernorm.weight".to_string(),
    ];
    assert_eq!(
        detect_architecture_from_tensors(&gemma_tensors),
        ModelArchitecture::Gemma
    );
}

#[test]
fn test_phi_arch_config() {
    let arch = ArchitectureConfig::phi();
    assert!(arch.parallel_residual);
    assert!(arch.fused_gate_up);
    assert_eq!(arch.activation, ActivationType::GELUTanh);
}
```

## Performance Notes

- All implementations support both eager and lazy loading
- Lazy variants use memory-mapped weights for reduced RAM usage
- `fast_forward()` uses SIMD/parallel kernels when available
- Architecture-specific optimizations:
  - Phi: Fused gate+up reduces memory bandwidth
  - Qwen: Fused QKV reduces projection overhead
  - Gemma: Tied embeddings save memory

## Future Work

- Add more architectures (Falcon, MPT, etc.)
- Implement architecture-specific attention optimizations
- Add quantization support for all architectures
- Improve lazy loading with architecture-specific tensor access patterns
