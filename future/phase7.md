# Phase 7: Multi-Format Support

**Status:** Implemented  
**Impact:** Access to pre-quantized model ecosystems (GGUF/llama.cpp, HuggingFace/Safetensors)

---

## Overview

Phase 7 adds support for loading models from multiple file formats, enabling compatibility with the broader LLM ecosystem. This allows users to load models directly from:

- **GGUF files** from llama.cpp and related projects
- **Safetensors files** from HuggingFace
- **Torchless Binary** (native format)

Format auto-detection makes loading seamless - just point to a model file and the correct loader is selected automatically.

---

## Components

### 1. GGUF Format Parser

**File:** `src/loader/formats/gguf.rs`

The GGUF (GGML Universal Format) parser provides full compatibility with llama.cpp model files.

#### Features

- **Version Support:** GGUF v1, v2, v3
- **Metadata Extraction:** Architecture, context length, hidden size, attention heads, etc.
- **Memory-Mapped Loading:** Efficient lazy loading via `memmap2`
- **Quantization Dequantization:** All common GGML types supported

#### Supported GGML Types

| Type | Block Size | Bytes/Block | Description |
|------|------------|-------------|-------------|
| F32 | 1 | 4 | 32-bit float |
| F16 | 1 | 2 | 16-bit float |
| BF16 | 1 | 2 | Brain float 16 |
| Q4_0 | 32 | 18 | 4-bit with scale |
| Q4_1 | 32 | 20 | 4-bit with scale+min |
| Q5_0 | 32 | 22 | 5-bit with scale |
| Q5_1 | 32 | 24 | 5-bit with scale+min |
| Q8_0 | 32 | 34 | 8-bit with scale |
| Q8_1 | 32 | 36 | 8-bit with scale+sum |
| Q4_K | 256 | 144 | K-quant 4-bit |
| Q6_K | 256 | 210 | K-quant 6-bit |

#### Usage

```rust
use torchless::{GGUFLoader, GGMLType};

// Load a GGUF model
let loader = GGUFLoader::load("model.gguf")?;

// Print summary
loader.print_summary();

// Access metadata
println!("Architecture: {:?}", loader.metadata.architecture());
println!("Layers: {:?}", loader.metadata.block_count());
println!("Hidden size: {:?}", loader.metadata.embedding_length());

// Get tensor as f32 (auto-dequantizes)
let weights = loader.get_tensor_f32("model.layers.0.self_attn.q_proj.weight")?;

// Get tensor info
if let Some(info) = loader.get_tensor_info("model.embed_tokens.weight") {
    println!("Shape: {:?}", info.shape);
    println!("Type: {}", info.dtype);
    println!("Size: {} bytes", info.size_bytes());
}
```

---

### 2. Safetensors Format Support

**File:** `src/loader/formats/safetensors.rs`

Safetensors is HuggingFace's safe tensor serialization format, widely used for model distribution.

#### Features

- **JSON Header Parsing:** Extracts tensor metadata and model info
- **All Data Types:** F32, F16, BF16, I8, U8, I16, U16, I32, U32, I64, U64, F64, Bool
- **Config Integration:** Load and parse HuggingFace `config.json`
- **Shape Inference:** Automatically infer model dimensions from tensor shapes
- **Sharded Models:** Basic support for multi-file models

#### Usage

```rust
use torchless::{SafetensorsLoader, load_safetensors_with_config, parse_hf_config};

// Load a safetensors model
let loader = SafetensorsLoader::load("model.safetensors")?;

// Print summary
loader.print_summary();

// Access metadata
for (key, value) in &loader.metadata {
    println!("{}: {}", key, value);
}

// Get tensor as f32
let weights = loader.get_tensor_f32("model.embed_tokens.weight")?;

// Load with HuggingFace config.json
let (loader, config) = load_safetensors_with_config("model.safetensors")?;
if let Some(config) = config {
    let unified = parse_hf_config(&config);
    println!("Model type: {:?}", unified.architecture);
    println!("Hidden size: {:?}", unified.hidden_size);
}
```

---

### 3. Format Auto-Detection

**File:** `src/loader/formats/mod.rs`

Automatic format detection and unified loading interface.

#### Detection Methods

| Format | Detection Method |
|--------|-----------------|
| GGUF | Magic bytes `GGUF` (0x46554747) |
| Safetensors | u64 header size + JSON starting with `{` |
| Torchless Binary | u64 header size + JSON with `"metadata"` and `"tensors"` keys |

#### Usage

```rust
use torchless::{detect_format, load_model_auto, ModelFormat};

// Detect format
let format = detect_format("model.bin")?;
match format {
    ModelFormat::GGUF => println!("GGUF format detected"),
    ModelFormat::Safetensors => println!("Safetensors format detected"),
    ModelFormat::TorchlessBinary => println!("Torchless format detected"),
    ModelFormat::Unknown => println!("Unknown format"),
}

// Auto-load any supported format
let model_data = load_model_auto("model.bin")?;

// Access unified interface
println!("Format: {}", model_data.format);
println!("Tensors: {:?}", model_data.tensor_names.len());

// Get config (normalized across formats)
let config = &model_data.config;
println!("Architecture: {:?}", config.architecture);
println!("Hidden size: {:?}", config.hidden_size);
println!("Layers: {:?}", config.n_layers);

// Get tensor data
let weights = model_data.get_tensor("model.embed_tokens.weight")?;
```

---

## Unified Configuration

The `UnifiedConfig` struct normalizes model configuration across all formats:

```rust
pub struct UnifiedConfig {
    pub architecture: Option<String>,       // e.g., "llama", "mistral"
    pub hidden_size: Option<usize>,         // Embedding dimension
    pub intermediate_size: Option<usize>,   // MLP intermediate size
    pub n_layers: Option<usize>,            // Number of transformer layers
    pub n_heads: Option<usize>,             // Attention heads
    pub n_kv_heads: Option<usize>,          // KV heads (for GQA)
    pub vocab_size: Option<usize>,          // Vocabulary size
    pub max_position_embeddings: Option<usize>, // Context length
    pub rope_theta: Option<f32>,            // RoPE base frequency
    pub norm_eps: Option<f32>,              // Layer norm epsilon
    pub quantization: Option<String>,       // Quantization type
    pub metadata: HashMap<String, String>,  // Additional metadata
}
```

---

## API Reference

### Public Types

```rust
// Format detection
pub enum ModelFormat { TorchlessBinary, GGUF, Safetensors, Unknown }
pub fn detect_format<P: AsRef<Path>>(path: P) -> Result<ModelFormat>
pub fn load_model_auto<P: AsRef<Path>>(path: P) -> Result<UnifiedModelData>

// Unified data wrapper
pub struct UnifiedModelData { ... }
pub struct UnifiedConfig { ... }

// GGUF types
pub struct GGUFLoader { ... }
pub struct GGUFMetadata { ... }
pub struct GGUFTensorInfo { ... }
pub enum GGMLType { F32, F16, Q4_0, Q4_K, ... }

// Safetensors types
pub struct SafetensorsLoader { ... }
pub struct SafetensorsTensorInfo { ... }
pub enum SafetensorsDtype { F32, F16, BF16, I8, ... }

// Utilities
pub fn load_safetensors_with_config<P>(path: P) -> Result<(SafetensorsLoader, Option<Value>)>
pub fn parse_hf_config(config: &Value) -> UnifiedConfig
```

---

## File Structure

```
src/loader/
├── mod.rs              # Main loader module (updated exports)
├── parameters.rs       # Torchless binary format
├── quantization.rs     # Quantization blocks
└── formats/
    ├── mod.rs          # Format detection, UnifiedModelData
    ├── gguf.rs         # GGUF parser
    └── safetensors.rs  # Safetensors loader
```

---

## Dependencies Added

```toml
# Cargo.toml
safetensors = "0.4"  # HuggingFace safetensors format
```

---

## Testing

All format-related tests pass:

```
test loader::formats::gguf::tests::test_ggml_type_sizes ... ok
test loader::formats::gguf::tests::test_ggml_type_from_u32 ... ok
test loader::formats::gguf::tests::test_gguf_value_conversions ... ok
test loader::formats::safetensors::tests::test_dtype_from_str ... ok
test loader::formats::safetensors::tests::test_dtype_bytes ... ok
test loader::formats::safetensors::tests::test_parse_hf_config ... ok
test loader::formats::tests::test_format_display ... ok
test loader::formats::tests::test_unified_config_default ... ok
```

---

## Future Enhancements

Potential improvements for future iterations:

1. **More GGML Types:** IQ2_XXS, IQ3_S, and other imatrix quants
2. **Tokenizer Loading:** Extract tokenizer from GGUF metadata
3. **Model Conversion:** Convert between formats
4. **Streaming Loading:** Progressive tensor loading for large models
5. **Validation:** Checksum verification for safetensors
