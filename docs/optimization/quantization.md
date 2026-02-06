# Quantization

Comprehensive quantization support for reduced memory usage while maintaining inference quality.

## Overview

| Format | Bytes/Element | Compression vs FP32 | Quality Impact |
|--------|---------------|---------------------|----------------|
| F32    | 4.0           | 1x (baseline)       | Best           |
| F16    | 2.0           | 2x                  | Excellent      |
| BF16   | 2.0           | 2x                  | Excellent      |
| Q8_0   | ~1.06         | ~3.8x               | Very Good      |
| Q4_0   | ~0.56         | ~7.1x               | Good           |
| Q4_K_M | ~0.56         | ~7.1x               | Good+          |
| Q4_K_S | ~0.54         | ~7.4x               | Good           |

## Quantization Formats

### Q4_0: Basic 4-bit Quantization

The simplest 4-bit format with one scale per 32 elements.

**Block Structure (18 bytes for 32 weights):**
```
┌────────┬────────────────────────────────────────┐
│ scale  │ 16 bytes of packed 4-bit values        │
│ (f16)  │ (2 values per byte, lower nibble first)│
└────────┴────────────────────────────────────────┘
```

**Dequantization:**
```rust
weight[i] = (nibble[i] - 8) * scale
```

### Q8_0: 8-bit Quantization

Higher precision quantization with one scale per 32 elements.

**Block Structure (34 bytes for 32 weights):**
```
┌────────┬────────────────────────────────────────┐
│ scale  │ 32 bytes of int8 values                │
│ (f16)  │                                        │
└────────┴────────────────────────────────────────┘
```

**Dequantization:**
```rust
weight[i] = qs[i] * scale
```

### Q4_K_M: K-Quantization (Medium)

Advanced 4-bit format with per-block scales and minimums for better accuracy.

**Super-block Structure (144 bytes for 256 weights):**
```
┌─────────┬────────────┬───────────────────────────┬───────────────┐
│ d (f16) │ dmin (f16) │ scales[8] + mins[8]       │ 128 bytes qs  │
│ 2 bytes │ 2 bytes    │ 8 bytes packed            │ packed nibbles│
└─────────┴────────────┴───────────────────────────┴───────────────┘
```

**Dequantization:**
```rust
weight[i] = nibble * (d * scale_bits) - (dmin * min_bits)
```

### Q4_K_S: K-Quantization (Small)

Compact variant of Q4_K_M with reduced scale precision.

## API Usage

### Reading Quantized Tensors

```rust
use torchless::{Parameters, TensorDtype};

let params = Parameters::load("model.bin")?;

// Get a tensor view (lazy, zero-copy)
let weight = params.get_tensor_view("layer.0.weight")?;

// Check the dtype
match weight.dtype {
    TensorDtype::F16 => println!("Half precision"),
    TensorDtype::Q4_0 => println!("4-bit quantized"),
    TensorDtype::Q4_K_M => println!("4-bit K-quant"),
    _ => {}
}

// Dequantize a row on-the-fly
let row = weight.get_row(0);

// Fused dequantize + matmul (most efficient)
let output = weight.matmul_vec(&input);
```

### Creating Quantized Tensors

```rust
use torchless::{QuantizedTensor, QuantFormat};

let data: Vec<f32> = vec![...]; // Your f32 data

// Quantize to different formats
let q4_0 = QuantizedTensor::from_f32(&data, vec![1024, 4096], QuantFormat::Q4_0);
let q4_k_m = QuantizedTensor::from_f32(&data, vec![1024, 4096], QuantFormat::Q4_K_M);
let f16 = QuantizedTensor::from_f32(&data, vec![1024, 4096], QuantFormat::F16);

// Check compression ratio
println!("Q4_0 compression: {}x", q4_0.compression_ratio());
// Output: Q4_0 compression: 7.1x

// Dequantize back to f32
let recovered = q4_0.to_f32();
```

### Mixed Precision Configuration

```rust
use torchless::{MixedPrecisionConfig, ModelSizeParams, Dtype};

// Use predefined configurations
let config = MixedPrecisionConfig::balanced(); // INT8 weights, FP16 activations
let config = MixedPrecisionConfig::aggressive(); // INT4 weights, FP16 activations

// Or customize
let config = MixedPrecisionConfig {
    weights_dtype: Dtype::Int4,
    embedding_dtype: Dtype::F16,
    activation_dtype: Dtype::F16,
    kv_cache_dtype: Dtype::F16,
    attention_dtype: Dtype::F32,
    output_dtype: Dtype::F32,
};

// Estimate memory usage
let params = ModelSizeParams::mistral_7b(4096); // 4096 max sequence length
let memory_mb = config.estimate_memory_mb(params);
println!("Estimated memory: {} MB", memory_mb);
```

## Fused Dequantize + MatMul

The `TensorView::matmul_vec()` method fuses dequantization with matrix-vector multiplication, avoiding intermediate allocations:

```rust
// Instead of:
let weights_f32 = tensor_view.get_row(row); // Allocates
let dot = weights_f32.iter().zip(x).map(|(w, x)| w * x).sum();

// Use:
let output = tensor_view.matmul_vec(&x); // No intermediate allocation
```

This is critical for performance as:
1. Memory bandwidth is the bottleneck for large models
2. Quantized data is smaller, so less memory to read
3. Dequantization happens in CPU registers during the dot product

## Memory Layout

All quantization formats use row-major layout with block alignment:

```
Row 0: [Block 0][Block 1][Block 2]...
Row 1: [Block 0][Block 1][Block 2]...
...
```

This enables efficient sequential reads and cache utilization.

## Memory Savings Summary

| Configuration | 7B Model RAM |
|---------------|--------------|
| FP32 | ~28 GB |
| FP16 | ~14 GB |
| INT8 (Q8_0) | ~8 GB |
| INT4 (Q4_0) | ~4 GB |
| INT4 (Q4_K_M) | ~4 GB |

## GGUF Compatibility

The Q4_0 and Q8_0 formats are designed to be compatible with GGUF quantization. GGUF files can be loaded directly with automatic dequantization.

See [Model Formats](../formats/model-formats.md) for GGUF loading details.
