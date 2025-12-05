# Torchless-rs

![CI](https://github.com/Arkham/torchless-rs/workflows/CI/badge.svg)

A Rust port of [Torchless](https://github.com/ryanssenn/torchless) - a custom-built LLM inference engine for running Mistral 7B on CPU.

This is a from-scratch implementation that replicates the full transformer architecture including:

- Custom BPE tokenizer with Metaspace pre-tokenization
- Grouped-Query Attention (GQA)
- Rotary Positional Embeddings (RoPE)
- SwiGLU feedforward blocks
- Per-group symmetric quantization (int8)

## Features

- **Full feature parity** with the C++ version (architecturally)
- **Binary format compatibility** - reads the same `.bin` files exported by the Python script
- **Test-driven development** - 14 unit tests covering all core components
- **Clean Rust implementation** using ndarray for tensor operations

## Architecture

```
torchless-rs/
├── src/
│   ├── loader/         # Binary format reader & config parsing
│   ├── tokenizer/      # BPE tokenizer with byte fallback
│   ├── kernels/        # CPU ops (matmul, softmax, RoPE, RMSNorm, SiLU)
│   ├── model/          # Transformer modules
│   │   └── modules/    # Embedding, Attention, MLP, Layer
│   ├── sampler/        # Greedy & multinomial sampling
│   ├── lib.rs          # Library exports
│   └── main.rs         # CLI binary
└── tests/
    └── fixtures/       # Integration test fixtures (TBD)
```

## Setup

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- Mistral 7B model exported to binary format

### Installation

```bash
git clone https://github.com/ryanssenn/torchless.git
cd torchless-rs
cargo build --release
```

## Usage

### Prepare the Model

You'll need the Mistral 7B model exported to the custom binary format. Use the Python export script from the [original Torchless project](https://github.com/ryanssenn/torchless):

```bash
# Download Mistral 7B
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1 && cd Mistral-7B-v0.1 && git lfs install && git lfs pull && cd ..

# Clone original project
git clone https://github.com/ryanssenn/torchless.git && cd torchless

# Export model to binary
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 export_mistral.py \
  --model_dir ../Mistral-7B-v0.1 \
  --out ./mistral.bin \
  --quant f32
```

### Run Inference

```bash
# Basic usage
./target/release/torchless <model_path> "<prompt>"

# Example
./target/release/torchless ../torchless/mistral.bin "Paris is the capital of"

# With debug logging (shows layer-by-layer progress)
./target/release/torchless --debug ../torchless/mistral.bin "Hello"
```

**Example Output (with `--debug`):**

```
Loading model from: ../torchless/mistral.bin
Loading embedding table...
Loading final norm...
Loading LM head...
Loading 32 layers...
  Loading layer 0/32...
  Loading layer 4/32...
  ...
Initializing inference state...
Tokenizing prompt: Paris
Tokens: [1, 5465]
Processing 1 prompt tokens...

Prompt token 1/1
  Layer 0/32
  Layer 8/32
  ...

Generating tokens:
Paris [generated text...]
```

**Note**: Due to current performance limitations, model loading takes several minutes and inference is slow. See Limitations section below.

## Testing

```bash
# Run all tests (unit + integration)
cargo test

# Run specific test suites
cargo test tokenizer  # BPE tokenizer tests
cargo test kernels    # CPU kernel tests
cargo test loader     # Binary format tests
cargo test integration_test  # End-to-end tests

# Run with output
cargo test -- --nocapture
```

**Test Model:**
A tiny 154KB test model (`tests/fixtures/test_model.bin`) is included for fast testing without needing the 27GB Mistral model. Generate it with:

```bash
# Requires Python with numpy
cd /Users/arkham/code/torchless && source venv/bin/activate
python3 /path/to/torchless-rs/tests/generate_test_model.py
```

## Current Limitations

### Performance

This implementation prioritizes **correctness and clarity** over performance. It successfully demonstrates a complete Mistral 7B implementation but has significant performance limitations:

**Memory Usage (~25GB)**

- Current implementation eagerly loads all 7B parameters into RAM
- Each tensor is dequantized and copied into a separate `Vec<f32>`
- **Optimization needed**: Implement lazy loading with memory-mapped tensor views

**Inference Speed (Minutes per token)**

- No SIMD optimizations
- Generic ndarray operations instead of optimized BLAS
- Lots of array allocations in attention module
- No multithreading
- **Optimization needed**: Use optimized linear algebra library, reduce allocations, add SIMD

### Recommended for

- ✅ **Learning** - Understanding transformer architecture from scratch
- ✅ **Reference** - Clean, readable Rust implementation of Mistral 7B
- ✅ **Testing** - Verifying correctness with comprehensive unit tests
- ✅ **Prototyping** - Experimenting with architecture modifications

### Not Recommended for

- ❌ **Production inference** - Use optimized libraries (llama.cpp, candle, etc.)
- ❌ **Performance benchmarking** - Needs SIMD and optimized matmul
- ❌ **Low-memory systems** - Currently requires 25GB+ RAM

## Implementation Details

### Binary Format

The model binary uses the same format as the C++ version:

- 8-byte header size (little-endian u64)
- JSON header with metadata, tensor info, vocab, and merges
- 64-byte aligned tensor payload
- Per-group int8 quantization with f32 scales (group_size=64)

### Tokenizer

Full BPE implementation with:

- Metaspace pre-tokenization (replace spaces with ▁)
- UTF-8 character handling
- Byte fallback for unknown characters (`<0xXX>`)
- BOS token insertion

### Transformer

- **32 decoder layers**
- **Grouped-Query Attention**: 32 query heads, 8 KV heads (4x reuse)
- **RoPE**: Half-split layout (rotate dim i with dim i+head_dim/2)
- **SwiGLU MLP**: gate_proj + up_proj with SiLU, then down_proj
- **RMSNorm**: Layer normalization
- **KV Cache**: 4D tensor [layers, kv_heads, seq_len, head_dim]

## Future Improvements

1. **Memory Optimization**

   - Implement lazy tensor loading with memory-mapped views
   - Reduce eager allocation during model load
   - Target: <2GB RAM usage

2. **Performance Optimization**

   - Replace ndarray matmul with optimized BLAS (BLIS, OpenBLAS)
   - Add CPU SIMD instructions (AVX2/AVX-512)
   - Reduce allocations in attention module
   - Add multithreading for layer-parallel matmul
   - Target: Match C++ version performance

3. **Testing**
   - Generate integration test fixtures from HuggingFace
   - Add module-level unit tests
   - Property-based testing with proptest

## Development

```bash
# Check code compiles
cargo check

# Run tests
cargo test

# Check formatting
cargo fmt --all -- --check

# Run linter
cargo clippy -- -D warnings

# Build optimized binary
cargo build --release

# Run with backtrace on panic
RUST_BACKTRACE=1 ./target/release/torchless <args>
```

## License

This project follows the same license as the original Torchless project.

## Acknowledgments

This Rust port is based on the [original Torchless C++ implementation](https://github.com/ryanssenn/torchless) by Ryan Ssenn.
