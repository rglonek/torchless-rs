# Torchless-rs

![CI](https://github.com/Arkham/torchless-rs/workflows/CI/badge.svg)

A Rust implementation of [Torchless](https://github.com/ryanssenn/torchless) — a from-scratch LLM inference engine for running Mistral 7B on CPU.

## Features

- **Full Mistral 7B transformer** — GQA, RoPE, SwiGLU, RMSNorm
- **Custom BPE tokenizer** — Metaspace pre-tokenization, UTF-8, byte fallback
- **INT8 quantization** — Per-group symmetric quantization with on-demand dequantization
- **Memory-efficient mode** — <2GB RAM via lazy tensor loading (vs ~25GB eager)
- **BLAS acceleration** — Accelerate (macOS), OpenBLAS (Linux), BLIS (AMD)
- **SIMD kernels** — Portable 8-wide vectorization for core operations
- **Parallel processing** — Multi-threaded matmul and attention heads

## Quick Start

```bash
git clone https://github.com/ryanssenn/torchless.git
cd torchless-rs
cargo build --release
./target/release/torchless mistral.bin "Paris is the capital of"
```

## Building

### Prerequisites

- Rust 1.70+ ([rustup.rs](https://rustup.rs))
- Mistral 7B model in binary format (see [Model Preparation](#model-preparation))

### Basic Build

```bash
cargo build --release
```

### Optimized Build (Recommended)

| Platform | Command |
|----------|---------|
| macOS | `cargo build --release --features "simd,parallel,accelerate"` |
| Linux | `cargo build --release --features "simd,parallel,openblas"` |
| Linux (AMD) | `cargo build --release --features "simd,parallel,blis"` |

### Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `accelerate` | Apple Accelerate BLAS (macOS) | Built-in |
| `openblas` | OpenBLAS backend | `libopenblas-dev` |
| `blis` | BLIS backend (AMD optimized) | BLIS library |
| `simd` | SIMD-optimized kernels | None |
| `parallel` | Rayon multithreading | None |

### Platform Dependencies

<details>
<summary><b>macOS</b> — No setup needed</summary>

Accelerate framework is built-in.

```bash
cargo build --release --features "simd,parallel,accelerate"
```

</details>

<details>
<summary><b>Ubuntu/Debian</b></summary>

```bash
sudo apt install libopenblas-dev
cargo build --release --features "simd,parallel,openblas"
```

</details>

<details>
<summary><b>Fedora/RHEL</b></summary>

```bash
sudo dnf install openblas-devel
cargo build --release --features "simd,parallel,openblas"
```

</details>

<details>
<summary><b>AMD CPUs (BLIS)</b></summary>

Install BLIS from [AMD AOCL](https://developer.amd.com/amd-aocl/) or [build from source](https://github.com/flame/blis).

```bash
cargo build --release --features "simd,parallel,blis"
```

</details>

## Usage

```bash
./target/release/torchless <model_path> "<prompt>"

# Examples
./target/release/torchless mistral.bin "The quick brown fox"
./target/release/torchless --debug mistral.bin "Hello"  # verbose output
```

### Model Preparation

Export Mistral 7B using the [original Torchless](https://github.com/ryanssenn/torchless) export script:

```bash
# Get Mistral 7B
git clone https://huggingface.co/mistralai/Mistral-7B-v0.1
cd Mistral-7B-v0.1 && git lfs install && git lfs pull && cd ..

# Export to binary format
git clone https://github.com/ryanssenn/torchless.git && cd torchless
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 export_mistral.py --model_dir ../Mistral-7B-v0.1 --out mistral.bin --quant f32
```

## Performance Notes

| Mode | Memory | Best For |
|------|--------|----------|
| Eager (`Mistral`) | ~25GB | Fast inference, high-memory systems |
| Lazy (`LazyMistral`) | <2GB | Memory-constrained environments |

> This is a reference implementation prioritizing clarity. For production, consider [llama.cpp](https://github.com/ggerganov/llama.cpp) or [candle](https://github.com/huggingface/candle).

## Documentation

| Document | Contents |
|----------|----------|
| [LIBRARY.md](LIBRARY.md) | Rust library API and examples |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | Technical details (architecture, formats) |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Testing, benchmarking, contributing |
| [FUTURE.md](FUTURE.md) | Roadmap (INT4, GPU, Flash Attention) |

## License

Same license as the original Torchless project.

## Acknowledgments

Based on [Torchless](https://github.com/ryanssenn/torchless) by Ryan Ssenn.
