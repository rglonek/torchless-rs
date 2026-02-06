# Torchless-rs

![CI](https://github.com/Arkham/torchless-rs/workflows/CI/badge.svg)

A Rust implementation of [Torchless](https://github.com/ryanssenn/torchless) — a from-scratch LLM inference engine for running Mistral 7B and other transformer models on CPU and GPU.

## Features

- **Multi-Architecture Support** — Mistral, LLaMA, Phi, Gemma, Qwen
- **GPU Acceleration** — CUDA (NVIDIA), ROCm (AMD), Metal (Apple), OpenCL
- **Quantization** — FP32, FP16, INT8, INT4 (Q4_0, Q4_K_M)
- **Model Formats** — GGUF, Safetensors, native binary
- **Memory Efficient** — Lazy loading (<2GB RAM), Flash Attention
- **CPU Optimizations** — AVX-512, ARM NEON, SIMD kernels, Rayon parallelism

## Quick Start

```bash
git clone https://github.com/ryanssenn/torchless.git
cd torchless-rs
cargo build --release
./target/release/torchless mistral.bin "Paris is the capital of"
```

### Build Scripts

Use the provided build scripts for easy configuration:

```bash
# Simple build with defaults (simd, parallel)
./scripts/build.sh

# Build with all GPU backends (recommended - select at runtime)
./scripts/build.sh --gpu

# Build with PGO for 5-15% speedup
./scripts/build.sh --pgo

# Build release binaries (cpu + gpu flavors)
./scripts/build_releases.sh

# Preview what would be built
./scripts/build_releases.sh --dry-run
```

GPU backends are compiled in but only initialized when selected via `--backend`.

### Manual Build

| Platform | Command |
|----------|---------|
| macOS | `cargo build --release --features "simd,parallel,accelerate"` |
| macOS (GPU) | `cargo build --release --features "metal-gpu,simd,parallel"` |
| Linux | `cargo build --release --features "simd,parallel,openblas"` |
| Linux (NVIDIA) | `cargo build --release --features "cuda,simd,parallel"` |
| Linux (AMD) | `cargo build --release --features "rocm,simd,parallel"` |

## Usage

```bash
./target/release/torchless [OPTIONS] <model_path> "<prompt>"

# Basic usage
./target/release/torchless model.bin "The quick brown fox"

# Select GPU backend at runtime
./target/release/torchless --backend cuda model.bin "Hello"
./target/release/torchless --backend opencl model.bin "Hello"
./target/release/torchless --backend auto model.bin "Hello"  # auto-select best

# List available backends
./target/release/torchless --list-backends

# More options
./target/release/torchless --max-tokens 100 --temperature 0.7 model.bin "Once upon"
```

## Performance

| Mode | Memory (7B) | Speed |
|------|-------------|-------|
| Eager CPU | ~25GB | ~1 tok/s |
| Lazy CPU | <2GB | ~0.5 tok/s |
| GPU (FP16) | ~14GB VRAM | 20-50 tok/s |
| GPU (INT4) | ~4GB VRAM | 30-80 tok/s |

## Documentation

| Topic | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation and first steps |
| [Library API](docs/library.md) | Using as a Rust library |
| [Development](docs/development.md) | Testing and contributing |

### Architecture
| Topic | Description |
|-------|-------------|
| [Backend System](docs/architecture/backend-system.md) | CPU/GPU compute abstraction |
| [Tensor Storage](docs/architecture/tensor-storage.md) | Multi-dtype tensor support |
| [Model Architectures](docs/architecture/model-architectures.md) | LLaMA, Phi, Gemma, Qwen |

### Optimization
| Topic | Description |
|-------|-------------|
| [Build Optimization](docs/optimization/build.md) | LTO, PGO, build scripts |
| [Quantization](docs/optimization/quantization.md) | FP16, INT8, INT4 formats |
| [SIMD Kernels](docs/optimization/simd.md) | AVX-512, ARM NEON |
| [Memory](docs/optimization/memory.md) | Allocators, prefetching |
| [Algorithms](docs/optimization/algorithms.md) | Flash Attention, speculative decoding |
| [Parallelization](docs/optimization/parallelization.md) | Pipeline and tensor parallelism |

### GPU
| Topic | Description |
|-------|-------------|
| [GPU Overview](docs/gpu/overview.md) | Backend selection |
| [CUDA](docs/gpu/cuda.md) | NVIDIA support |
| [ROCm](docs/gpu/rocm.md) | AMD support |
| [Metal](docs/gpu/metal.md) | Apple Silicon support |
| [OpenCL](docs/gpu/opencl.md) | Cross-platform support |

### Reference
| Topic | Description |
|-------|-------------|
| [Model Formats](docs/formats/model-formats.md) | GGUF, Safetensors loading |
| [Implementation](docs/implementation.md) | Technical internals |

## License

Same license as the original Torchless project.

## Acknowledgments

Based on [Torchless](https://github.com/ryanssenn/torchless) by Ryan Ssenn.
