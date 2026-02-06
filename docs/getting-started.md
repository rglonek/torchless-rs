# Getting Started

Quick start guide for torchless-rs.

## Prerequisites

- Rust 1.70+ ([rustup.rs](https://rustup.rs))
- Mistral 7B model in binary format (see [Model Preparation](#model-preparation))

## Installation

```bash
git clone https://github.com/ryanssenn/torchless.git
cd torchless-rs
cargo build --release
```

## Basic Usage

```bash
./target/release/torchless <model_path> "<prompt>"

# Examples
./target/release/torchless mistral.bin "The quick brown fox"
./target/release/torchless --debug mistral.bin "Hello"  # verbose output
```

## Optimized Builds

| Platform | Command |
|----------|---------|
| macOS | `cargo build --release --features "simd,parallel,accelerate"` |
| Linux | `cargo build --release --features "simd,parallel,openblas"` |
| Linux (AMD) | `cargo build --release --features "simd,parallel,blis"` |
| NVIDIA GPU | `cargo build --release --features "cuda,simd,parallel"` |
| Apple Silicon GPU | `cargo build --release --features "metal-gpu,simd,parallel"` |
| AMD GPU | `cargo build --release --features "rocm,simd,parallel"` |

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `accelerate` | Apple Accelerate BLAS (macOS) | Built-in |
| `openblas` | OpenBLAS backend | `libopenblas-dev` |
| `blis` | BLIS backend (AMD optimized) | BLIS library |
| `simd` | SIMD-optimized kernels | None |
| `parallel` | Rayon multithreading | None |
| `cuda` | NVIDIA GPU support | CUDA Toolkit |
| `rocm` | AMD GPU support | ROCm |
| `metal-gpu` | Apple Silicon GPU | None (macOS only) |
| `opencl` | Cross-platform GPU | OpenCL runtime |

## Platform Setup

<details>
<summary><b>macOS</b> - No setup needed</summary>

Accelerate framework is built-in.

```bash
cargo build --release --features "simd,parallel,accelerate"
```

For GPU acceleration on Apple Silicon:
```bash
cargo build --release --features "metal-gpu,simd,parallel"
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

<details>
<summary><b>NVIDIA GPU (CUDA)</b></summary>

Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (12.0+ recommended).

```bash
cargo build --release --features "cuda,simd,parallel"
```

</details>

<details>
<summary><b>AMD GPU (ROCm)</b></summary>

Install ROCm (Linux only):

```bash
# Ubuntu
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb
sudo amdgpu-install --usecase=rocm

cargo build --release --features "rocm,simd,parallel"
```

</details>

## Model Preparation

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

### Using Pre-Quantized Models (GGUF)

You can also load GGUF models directly from llama.cpp or HuggingFace:

```bash
# Download a GGUF model
wget https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf

# Run with GGUF (auto-detected)
./target/release/torchless mistral-7b-v0.1.Q4_K_M.gguf "Hello"
```

## Memory Modes

| Mode | Memory | Best For |
|------|--------|----------|
| Eager (`Mistral`) | ~25GB | Fast inference, high-memory systems |
| Lazy (`LazyMistral`) | <2GB | Memory-constrained environments |

## Next Steps

- [Library API](library.md) - Use torchless-rs as a Rust library
- [Development Guide](development.md) - Testing and benchmarking
- [GPU Overview](gpu/overview.md) - GPU acceleration setup
- [Quantization](optimization/quantization.md) - Reduce memory usage
