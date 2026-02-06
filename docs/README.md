# Torchless-rs Documentation

Complete documentation for torchless-rs, a Rust LLM inference engine.

## Quick Navigation

### Getting Started
- [Getting Started Guide](getting-started.md) - Installation, building, and first steps
- [Library API](library.md) - Using torchless-rs as a Rust library
- [Development Guide](development.md) - Testing, benchmarking, and contributing

### Architecture
- [Backend System](architecture/backend-system.md) - Compute backend abstraction (CPU, GPU)
- [Tensor Storage](architecture/tensor-storage.md) - Unified tensor interface and data types
- [Model Architectures](architecture/model-architectures.md) - Supported model families (LLaMA, Phi, Gemma, Qwen)

### Optimization
- [Quantization](optimization/quantization.md) - FP16, INT8, INT4 support
- [SIMD Kernels](optimization/simd.md) - AVX-512, ARM NEON, runtime dispatch
- [Memory Optimization](optimization/memory.md) - Arena allocators, cache alignment, prefetching
- [Algorithms](optimization/algorithms.md) - Flash Attention, speculative decoding, continuous batching
- [Parallelization](optimization/parallelization.md) - Work distribution, pipeline and tensor parallelism

### GPU Backends
- [GPU Overview](gpu/overview.md) - GPU acceleration overview and backend selection
- [CUDA (NVIDIA)](gpu/cuda.md) - NVIDIA GPU support via cuBLAS and custom kernels
- [ROCm (AMD)](gpu/rocm.md) - AMD GPU support via HIP and rocBLAS
- [Metal (Apple)](gpu/metal.md) - Apple Silicon support via Metal compute shaders
- [OpenCL](gpu/opencl.md) - Cross-platform GPU support
- [Memory Management](gpu/memory-management.md) - Unified GPU memory management

### Model Formats
- [Model Formats](formats/model-formats.md) - GGUF, Safetensors, and native format support

### Technical Reference
- [Implementation Details](implementation.md) - Binary format, transformer architecture, memory layout

## Feature Overview

| Feature | Description |
|---------|-------------|
| **Multi-Architecture** | Mistral, LLaMA, Phi, Gemma, Qwen |
| **Quantization** | FP32, FP16, INT8, INT4 (Q4_0, Q4_K_M) |
| **GPU Backends** | CUDA, ROCm, Metal, OpenCL |
| **CPU SIMD** | AVX-512, ARM NEON, runtime dispatch |
| **Memory Efficient** | Lazy loading, arena allocators, Flash Attention |
| **Model Formats** | GGUF, Safetensors, native binary |
| **Parallelization** | Rayon, pipeline parallelism, tensor parallelism |

## Performance Summary

| Configuration | RAM (7B) | Relative Speed | Quality |
|---------------|----------|----------------|---------|
| FP32 CPU | ~28GB | 1x | Best |
| FP16 CPU | ~14GB | 1.2x | Excellent |
| INT8 CPU | ~7GB | 1.5x | Very Good |
| INT4 CPU | ~4GB | 1.8x | Good |
| FP16 GPU | ~14GB VRAM | 20-50x | Excellent |
| INT4 GPU | ~4GB VRAM | 30-80x | Good |
