# Build Optimizations

Build-time optimizations for maximum runtime performance.

## Build Scripts

torchless includes flexible build scripts for creating optimized binaries.

### Main Build Script

The `scripts/build.sh` script provides a unified interface for building with different configurations:

```bash
# Build for current host with defaults (simd, parallel)
./scripts/build.sh

# Build with CUDA support
./scripts/build.sh --cuda

# Build with Metal support (macOS Apple Silicon)
./scripts/build.sh --metal

# Build CPU-only with native CPU optimizations
./scripts/build.sh --cpu-only --native

# Build with all GPU backends available for target
./scripts/build.sh --all-gpu

# Build with PGO optimization
./scripts/build.sh --pgo
```

See `./scripts/build.sh --help` for all options.

### Release Builder

The `scripts/build_releases.sh` script creates ONE binary per platform with all GPU backends compiled in:

```bash
# Build for current host
./scripts/build_releases.sh

# Build for all platforms
./scripts/build_releases.sh --all-platforms

# Build specific platforms
./scripts/build_releases.sh --platforms linux-x86_64,macos-aarch64

# Preview builds without executing
./scripts/build_releases.sh --dry-run
```

Output structure (5 binaries total):
```
releases/
├── torchless-0.1.0-linux-x86_64       # cuda + rocm + opencl + cpu
├── torchless-0.1.0-linux-aarch64      # opencl + cpu
├── torchless-0.1.0-macos-x86_64       # opencl + accelerate + cpu
├── torchless-0.1.0-macos-aarch64      # metal + opencl + accelerate + cpu
└── torchless-0.1.0-windows-x86_64.exe # cuda + opencl + cpu
```

Note: Release builds do not include OpenBLAS/BLIS (requires external libraries). 
macOS builds include Accelerate (bundled with OS). See [BLAS Support](#blas-support-optional) below.

Users select the backend at runtime:
```bash
./torchless --backend auto model.bin "Hello"   # Auto-select best (GPU if available)
./torchless --backend cuda model.bin "Hello"   # Force CUDA
./torchless --backend cpu model.bin "Hello"    # Force CPU only
./torchless --list-backends                    # Show available backends
```

---

## Link-Time Optimization (LTO)

LTO enables cross-crate inlining and whole-program optimization. It is enabled by default in release builds.

**Configuration** (in `Cargo.toml`):
```toml
[profile.release]
lto = "fat"           # Maximum optimization
codegen-units = 1     # Single codegen unit for better optimization
panic = "abort"       # Smaller binary, slightly faster
opt-level = 3         # Maximum optimization level
```

**Impact:** 5-10% speedup, larger compile times

---

## Profile-Guided Optimization (PGO)

PGO uses runtime profiling data to optimize hot paths. The `scripts/build_pgo.sh` script automates the process.

### Quick Start

```bash
# Basic PGO build (uses benchmarks as workload)
./scripts/build_pgo.sh

# PGO build with model inference as workload
MODEL_PATH=/path/to/model.bin ./scripts/build_pgo.sh
```

### How It Works

1. **Instrumented Build**: Compiles with profiling instrumentation
2. **Profile Collection**: Runs workload to collect branch/call statistics
3. **Profile Merge**: Combines profile data with `llvm-profdata`
4. **Optimized Build**: Rebuilds using profile data

### Manual PGO Workflow

```bash
# Step 1: Build with profile generation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release

# Step 2: Run representative workload
export LLVM_PROFILE_FILE="/tmp/pgo-data/default_%p.profraw"
./target/release/torchless model.bin "Hello, world" --max-tokens 100
# Or run benchmarks:
cargo bench --release

# Step 3: Merge profile data
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data

# Step 4: Build with PGO
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release
```

### Prerequisites

- Rust toolchain
- LLVM tools (`llvm-profdata`)
  - macOS: `brew install llvm`
  - Ubuntu: `apt install llvm`
  - Or use the one bundled with rustc

### Tips for Best Results

1. **Use representative workloads**: Profile with typical inference patterns
2. **Include prompt processing**: Short and long prompts exercise different paths
3. **Profile multiple models**: If supporting various sizes
4. **Re-profile periodically**: After significant code changes

**Impact:** 5-15% speedup

---

## Target-Specific Builds

Optimize for specific CPU architectures:

```bash
# Optimize for current CPU
RUSTFLAGS="-C target-cpu=native" cargo build --release

# Or use the build script
./scripts/build.sh --native

# Target specific architectures
RUSTFLAGS="-C target-cpu=skylake" cargo build --release
RUSTFLAGS="-C target-cpu=znver3" cargo build --release  # AMD Zen 3
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release
```

Common target CPUs:
| Target | Description |
|--------|-------------|
| `native` | Current CPU (best for local use) |
| `x86-64-v3` | AVX2 baseline (Haswell+) |
| `x86-64-v4` | AVX-512 baseline |
| `skylake` | Intel Skylake |
| `znver3` | AMD Zen 3 |
| `apple-m1` | Apple M1 |

**Impact:** 5-20% depending on workload and CPU features used

---

## Build Profiles

Available Cargo profiles:

| Profile | Use Case | LTO | Debug Info |
|---------|----------|-----|------------|
| `release` | Production | fat | No |
| `profiling` | Performance analysis | fat | Yes |
| `release-instrumented` | PGO data collection | No | No |

```bash
# Production build
cargo build --release

# Build for profiling with perf/instruments
cargo build --profile profiling

# PGO instrumentation (used by build_pgo.sh)
cargo build --profile release-instrumented
```

---

## Feature Combinations

Recommended feature sets for different use cases:

### CPU-Only (Portable)
```bash
cargo build --release --features "simd,parallel"
```

### Linux with NVIDIA GPU
```bash
cargo build --release --features "simd,parallel,cuda,openblas"
```

### Linux with AMD GPU
```bash
cargo build --release --features "simd,parallel,rocm,openblas"
```

### macOS Apple Silicon
```bash
cargo build --release --features "simd,parallel,metal-gpu,accelerate"
```

### macOS Intel
```bash
cargo build --release --features "simd,parallel,opencl,accelerate"
```

### Cross-Platform GPU
```bash
cargo build --release --features "simd,parallel,opencl"
```

---

## BLAS Support (Optional)

BLAS (Basic Linear Algebra Subprograms) libraries provide highly optimized CPU matrix operations. torchless supports OpenBLAS, BLIS, and Apple Accelerate.

### Why BLAS Isn't in Release Builds

The pre-built release binaries do **not** include OpenBLAS or BLIS because:
- They require external system libraries to be installed
- Cross-compilation with BLAS is complex
- GPU acceleration (CUDA, Metal, ROCm, OpenCL) provides greater speedups for most users

macOS releases **do** include Accelerate, which is bundled with the OS.

### When BLAS Helps

BLAS provides 2-10x faster CPU matrix operations, beneficial when:
- Using `--backend cpu` (no GPU)
- Running on systems without GPU support
- GPU memory is insufficient for the model

### Building with BLAS

If you want BLAS support, build from source:

**Linux with OpenBLAS:**
```bash
# Install OpenBLAS
sudo apt install libopenblas-dev  # Debian/Ubuntu
sudo dnf install openblas-devel   # Fedora/RHEL

# Build with OpenBLAS
cargo build --release --features "simd,parallel,openblas"
```

**Linux with BLIS (good for AMD CPUs):**
```bash
# Install BLIS
sudo apt install libblis-dev

# Build with BLIS
cargo build --release --features "simd,parallel,blis"
```

**macOS (Accelerate is automatic):**
```bash
# Accelerate is included in macOS, no install needed
cargo build --release --features "simd,parallel,accelerate"
```

### Available BLAS Features

| Feature | Library | Best For | Install Required |
|---------|---------|----------|------------------|
| `openblas` | OpenBLAS | General Linux/Windows | Yes (`libopenblas-dev`) |
| `blis` | BLIS | AMD CPUs | Yes (`libblis-dev`) |
| `accelerate` | Apple Accelerate | macOS | No (bundled with OS) |

### Combining with GPU

You can enable both BLAS and GPU features. BLAS will be used when `--backend cpu` is selected:

```bash
cargo build --release --features "simd,parallel,cuda,openblas"
```

---

## Optimization Summary

| Optimization | Speedup | Build Time | Portability |
|--------------|---------|------------|-------------|
| LTO (default) | 5-10% | +50-100% | Portable |
| PGO | 5-15% | +200%+ | Portable |
| target-cpu=native | 5-20% | Same | Host only |
| Combined | 15-40% | +300%+ | Varies |

For maximum performance on a specific machine:
```bash
# All optimizations combined
./scripts/build.sh --pgo --native --features simd,parallel,cuda
```
