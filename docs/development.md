# Development Guide

Testing, benchmarking, and contributing to torchless-rs.

## Testing

```bash
# Run all tests
cargo test

# Run with all features enabled
cargo test --features "simd,parallel"

# Specific test suites
cargo test tokenizer      # BPE tokenizer
cargo test kernels        # CPU operations
cargo test loader         # Binary format parsing
cargo test integration    # End-to-end tests

# With output
cargo test -- --nocapture
```

### Test Model

A small test model (`tests/fixtures/test_model.bin`, 154KB) is included for testing without the full 27GB Mistral model.

To regenerate:

```bash
cd tests
python3 generate_test_model.py
```

### Memory Tests

Verify allocation behavior and memory usage:

```bash
cargo test --test memory_test --features parallel -- --nocapture
```

Example output:

```
Config size: 120 bytes
InferenceState base size: 1264 bytes
Estimated InferenceState memory: 139984 bytes (0.13 MB)
Lazy model eager portion: 640 bytes (0.62 KB)
Memory savings: 236.20x reduction
```

## Benchmarking

Criterion benchmarks for performance profiling:

```bash
# All benchmarks
cargo bench

# Specific suites
cargo bench --bench matmul_bench      # Matrix multiplication
cargo bench --bench attention_bench   # Attention module
cargo bench --bench e2e_bench         # End-to-end inference

# Filter specific benchmark
cargo bench -- "forward_pass"

# With optimizations
cargo bench --features "simd,parallel"
```

### Benchmark Suites

**matmul_bench** - Matrix multiplication:
- Pure ndarray vs manual implementations
- Pre-allocated vs allocating variants
- Parallel variants (with `--features parallel`)
- Sizes: 128x128 to 32000x4096

**attention_bench** - Attention module:
- Score computation variants
- Softmax implementations
- Multi-head parallel processing
- SIMD kernel comparisons

**e2e_bench** - End-to-end:
- Single token forward pass
- Sequence processing
- Model component timing
- Generation loop

## Code Quality

```bash
# Format check
cargo fmt --all -- --check

# Linting
cargo clippy -- -D warnings

# Full check
cargo check
```

## Memory Profiling

External tools for detailed analysis:

```bash
# Valgrind massif (Linux)
cargo build --release
valgrind --tool=massif ./target/release/torchless model.bin "test"
ms_print massif.out.*

# heaptrack (Linux)
heaptrack ./target/release/torchless model.bin "test"
heaptrack_gui heaptrack.torchless.*

# Instruments (macOS)
xcrun xctrace record --template "Allocations" --launch ./target/release/torchless model.bin "test"
```

## Build Configurations

### Debug Build

```bash
cargo build
RUST_BACKTRACE=1 ./target/debug/torchless model.bin "test"
```

### Release Build

```bash
cargo build --release
```

### Optimized Release

The release profile is configured with Link-Time Optimization (LTO):

```toml
# Cargo.toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
opt-level = 3
```

For additional optimization, use target-specific CPU instructions:

```bash
# Build for current CPU architecture (recommended for best performance)
RUSTFLAGS="-C target-cpu=native" cargo build --release --features "simd,parallel,openblas"

# Build for specific architectures (for distribution)
RUSTFLAGS="-C target-cpu=haswell" cargo build --release     # Intel 4th gen+
RUSTFLAGS="-C target-cpu=skylake" cargo build --release     # Intel 6th gen+
RUSTFLAGS="-C target-cpu=znver3" cargo build --release      # AMD Zen 3+
RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release    # Apple M1+
```

### Profile-Guided Optimization (PGO)

PGO can provide 10-15% additional performance improvements. Use the provided script:

```bash
# Automated PGO build (recommended)
MODEL_PATH=/path/to/model.bin ./scripts/build_pgo.sh

# Manual PGO build
# Step 1: Build with profiling instrumentation
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release

# Step 2: Run representative workload
./target/release/torchless model.bin "test" --max-tokens 100

# Step 3: Merge profile data
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/*.profraw

# Step 4: Build with profile data
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release
```

### GPU Feature Builds

Build with GPU support:

```bash
# NVIDIA CUDA (Linux/Windows)
cargo build --release --features "cuda"

# Apple Metal (macOS)
cargo build --release --features "metal-gpu"

# AMD ROCm (Linux)
cargo build --release --features "rocm"

# OpenCL (cross-platform)
cargo build --release --features "opencl"

# Combine with other features
cargo build --release --features "cuda,simd,parallel"
```

### Build Performance Tips

| Configuration | Build Time | Runtime Performance |
|--------------|------------|---------------------|
| `cargo build` | Fast | Slow (debug) |
| `cargo build --release` | Medium | Good |
| `--release` + LTO | Slow | Better (+5-10%) |
| `--release` + LTO + `target-cpu=native` | Slow | Best (+10-15%) |
| `--release` + LTO + PGO | Very slow | Optimal (+15-25%) |

## Running GPU Tests

GPU tests are marked with `#[ignore]` and only run when hardware is available:

```bash
# CUDA tests
cargo test --features cuda -- --ignored

# Metal tests
cargo test --features metal-gpu -- --ignored

# ROCm tests
cargo test --features rocm -- --ignored

# OpenCL tests
cargo test --features opencl -- --ignored
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `cargo fmt` and `cargo clippy`
5. Submit a pull request

For optimizations:
- Include before/after benchmarks
- Document the approach
- See existing SIMD/parallel implementations as reference
