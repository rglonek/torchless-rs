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

**matmul_bench** — Matrix multiplication:
- Pure ndarray vs manual implementations
- Pre-allocated vs allocating variants
- Parallel variants (with `--features parallel`)
- Sizes: 128×128 to 32000×4096

**attention_bench** — Attention module:
- Score computation variants
- Softmax implementations
- Multi-head parallel processing
- SIMD kernel comparisons

**e2e_bench** — End-to-end:
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

```toml
# Cargo.toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --features "simd,parallel,openblas"
```

### Profile-Guided Optimization (PGO)

```bash
# Build with profiling
RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release

# Run representative workload
./target/release/torchless model.bin "test" --max-tokens 100

# Build with profile data
llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data
RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata" cargo build --release
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
