#!/bin/bash
#
# Profile-Guided Optimization (PGO) build automation for torchless.
# Builds with instrumentation, runs a representative workload, merges
# profile data, then rebuilds with PGO for an expected 5-15% speedup.
#
# Prerequisites: Rust toolchain, llvm-profdata (LLVM tools)
#
# Usage:
#   ./scripts/build_pgo.sh
#   MODEL_PATH=/path/to/model.bin ./scripts/build_pgo.sh
#   ./scripts/build_pgo.sh --help
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PGO_DIR="${PGO_DIR:-/tmp/torchless-pgo}"
MODEL_PATH="${MODEL_PATH:-}"
PROMPT="${PROMPT:-Hello, how are you today?}"
MAX_TOKENS="${MAX_TOKENS:-100}"
FEATURES="${FEATURES:-parallel,simd}"

# Use $'...' syntax for portable color codes
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'
NC=$'\033[0m'

log_info() {
    printf '%s[PGO]%s %s\n' "$GREEN" "$NC" "$1"
}

log_warn() {
    printf '%s[PGO]%s %s\n' "$YELLOW" "$NC" "$1"
}

log_error() {
    printf '%s[PGO]%s %s\n' "$RED" "$NC" "$1"
}

log_step() {
    printf '%s[PGO]%s %s\n' "$CYAN" "$NC" "$1"
}

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Profile-Guided Optimization build for torchless.

Environment variables:
  PGO_DIR       Directory for profile data (default: /tmp/torchless-pgo)
  MODEL_PATH   Path to model file for profiling (optional; falls back to cargo bench)
  PROMPT       Prompt for inference profiling (default: "Hello, how are you today?")
  MAX_TOKENS   Max tokens for inference run (default: 100)
  FEATURES     Cargo features (default: parallel,simd)

Steps:
  1. Build with profile generation instrumentation
  2. Run profiling workload (inference if MODEL_PATH set, else benchmarks)
  3. Merge profile data with llvm-profdata
  4. Build with PGO optimization

Prerequisites:
  - Rust (cargo, rustc)
  - llvm-profdata (install via: brew install llvm, or apt install llvm)

Options:
  -h, --help   Show this help
EOF
}

check_prerequisites() {
    if ! command -v cargo &>/dev/null; then
        log_error "cargo not found. Install the Rust toolchain."
        exit 1
    fi
    if ! command -v llvm-profdata &>/dev/null; then
        LLVM_PROFDATA=$(find "$(rustc --print sysroot 2>/dev/null)" -name 'llvm-profdata' 2>/dev/null | head -1)
        if [ -z "$LLVM_PROFDATA" ]; then
            log_error "llvm-profdata not found. Install LLVM tools (e.g. brew install llvm, apt install llvm)."
            exit 1
        fi
    else
        LLVM_PROFDATA="llvm-profdata"
    fi
}

main() {
    if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
        show_help
        exit 0
    fi

    cd "$PROJECT_DIR"
    check_prerequisites
    mkdir -p "$PGO_DIR"

    log_step "Step 1: Building with profile generation..."
    RUSTFLAGS="-Cprofile-generate=$PGO_DIR" cargo build --release --features "$FEATURES"

    log_step "Step 2: Running profiling workload..."
    # Instrumented binaries write .profraw to LLVM_PROFILE_FILE; put them in PGO_DIR for merge.
    export LLVM_PROFILE_FILE="$PGO_DIR/default_%p.profraw"
    if [ -n "$MODEL_PATH" ]; then
        if [ ! -f "$MODEL_PATH" ]; then
            log_error "Model file not found: $MODEL_PATH"
            exit 1
        fi
        "$PROJECT_DIR/target/release/torchless" "$MODEL_PATH" "$PROMPT" --max-tokens "$MAX_TOKENS" || true
    else
        cargo bench --release --no-run 2>/dev/null || true
        cargo bench --release || true
    fi
    unset -v LLVM_PROFILE_FILE

    log_step "Step 3: Merging profile data..."
    "$LLVM_PROFDATA" merge -o "$PGO_DIR/merged.profdata" "$PGO_DIR" 2>/dev/null || {
        log_warn "No profile data to merge (workload may not have produced .profraw). Building without PGO."
        cargo build --release --features "$FEATURES"
        log_info "Build complete (without PGO). Set MODEL_PATH for full PGO."
        exit 0
    }

    log_step "Step 4: Building with PGO optimization..."
    RUSTFLAGS="-Cprofile-use=$PGO_DIR/merged.profdata" cargo build --release --features "$FEATURES"

    log_info "PGO build complete! Binary: target/release/torchless"
}

main "$@"