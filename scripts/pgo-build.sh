#!/bin/bash
#
# Profile-Guided Optimization (PGO) Build Script
#
# This script performs a PGO build of torchless, which can provide
# 10-15% performance improvements for inference workloads.
#
# Prerequisites:
# - Rust nightly toolchain (for PGO support)
# - llvm-profdata (usually installed with LLVM)
# - A model file for profiling (specified via MODEL_PATH)
#
# Usage:
#   MODEL_PATH=/path/to/model.bin ./scripts/pgo-build.sh
#
# The resulting optimized binary will be at target/release/torchless

set -euo pipefail

# Configuration
PROFILE_DIR="${PROFILE_DIR:-/tmp/pgo-data}"
MODEL_PATH="${MODEL_PATH:-}"
PROMPT="${PROMPT:-Hello, world!}"
MAX_TOKENS="${MAX_TOKENS:-50}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check for Rust
    if ! command -v rustc &> /dev/null; then
        log_error "Rust is not installed. Please install Rust first."
        exit 1
    fi

    # Check for llvm-profdata
    if ! command -v llvm-profdata &> /dev/null; then
        log_warn "llvm-profdata not found in PATH."
        log_warn "Trying to find it via rustup..."
        
        # Try to find llvm-profdata in the Rust toolchain
        LLVM_PROFDATA=$(find "$(rustc --print sysroot)" -name 'llvm-profdata' 2>/dev/null | head -1)
        
        if [ -z "$LLVM_PROFDATA" ]; then
            log_error "Could not find llvm-profdata. Please install LLVM tools."
            log_error "On macOS: brew install llvm"
            log_error "On Ubuntu: apt install llvm"
            exit 1
        fi
        log_info "Found llvm-profdata at: $LLVM_PROFDATA"
    else
        LLVM_PROFDATA="llvm-profdata"
    fi

    # Check for model file (optional for basic build)
    if [ -z "$MODEL_PATH" ]; then
        log_warn "MODEL_PATH not set. Will skip profiling step."
        log_warn "For optimal PGO results, set MODEL_PATH to a model file."
        SKIP_PROFILING=true
    elif [ ! -f "$MODEL_PATH" ]; then
        log_error "Model file not found: $MODEL_PATH"
        exit 1
    else
        SKIP_PROFILING=false
    fi
}

# Clean previous profile data
clean_profile_data() {
    log_info "Cleaning previous profile data..."
    rm -rf "$PROFILE_DIR"
    mkdir -p "$PROFILE_DIR"
}

# Step 1: Build instrumented binary
build_instrumented() {
    log_info "Step 1/4: Building instrumented binary..."
    
    RUSTFLAGS="-Cprofile-generate=$PROFILE_DIR" \
        cargo build --release --profile release-instrumented
    
    log_info "Instrumented binary built successfully."
}

# Step 2: Run workload to collect profile data
collect_profile_data() {
    if [ "$SKIP_PROFILING" = true ]; then
        log_warn "Step 2/4: Skipping profile collection (no model specified)."
        return
    fi

    log_info "Step 2/4: Collecting profile data..."
    
    # Run the instrumented binary with a representative workload
    # Adjust these parameters based on your typical use case
    
    log_info "Running inference workload for profiling..."
    ./target/release/torchless "$MODEL_PATH" "$PROMPT" --max-tokens "$MAX_TOKENS" || {
        log_warn "Inference run failed, but continuing with available profile data..."
    }
    
    log_info "Profile data collected."
}

# Step 3: Merge profile data
merge_profile_data() {
    if [ "$SKIP_PROFILING" = true ]; then
        log_warn "Step 3/4: Skipping profile merge (no profile data)."
        return
    fi

    log_info "Step 3/4: Merging profile data..."
    
    # Merge all .profraw files into a single .profdata file
    $LLVM_PROFDATA merge -o "$PROFILE_DIR/merged.profdata" "$PROFILE_DIR"/*.profraw 2>/dev/null || {
        log_warn "No .profraw files found. Profile data may be incomplete."
    }
    
    if [ -f "$PROFILE_DIR/merged.profdata" ]; then
        log_info "Profile data merged successfully."
    fi
}

# Step 4: Build optimized binary using profile data
build_optimized() {
    log_info "Step 4/4: Building PGO-optimized binary..."
    
    if [ "$SKIP_PROFILING" = true ] || [ ! -f "$PROFILE_DIR/merged.profdata" ]; then
        log_warn "No profile data available. Building with LTO only..."
        cargo build --release
    else
        RUSTFLAGS="-Cprofile-use=$PROFILE_DIR/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
            cargo build --release
    fi
    
    log_info "Optimized binary built successfully!"
}

# Main execution
main() {
    echo "============================================"
    echo "  Torchless PGO Build Script"
    echo "============================================"
    echo ""
    
    check_prerequisites
    clean_profile_data
    build_instrumented
    collect_profile_data
    merge_profile_data
    build_optimized
    
    echo ""
    echo "============================================"
    echo "  Build Complete!"
    echo "============================================"
    echo ""
    log_info "Optimized binary: target/release/torchless"
    echo ""
    
    if [ "$SKIP_PROFILING" = true ]; then
        log_warn "Note: This build used LTO only, not full PGO."
        log_warn "For full PGO optimization, run with MODEL_PATH set."
    fi
}

main "$@"
