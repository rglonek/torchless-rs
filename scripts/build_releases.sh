#!/bin/bash
#
# Build release binaries of torchless for distribution.
# Creates ONE binary per platform with all available GPU backends compiled in.
# Users select the backend at runtime with --backend flag.
#
# Usage:
#   ./scripts/build_releases.sh [OPTIONS]
#   ./scripts/build_releases.sh --platforms linux-x86_64,macos-aarch64
#
# See --help for all options.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_SCRIPT="$SCRIPT_DIR/build.sh"

# =============================================================================
# Colors and Logging
# =============================================================================

# Use printf for portable color output
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
NC=$'\033[0m'

log_info()  { printf '%s[RELEASE]%s %s\n' "$GREEN" "$NC" "$1"; }
log_warn()  { printf '%s[RELEASE]%s %s\n' "$YELLOW" "$NC" "$1"; }
log_error() { printf '%s[RELEASE]%s %s\n' "$RED" "$NC" "$1"; }
log_step()  { printf '%s[RELEASE]%s %s%s%s\n' "$CYAN" "$NC" "$BOLD" "$1" "$NC"; }

# =============================================================================
# Default Configuration
# =============================================================================

detect_host() {
    local os arch
    case "$(uname -s)" in
        Linux*)  os="linux" ;;
        Darwin*) os="macos" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *) os="unknown" ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64) arch="x86_64" ;;
        arm64|aarch64) arch="aarch64" ;;
        *) arch="unknown" ;;
    esac
    echo "${os}-${arch}"
}

HOST_TARGET=$(detect_host)
HOST_OS="${HOST_TARGET%-*}"

PLATFORMS="${PLATFORMS:-$HOST_TARGET}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/releases}"
VERSION="${VERSION:-$(grep '^version' "$PROJECT_DIR/Cargo.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')}"
DRY_RUN="${DRY_RUN:-false}"
CLEAN="${CLEAN:-false}"

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat <<EOF
${BOLD}torchless Release Builder${NC}

Builds ONE release binary per platform with all GPU backends compiled in.
Users select the backend at runtime with --backend flag.

Usage: $0 [OPTIONS]

${BOLD}Platform Options:${NC}
  --platforms PLATFORMS   Comma-separated build targets (default: host only)
                          Available: linux-x86_64, linux-aarch64,
                                     macos-x86_64, macos-aarch64,
                                     windows-x86_64
  --all-platforms         Build for all platforms (requires cross-compilation)
  --host-only             Build only for current host (default)

${BOLD}Output Options:${NC}
  --output-dir DIR        Output directory (default: releases/)
  --version VERSION       Version string for naming (default: from Cargo.toml)
  --clean                 Clean output directory before building

${BOLD}Other:${NC}
  --dry-run               Show what would be built without building
  -h, --help              Show this help

${BOLD}Backends per Platform:${NC}
  linux-x86_64:   cuda + rocm + opencl + cpu
  linux-aarch64:  opencl + cpu
  macos-x86_64:   opencl + accelerate + cpu
  macos-aarch64:  metal + opencl + accelerate + cpu
  windows-x86_64: cuda + opencl + cpu

${BOLD}Note on BLAS:${NC}
  Release builds do NOT include OpenBLAS/BLIS (external dependency).
  macOS builds include Accelerate (bundled with OS).
  For faster CPU inference, build from source with --features openblas.
  See docs/optimization/build.md for details.

${BOLD}Runtime Backend Selection:${NC}
  Each binary includes ALL backends for its platform. Select at runtime:

    ./torchless --backend auto model.bin "prompt"     # Auto-select best GPU, fallback to CPU
    ./torchless --backend cuda model.bin "prompt"     # Force CUDA
    ./torchless --backend cpu model.bin "prompt"      # Force CPU (no GPU)
    ./torchless --list-backends                       # Show what's available

${BOLD}Examples:${NC}
  # Build for current host
  $0

  # Build for all platforms
  $0 --all-platforms

  # Preview builds
  $0 --all-platforms --dry-run

${BOLD}Output Structure:${NC}
  releases/
  ├── torchless-${VERSION}-linux-x86_64       # cuda + rocm + opencl + cpu
  ├── torchless-${VERSION}-linux-aarch64      # opencl + cpu
  ├── torchless-${VERSION}-macos-x86_64       # opencl + accelerate + cpu
  ├── torchless-${VERSION}-macos-aarch64      # metal + opencl + accelerate + cpu
  └── torchless-${VERSION}-windows-x86_64.exe # cuda + opencl + cpu
EOF
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --platforms)
                PLATFORMS="$2"; shift 2 ;;
            --all-platforms)
                PLATFORMS="linux-x86_64,linux-aarch64,macos-x86_64,macos-aarch64,windows-x86_64"
                shift ;;
            --host-only)
                PLATFORMS="$HOST_TARGET"; shift ;;
            --output-dir)
                OUTPUT_DIR="$2"; shift 2 ;;
            --version)
                VERSION="$2"; shift 2 ;;
            --clean)
                CLEAN=true; shift ;;
            --dry-run)
                DRY_RUN=true; shift ;;
            -h|--help)
                show_help; exit 0 ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information."
                exit 1 ;;
        esac
    done
}

# =============================================================================
# Configuration Functions
# =============================================================================

# Get ALL features for a given platform (GPU + CPU optimizations)
# Note: OpenBLAS/BLIS are NOT included because they require external system libraries.
# macOS uses Accelerate which is bundled with the OS.
# Users who want BLAS can build from source: cargo build --features openblas
get_platform_features() {
    local platform="$1"
    local base="simd,parallel"
    
    case "$platform" in
        linux-x86_64)
            echo "${base},cuda,rocm,opencl" ;;
        linux-aarch64)
            echo "${base},opencl" ;;
        macos-x86_64)
            echo "${base},opencl,accelerate" ;;
        macos-aarch64)
            echo "${base},metal-gpu,opencl,accelerate" ;;
        windows-x86_64)
            echo "${base},cuda,opencl" ;;
        *)
            echo "${base}" ;;
    esac
}

# Get description of backends for a platform
get_platform_backends() {
    local platform="$1"
    
    case "$platform" in
        linux-x86_64)
            echo "cuda + rocm + opencl + cpu" ;;
        linux-aarch64)
            echo "opencl + cpu" ;;
        macos-x86_64)
            echo "opencl + accelerate + cpu" ;;
        macos-aarch64)
            echo "metal + opencl + accelerate + cpu" ;;
        windows-x86_64)
            echo "cuda + opencl + cpu" ;;
        *)
            echo "cpu" ;;
    esac
}

# Generate output filename
get_output_name() {
    local platform="$1"
    local ext=""
    
    [[ "$platform" == windows-* ]] && ext=".exe"
    
    echo "torchless-${VERSION}-${platform}${ext}"
}

# =============================================================================
# Build Functions
# =============================================================================

build_platform() {
    local platform="$1"
    local features output_name backends
    
    features=$(get_platform_features "$platform")
    output_name=$(get_output_name "$platform")
    backends=$(get_platform_backends "$platform")
    
    log_step "Building: $output_name"
    log_info "  Platform:  $platform"
    log_info "  Backends:  $backends"
    log_info "  Features:  $features"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "  [DRY RUN - skipping actual build]"
        return 0
    fi
    
    # Check if we can build this platform
    local can_build=true
    if [ "$platform" != "$HOST_TARGET" ]; then
        local target_os="${platform%-*}"
        if [ "$target_os" != "$HOST_OS" ]; then
            log_warn "  Skipping: cross-OS compilation not supported"
            log_warn "  (Cannot build $platform from $HOST_TARGET)"
            can_build=false
        fi
    fi
    
    if [ "$can_build" = true ]; then
        "$BUILD_SCRIPT" \
            --target "$platform" \
            --features "$features" \
            --profile release \
            --output-dir "$OUTPUT_DIR" \
            --output-name "$output_name" \
            || {
                log_error "  Build failed for $output_name"
                return 1
            }
        log_info "  Success: $OUTPUT_DIR/$output_name"
    fi
    
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    parse_args "$@"
    
    # Clean output directory if requested
    if [ "$CLEAN" = true ] && [ -d "$OUTPUT_DIR" ]; then
        log_info "Cleaning output directory: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    # Parse platforms
    IFS=',' read -ra platform_array <<< "$PLATFORMS"
    
    # Summary
    echo ""
    log_step "=== Release Build Plan ==="
    echo "  Version:    $VERSION"
    echo "  Output:     $OUTPUT_DIR"
    echo "  Host:       $HOST_TARGET"
    echo "  Builds:     ${#platform_array[@]} total"
    echo ""
    
    if [ ${#platform_array[@]} -eq 0 ]; then
        log_error "No builds to perform!"
        exit 1
    fi
    
    log_info "Planned builds:"
    for platform in "${platform_array[@]}"; do
        platform=$(echo "$platform" | xargs)
        local backends=$(get_platform_backends "$platform")
        echo "    - $(get_output_name "$platform")  ($backends)"
    done
    echo ""
    
    # Build each platform
    local success=0
    local failed=0
    local skipped=0
    
    for platform in "${platform_array[@]}"; do
        platform=$(echo "$platform" | xargs)
        
        if build_platform "$platform"; then
            if [ "$DRY_RUN" = true ]; then
                ((skipped++))
            else
                ((success++))
            fi
        else
            ((failed++))
        fi
    done
    
    # Summary
    echo ""
    log_step "=== Build Summary ==="
    echo "  Successful: $success"
    echo "  Failed:     $failed"
    echo "  Skipped:    $skipped"
    echo ""
    
    if [ "$DRY_RUN" = true ]; then
        log_info "Dry run complete. Use without --dry-run to build."
    elif [ $failed -gt 0 ]; then
        log_error "Some builds failed!"
        exit 1
    else
        log_info "All builds complete!"
        log_info "Releases in: $OUTPUT_DIR"
        
        if [ -d "$OUTPUT_DIR" ]; then
            echo ""
            log_info "Output files:"
            ls -lh "$OUTPUT_DIR" 2>/dev/null | tail -n +2 | while read -r line; do
                echo "    $line"
            done
        fi
        
        echo ""
        log_info "Runtime usage:"
        echo "    ./torchless-* --backend auto model.bin \"prompt\"   # Auto-select (GPU if available)"
        echo "    ./torchless-* --backend cuda model.bin \"prompt\"   # Force CUDA"
        echo "    ./torchless-* --backend cpu model.bin \"prompt\"    # Force CPU only"
        echo "    ./torchless-* --list-backends                      # List available backends"
    fi
}

main "$@"
