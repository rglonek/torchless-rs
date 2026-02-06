#!/bin/bash
#
# Build multiple release flavors of torchless for distribution.
# Creates binaries for different platforms and GPU backends.
#
# Usage:
#   ./scripts/build_releases.sh [OPTIONS]
#   ./scripts/build_releases.sh --platforms linux-x86_64,macos-aarch64
#   ./scripts/build_releases.sh --flavors cpu,cuda,metal
#
# See --help for all options.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_SCRIPT="$SCRIPT_DIR/build.sh"

# =============================================================================
# Colors and Logging
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[RELEASE]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[RELEASE]${NC} $1"; }
log_error() { echo -e "${RED}[RELEASE]${NC} $1"; }
log_step()  { echo -e "${CYAN}[RELEASE]${NC} ${BOLD}$1${NC}"; }

# =============================================================================
# Default Configuration
# =============================================================================

# Detect host for determining which platforms we can build
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

# Default: build for host platform only (cross-compilation requires setup)
PLATFORMS="${PLATFORMS:-$HOST_TARGET}"
FLAVORS="${FLAVORS:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/releases}"
VERSION="${VERSION:-$(grep '^version' "$PROJECT_DIR/Cargo.toml" | head -1 | sed 's/.*"\(.*\)"/\1/')}"
PARALLEL="${PARALLEL:-false}"
DRY_RUN="${DRY_RUN:-false}"
CLEAN="${CLEAN:-false}"

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat <<EOF
${BOLD}torchless Release Builder${NC}

Builds multiple release variants of torchless for distribution.

Usage: $0 [OPTIONS]

${BOLD}Platform Options:${NC}
  --platforms PLATFORMS   Comma-separated build targets (default: host only)
                          Available: linux-x86_64, linux-aarch64,
                                     macos-x86_64, macos-aarch64,
                                     windows-x86_64
  --all-platforms         Build for all platforms (requires cross-compilation setup)
  --host-only             Build only for current host (default)

${BOLD}Flavor Options:${NC}
  --flavors FLAVORS       Comma-separated GPU flavors (default: all)
                          Available: cpu, cuda, rocm, metal, opencl, all
  --cpu-only              Build CPU-only flavor
  --gpu-all               Build all GPU flavors for each platform

${BOLD}Output Options:${NC}
  --output-dir DIR        Output directory (default: releases/)
  --version VERSION       Version string for naming (default: from Cargo.toml)
  --clean                 Clean output directory before building

${BOLD}Build Options:${NC}
  --parallel              Build flavors in parallel (experimental)
  --dry-run               Show what would be built without building

${BOLD}Other:${NC}
  -h, --help              Show this help

${BOLD}Platform/Flavor Matrix:${NC}
  linux-x86_64:   cpu, cuda, rocm, opencl
  linux-aarch64:  cpu, opencl
  macos-x86_64:   cpu, opencl
  macos-aarch64:  cpu, metal, opencl
  windows-x86_64: cpu, cuda, opencl

${BOLD}Examples:${NC}
  # Build all flavors for current host
  $0

  # Build CPU-only for all platforms
  $0 --all-platforms --cpu-only

  # Build Linux CUDA and ROCm releases
  $0 --platforms linux-x86_64 --flavors cuda,rocm

  # Build macOS releases with Metal
  $0 --platforms macos-aarch64 --flavors cpu,metal

  # Preview all builds
  $0 --all-platforms --dry-run

${BOLD}Output Structure:${NC}
  releases/
  ├── torchless-${VERSION}-linux-x86_64-cpu
  ├── torchless-${VERSION}-linux-x86_64-cuda
  ├── torchless-${VERSION}-macos-aarch64-cpu
  ├── torchless-${VERSION}-macos-aarch64-metal
  └── ...
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
            --flavors)
                FLAVORS="$2"; shift 2 ;;
            --cpu-only)
                FLAVORS="cpu"; shift ;;
            --gpu-all)
                FLAVORS="all"; shift ;;
            --output-dir)
                OUTPUT_DIR="$2"; shift 2 ;;
            --version)
                VERSION="$2"; shift 2 ;;
            --clean)
                CLEAN=true; shift ;;
            --parallel)
                PARALLEL=true; shift ;;
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

# Get available flavors for a given platform
get_platform_flavors() {
    local platform="$1"
    case "$platform" in
        linux-x86_64)
            echo "cpu cuda rocm opencl" ;;
        linux-aarch64)
            echo "cpu opencl" ;;
        macos-x86_64)
            echo "cpu opencl" ;;
        macos-aarch64)
            echo "cpu metal opencl" ;;
        windows-x86_64)
            echo "cpu cuda opencl" ;;
        *)
            echo "cpu" ;;
    esac
}

# Convert flavor to cargo features
flavor_to_features() {
    local flavor="$1"
    local base_features="simd,parallel"
    
    case "$flavor" in
        cpu)
            echo "$base_features" ;;
        cuda)
            echo "$base_features,cuda" ;;
        rocm)
            echo "$base_features,rocm" ;;
        metal)
            echo "$base_features,metal-gpu" ;;
        opencl)
            echo "$base_features,opencl" ;;
        *)
            echo "$base_features" ;;
    esac
}

# Generate output filename
get_output_name() {
    local platform="$1"
    local flavor="$2"
    local ext=""
    
    [[ "$platform" == windows-* ]] && ext=".exe"
    
    echo "torchless-${VERSION}-${platform}-${flavor}${ext}"
}

# =============================================================================
# Build Functions
# =============================================================================

build_flavor() {
    local platform="$1"
    local flavor="$2"
    local features output_name
    
    features=$(flavor_to_features "$flavor")
    output_name=$(get_output_name "$platform" "$flavor")
    
    log_step "Building: $output_name"
    log_info "  Platform: $platform"
    log_info "  Flavor:   $flavor"
    log_info "  Features: $features"
    
    if [ "$DRY_RUN" = true ]; then
        log_info "  [DRY RUN - skipping actual build]"
        return 0
    fi
    
    # Check if we can build this platform
    local can_build=true
    if [ "$platform" != "$HOST_TARGET" ]; then
        # Cross-compilation check
        local target_os="${platform%-*}"
        if [ "$target_os" != "$HOST_OS" ]; then
            log_warn "  Skipping: cross-OS compilation requires additional setup"
            log_warn "  (Building $platform from $HOST_TARGET)"
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
    
    # Build list of (platform, flavor) pairs
    declare -a BUILD_JOBS=()
    
    IFS=',' read -ra platform_array <<< "$PLATFORMS"
    IFS=',' read -ra flavor_array <<< "$FLAVORS"
    
    for platform in "${platform_array[@]}"; do
        platform=$(echo "$platform" | xargs)  # trim whitespace
        
        # Get available flavors for this platform
        available_flavors=$(get_platform_flavors "$platform")
        
        for flavor in "${flavor_array[@]}"; do
            flavor=$(echo "$flavor" | xargs)  # trim whitespace
            
            if [ "$flavor" = "all" ]; then
                # Add all available flavors for this platform
                for available in $available_flavors; do
                    BUILD_JOBS+=("$platform:$available")
                done
            elif echo "$available_flavors" | grep -qw "$flavor"; then
                # Flavor is available for this platform
                BUILD_JOBS+=("$platform:$flavor")
            else
                log_warn "Flavor '$flavor' not available for $platform, skipping"
            fi
        done
    done
    
    # Remove duplicates
    BUILD_JOBS=($(printf '%s\n' "${BUILD_JOBS[@]}" | sort -u))
    
    # Summary
    echo ""
    log_step "=== Release Build Plan ==="
    echo "  Version:    $VERSION"
    echo "  Output:     $OUTPUT_DIR"
    echo "  Host:       $HOST_TARGET"
    echo "  Builds:     ${#BUILD_JOBS[@]} total"
    echo ""
    
    if [ ${#BUILD_JOBS[@]} -eq 0 ]; then
        log_error "No builds to perform!"
        exit 1
    fi
    
    log_info "Planned builds:"
    for job in "${BUILD_JOBS[@]}"; do
        local plat="${job%:*}"
        local flav="${job#*:}"
        echo "    - $(get_output_name "$plat" "$flav")"
    done
    echo ""
    
    # Build each job
    local success=0
    local failed=0
    local skipped=0
    
    for job in "${BUILD_JOBS[@]}"; do
        local platform="${job%:*}"
        local flavor="${job#*:}"
        
        if build_flavor "$platform" "$flavor"; then
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
        
        # List output files
        if [ -d "$OUTPUT_DIR" ] && command -v ls &>/dev/null; then
            echo ""
            log_info "Output files:"
            ls -lh "$OUTPUT_DIR" | tail -n +2 | while read -r line; do
                echo "    $line"
            done
        fi
    fi
}

main "$@"
