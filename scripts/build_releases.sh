#!/bin/bash
#
# Build release flavors of torchless for distribution.
# Creates multi-backend binaries to minimize the number of releases.
#
# Flavors:
#   - cpu: CPU-only (smallest binary, most portable)
#   - gpu: All GPU backends for the platform compiled in (user selects at runtime)
#
# Usage:
#   ./scripts/build_releases.sh [OPTIONS]
#   ./scripts/build_releases.sh --platforms linux-x86_64,macos-aarch64
#   ./scripts/build_releases.sh --flavors cpu
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
FLAVORS="${FLAVORS:-cpu,gpu}"
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

Builds release binaries with multi-backend support.
Users select the backend at runtime with --backend flag.

Usage: $0 [OPTIONS]

${BOLD}Platform Options:${NC}
  --platforms PLATFORMS   Comma-separated build targets (default: host only)
                          Available: linux-x86_64, linux-aarch64,
                                     macos-x86_64, macos-aarch64,
                                     windows-x86_64
  --all-platforms         Build for all platforms (requires cross-compilation)
  --host-only             Build only for current host (default)

${BOLD}Flavor Options:${NC}
  --flavors FLAVORS       Comma-separated flavors (default: cpu,gpu)
                          cpu - CPU only (smaller, portable)
                          gpu - All GPU backends for platform
  --cpu-only              Build only CPU flavor
  --gpu-only              Build only GPU flavor
  --all                   Build both cpu and gpu flavors (default)

${BOLD}Output Options:${NC}
  --output-dir DIR        Output directory (default: releases/)
  --version VERSION       Version string for naming (default: from Cargo.toml)
  --clean                 Clean output directory before building

${BOLD}Other:${NC}
  --dry-run               Show what would be built without building
  -h, --help              Show this help

${BOLD}GPU Backends per Platform:${NC}
  linux-x86_64:   cuda, rocm, opencl
  linux-aarch64:  opencl
  macos-x86_64:   opencl
  macos-aarch64:  metal, opencl
  windows-x86_64: cuda, opencl

${BOLD}Runtime Backend Selection:${NC}
  The GPU flavor includes all available backends. Users select at runtime:

    ./torchless --backend auto model.bin "prompt"     # Auto-select best
    ./torchless --backend cuda model.bin "prompt"     # Force CUDA
    ./torchless --backend opencl model.bin "prompt"   # Force OpenCL
    ./torchless --list-backends                       # Show available

${BOLD}Examples:${NC}
  # Build both cpu and gpu for current host
  $0

  # Build GPU-only for Linux
  $0 --platforms linux-x86_64 --gpu-only

  # Build CPU-only for all platforms
  $0 --all-platforms --cpu-only

  # Preview builds
  $0 --all-platforms --dry-run

${BOLD}Output Structure:${NC}
  releases/
  ├── torchless-${VERSION}-linux-x86_64-cpu      # CPU only
  ├── torchless-${VERSION}-linux-x86_64-gpu      # CUDA + ROCm + OpenCL
  ├── torchless-${VERSION}-macos-aarch64-cpu     # CPU only  
  ├── torchless-${VERSION}-macos-aarch64-gpu     # Metal + OpenCL
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
            --gpu-only)
                FLAVORS="gpu"; shift ;;
            --all)
                FLAVORS="cpu,gpu"; shift ;;
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

# Get GPU features for a given platform
get_gpu_features() {
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

# Get CPU features for a given platform
get_cpu_features() {
    local platform="$1"
    local base="simd,parallel"
    
    case "$platform" in
        macos-*)
            echo "${base},accelerate" ;;
        linux-*)
            echo "${base}" ;;
        *)
            echo "${base}" ;;
    esac
}

# Convert flavor to cargo features
flavor_to_features() {
    local platform="$1"
    local flavor="$2"
    
    case "$flavor" in
        cpu)
            get_cpu_features "$platform" ;;
        gpu)
            get_gpu_features "$platform" ;;
        *)
            get_cpu_features "$platform" ;;
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

# Get description for flavor
get_flavor_description() {
    local platform="$1"
    local flavor="$2"
    
    case "$flavor" in
        cpu)
            echo "CPU only (simd+parallel)" ;;
        gpu)
            case "$platform" in
                linux-x86_64)
                    echo "GPU (cuda+rocm+opencl)" ;;
                linux-aarch64)
                    echo "GPU (opencl)" ;;
                macos-x86_64)
                    echo "GPU (opencl)" ;;
                macos-aarch64)
                    echo "GPU (metal+opencl)" ;;
                windows-x86_64)
                    echo "GPU (cuda+opencl)" ;;
                *)
                    echo "GPU" ;;
            esac
            ;;
        *)
            echo "$flavor" ;;
    esac
}

# =============================================================================
# Build Functions
# =============================================================================

build_flavor() {
    local platform="$1"
    local flavor="$2"
    local features output_name description
    
    features=$(flavor_to_features "$platform" "$flavor")
    output_name=$(get_output_name "$platform" "$flavor")
    description=$(get_flavor_description "$platform" "$flavor")
    
    log_step "Building: $output_name"
    log_info "  Platform:    $platform"
    log_info "  Flavor:      $flavor - $description"
    log_info "  Features:    $features"
    
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
    
    # Build list of (platform, flavor) pairs
    declare -a BUILD_JOBS=()
    
    IFS=',' read -ra platform_array <<< "$PLATFORMS"
    IFS=',' read -ra flavor_array <<< "$FLAVORS"
    
    for platform in "${platform_array[@]}"; do
        platform=$(echo "$platform" | xargs)
        
        for flavor in "${flavor_array[@]}"; do
            flavor=$(echo "$flavor" | xargs)
            BUILD_JOBS+=("$platform:$flavor")
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
        local desc=$(get_flavor_description "$plat" "$flav")
        echo "    - $(get_output_name "$plat" "$flav")  ($desc)"
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
        
        if [ -d "$OUTPUT_DIR" ]; then
            echo ""
            log_info "Output files:"
            ls -lh "$OUTPUT_DIR" 2>/dev/null | tail -n +2 | while read -r line; do
                echo "    $line"
            done
        fi
        
        echo ""
        log_info "Runtime usage:"
        echo "    ./torchless-*-gpu --backend auto model.bin \"prompt\"    # Auto-select"
        echo "    ./torchless-*-gpu --backend cuda model.bin \"prompt\"    # Force CUDA"
        echo "    ./torchless-*-gpu --list-backends                       # List available"
    fi
}

main "$@"
