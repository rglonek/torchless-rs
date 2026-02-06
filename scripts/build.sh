#!/bin/bash
#
# Flexible build script for torchless with support for different targets,
# features, and optimization levels.
#
# Usage:
#   ./scripts/build.sh [OPTIONS]
#   ./scripts/build.sh --target linux-x86_64 --features cuda,simd,parallel
#   ./scripts/build.sh --target macos-aarch64 --features metal-gpu,simd,parallel
#
# See --help for all options.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Colors and Logging
# =============================================================================

# Use $'...' syntax for portable color codes
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
NC=$'\033[0m'

log_info()  { printf '%s[BUILD]%s %s\n' "$GREEN" "$NC" "$1"; }
log_warn()  { printf '%s[BUILD]%s %s\n' "$YELLOW" "$NC" "$1"; }
log_error() { printf '%s[BUILD]%s %s\n' "$RED" "$NC" "$1"; }
log_step()  { printf '%s[BUILD]%s %s%s%s\n' "$CYAN" "$NC" "$BOLD" "$1" "$NC"; }

# =============================================================================
# Default Configuration
# =============================================================================

# Detect host system
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
TARGET="${TARGET:-$HOST_TARGET}"
FEATURES="${FEATURES:-simd,parallel}"
PROFILE="${PROFILE:-release}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_NAME="${OUTPUT_NAME:-}"
PGO="${PGO:-false}"
PGO_DIR="${PGO_DIR:-/tmp/torchless-pgo}"
RUSTFLAGS_EXTRA="${RUSTFLAGS_EXTRA:-}"
NATIVE_CPU="${NATIVE_CPU:-false}"
STRIP_BINARY="${STRIP_BINARY:-true}"
VERBOSE="${VERBOSE:-false}"

# =============================================================================
# Help
# =============================================================================

show_help() {
    cat <<EOF
${BOLD}torchless build script${NC}

Usage: $0 [OPTIONS]

${BOLD}Target Options:${NC}
  --target TARGET       Build target: linux-x86_64, linux-aarch64, macos-x86_64,
                        macos-aarch64, windows-x86_64 (default: host system)
  --native              Optimize for native CPU (adds -C target-cpu=native)

${BOLD}Feature Options:${NC}
  --features FEATURES   Comma-separated cargo features (default: simd,parallel)
                        Available: simd, parallel, cuda, rocm, metal-gpu, opencl,
                                   openblas, blis, accelerate
  --cpu-only            Build without any GPU backends (simd,parallel only)
  --gpu                 Add all GPU backends available for target (recommended)
  --cuda                Add CUDA support
  --rocm                Add ROCm support
  --metal               Add Metal support
  --opencl              Add OpenCL support
  --all-gpu             Alias for --gpu

${BOLD}Build Options:${NC}
  --profile PROFILE     Build profile: release, debug, profiling (default: release)
  --pgo                 Enable Profile-Guided Optimization (uses build_pgo.sh)
  --pgo-dir DIR         Directory for PGO data (default: /tmp/torchless-pgo)

${BOLD}Output Options:${NC}
  --output-dir DIR      Output directory (default: target/<profile>)
  --output-name NAME    Output binary name (default: torchless-<target>-<features>)
  --no-strip            Don't strip debug symbols from release builds

${BOLD}Other:${NC}
  --verbose             Show detailed build output
  -h, --help            Show this help

${BOLD}Examples:${NC}
  # Build for current host with defaults (simd,parallel)
  $0

  # Build Linux release with CUDA support
  $0 --target linux-x86_64 --cuda

  # Build macOS Apple Silicon with Metal
  $0 --target macos-aarch64 --metal

  # Build CPU-only with native optimizations
  $0 --cpu-only --native

  # Build with PGO optimization
  $0 --pgo --features simd,parallel,cuda

${BOLD}Environment Variables:${NC}
  TARGET, FEATURES, PROFILE, OUTPUT_DIR, OUTPUT_NAME, PGO, PGO_DIR,
  RUSTFLAGS_EXTRA, NATIVE_CPU, STRIP_BINARY, VERBOSE
EOF
}

# =============================================================================
# Argument Parsing
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --target)
                TARGET="$2"; shift 2 ;;
            --features)
                FEATURES="$2"; shift 2 ;;
            --cpu-only)
                FEATURES="simd,parallel"; shift ;;
            --cuda)
                FEATURES="${FEATURES},cuda"; shift ;;
            --rocm)
                FEATURES="${FEATURES},rocm"; shift ;;
            --metal)
                FEATURES="${FEATURES},metal-gpu"; shift ;;
            --opencl)
                FEATURES="${FEATURES},opencl"; shift ;;
            --gpu|--all-gpu)
                ALL_GPU=true; shift ;;
            --profile)
                PROFILE="$2"; shift 2 ;;
            --pgo)
                PGO=true; shift ;;
            --pgo-dir)
                PGO_DIR="$2"; shift 2 ;;
            --output-dir)
                OUTPUT_DIR="$2"; shift 2 ;;
            --output-name)
                OUTPUT_NAME="$2"; shift 2 ;;
            --native)
                NATIVE_CPU=true; shift ;;
            --no-strip)
                STRIP_BINARY=false; shift ;;
            --verbose)
                VERBOSE=true; shift ;;
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
# Validation and Configuration
# =============================================================================

validate_target() {
    local os arch
    IFS='-' read -r os arch <<< "$TARGET"
    
    case "$os" in
        linux|macos|windows) ;;
        *)
            log_error "Invalid OS in target: $os (expected: linux, macos, windows)"
            exit 1 ;;
    esac
    
    case "$arch" in
        x86_64|aarch64) ;;
        *)
            log_error "Invalid arch in target: $arch (expected: x86_64, aarch64)"
            exit 1 ;;
    esac
    
    TARGET_OS="$os"
    TARGET_ARCH="$arch"
}

configure_target_triple() {
    if [ -n "${RUST_TARGET_OVERRIDE:-}" ]; then
        RUST_TARGET="$RUST_TARGET_OVERRIDE"
        log_info "Using RUST_TARGET override: $RUST_TARGET"
        return 0
    fi
    case "$TARGET" in
        linux-x86_64)   RUST_TARGET="x86_64-unknown-linux-gnu" ;;
        linux-aarch64)  RUST_TARGET="aarch64-unknown-linux-gnu" ;;
        macos-x86_64)   RUST_TARGET="x86_64-apple-darwin" ;;
        macos-aarch64)  RUST_TARGET="aarch64-apple-darwin" ;;
        windows-x86_64) RUST_TARGET="x86_64-pc-windows-msvc" ;;
        *)
            log_error "Unsupported target: $TARGET"
            exit 1 ;;
    esac
}

# Ensure the Rust target is installed when cross-compiling (e.g. macos-aarch64 from macos-x86_64).
# We add the target to the default toolchain and run cargo via "rustup run <toolchain> cargo"
# so the build uses the same toolchain we installed the target for.
ensure_target_installed() {
    if [ "$TARGET" = "$HOST_TARGET" ]; then
        return 0
    fi
    if ! command -v rustup &>/dev/null; then
        log_warn "rustup not found; ensure target $RUST_TARGET is installed for cross-compilation"
        return 0
    fi
    # Resolve the actual default toolchain name (e.g. stable-x86_64-apple-darwin); "default" is not valid for rustup run
    local toolchain
    toolchain=$(rustup show 2>/dev/null | grep '(default)' | head -1 | awk '{print $1}')
    if [ -z "$toolchain" ]; then
        log_error "Could not determine default Rust toolchain. Run: rustup target add $RUST_TARGET"
        exit 1
    fi
    log_info "Using toolchain: $toolchain"
    # Add target to that toolchain (idempotent)
    log_info "Ensuring Rust target $RUST_TARGET for $toolchain"
    rustup target add --toolchain "$toolchain" "$RUST_TARGET" || {
        log_error "Failed to install target $RUST_TARGET. Run: rustup target add --toolchain $toolchain $RUST_TARGET"
        exit 1
    }
    # Verify rustc for this toolchain can see the target
    if ! rustup run "$toolchain" rustc --target "$RUST_TARGET" --print sysroot >/dev/null 2>&1; then
        log_error "Toolchain $toolchain cannot use target $RUST_TARGET. Run: rustup target add --toolchain $toolchain $RUST_TARGET"
        exit 1
    fi
    # Force cargo/rustc to use this toolchain's binaries even if RUSTC is set
    export RUSTUP_TOOLCHAIN="$toolchain"
    export RUSTC="$(rustup which rustc --toolchain "$toolchain" 2>/dev/null)"
    export CARGO="$(rustup which cargo --toolchain "$toolchain" 2>/dev/null)"
}

configure_features() {
    # Handle --all-gpu based on target
    if [ "${ALL_GPU:-false}" = true ]; then
        case "$TARGET_OS" in
            linux)
                if [ "$TARGET_ARCH" = "x86_64" ]; then
                    FEATURES="${FEATURES},cuda,rocm,opencl"
                else
                    FEATURES="${FEATURES},opencl"
                fi
                ;;
            macos)
                if [ "$TARGET_ARCH" = "aarch64" ]; then
                    FEATURES="${FEATURES},metal-gpu,opencl"
                else
                    FEATURES="${FEATURES},opencl"
                fi
                ;;
            windows)
                FEATURES="${FEATURES},cuda,opencl"
                ;;
        esac
    fi
    
    # Remove duplicates from features
    FEATURES=$(echo "$FEATURES" | tr ',' '\n' | sort -u | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
    
    # Validate feature compatibility
    if [[ "$FEATURES" == *"metal-gpu"* ]] && [ "$TARGET_OS" != "macos" ]; then
        log_warn "Metal GPU is only available on macOS, removing from features"
        FEATURES=$(echo "$FEATURES" | sed 's/metal-gpu,\?//g' | sed 's/,$//')
    fi
    
    if [[ "$FEATURES" == *"accelerate"* ]] && [ "$TARGET_OS" != "macos" ]; then
        log_warn "Accelerate is only available on macOS, removing from features"
        FEATURES=$(echo "$FEATURES" | sed 's/accelerate,\?//g' | sed 's/,$//')
    fi
    
    if [[ "$FEATURES" == *"rocm"* ]] && [ "$TARGET_OS" != "linux" ]; then
        log_warn "ROCm is only available on Linux, removing from features"
        FEATURES=$(echo "$FEATURES" | sed 's/rocm,\?//g' | sed 's/,$//')
    fi
}

configure_output() {
    # Default output directory
    if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="$PROJECT_DIR/target/$PROFILE"
    fi
    
    # Default output name based on target and features
    if [ -z "$OUTPUT_NAME" ]; then
        local feature_suffix=""
        if [[ "$FEATURES" == *"cuda"* ]]; then
            feature_suffix="${feature_suffix}-cuda"
        fi
        if [[ "$FEATURES" == *"rocm"* ]]; then
            feature_suffix="${feature_suffix}-rocm"
        fi
        if [[ "$FEATURES" == *"metal-gpu"* ]]; then
            feature_suffix="${feature_suffix}-metal"
        fi
        if [[ "$FEATURES" == *"opencl"* ]]; then
            feature_suffix="${feature_suffix}-opencl"
        fi
        if [ -z "$feature_suffix" ]; then
            feature_suffix="-cpu"
        fi
        
        local ext=""
        [ "$TARGET_OS" = "windows" ] && ext=".exe"
        
        OUTPUT_NAME="torchless-${TARGET}${feature_suffix}${ext}"
    fi
}

configure_rustflags() {
    RUSTFLAGS=""
    
    # Native CPU optimization
    if [ "$NATIVE_CPU" = true ]; then
        RUSTFLAGS="$RUSTFLAGS -C target-cpu=native"
    fi
    
    # Extra RUSTFLAGS from environment
    if [ -n "$RUSTFLAGS_EXTRA" ]; then
        RUSTFLAGS="$RUSTFLAGS $RUSTFLAGS_EXTRA"
    fi
    
    # Trim leading/trailing spaces
    RUSTFLAGS=$(echo "$RUSTFLAGS" | xargs)
}

# =============================================================================
# Build Functions
# =============================================================================

build_regular() {
    log_step "Building torchless..."
    log_info "Target:   $TARGET ($RUST_TARGET)"
    log_info "Features: $FEATURES"
    log_info "Profile:  $PROFILE"
    [ -n "$RUSTFLAGS" ] && log_info "RUSTFLAGS: $RUSTFLAGS"
    
    local cargo_args=("build")
    
    # Profile
    case "$PROFILE" in
        release)
            cargo_args+=("--release")
            ;;
        debug)
            # default profile, no flag needed
            ;;
        profiling)
            cargo_args+=("--profile" "profiling")
            ;;
        *)
            log_error "Unknown profile: $PROFILE"
            exit 1
            ;;
    esac
    
    # Features
    if [ -n "$FEATURES" ]; then
        cargo_args+=("--features" "$FEATURES")
    fi
    
    # Cross-compilation target (only if different from host)
    if [ "$TARGET" != "$HOST_TARGET" ]; then
        cargo_args+=("--target" "$RUST_TARGET")
    fi
    
    # Verbose
    if [ "$VERBOSE" = true ]; then
        cargo_args+=("-v")
    fi
    
    # Run build (use rustup run when cross-compiling so we use the toolchain we installed the target for)
    cd "$PROJECT_DIR"
    local cargo_cmd="cargo"
    if [ -n "${CARGO:-}" ]; then
        cargo_cmd="$CARGO"
    elif [ "$TARGET" != "$HOST_TARGET" ] && [ -n "${RUSTUP_TOOLCHAIN:-}" ]; then
        cargo_cmd="rustup run $RUSTUP_TOOLCHAIN cargo"
    fi
    if [ -n "$RUSTFLAGS" ]; then
        RUSTFLAGS="$RUSTFLAGS" $cargo_cmd "${cargo_args[@]}"
    else
        $cargo_cmd "${cargo_args[@]}"
    fi
}

build_pgo() {
    log_step "Building with PGO optimization..."
    
    # Use the existing PGO script with our features
    export FEATURES
    export PGO_DIR
    
    "$SCRIPT_DIR/build_pgo.sh"
}

copy_output() {
    local src_dir="$PROJECT_DIR/target"
    
    # Determine source path
    if [ "$TARGET" != "$HOST_TARGET" ]; then
        src_dir="$src_dir/$RUST_TARGET"
    fi
    
    case "$PROFILE" in
        release|profiling)
            src_dir="$src_dir/release"
            ;;
        debug)
            src_dir="$src_dir/debug"
            ;;
    esac
    
    local src_binary="$src_dir/torchless"
    [ "$TARGET_OS" = "windows" ] && src_binary="${src_binary}.exe"
    
    if [ ! -f "$src_binary" ]; then
        log_error "Build output not found: $src_binary"
        exit 1
    fi
    
    # Create output directory if needed
    mkdir -p "$OUTPUT_DIR"
    
    local dest_binary="$OUTPUT_DIR/$OUTPUT_NAME"
    
    # Copy binary
    cp "$src_binary" "$dest_binary"
    
    # Strip debug symbols for release builds (if not cross-compiling and strip available)
    if [ "$STRIP_BINARY" = true ] && [ "$PROFILE" = "release" ] && [ "$TARGET" = "$HOST_TARGET" ]; then
        if command -v strip &>/dev/null; then
            strip "$dest_binary" 2>/dev/null || true
        fi
    fi
    
    log_info "Output: $dest_binary"
    
    # Show binary size
    if command -v du &>/dev/null; then
        local size=$(du -h "$dest_binary" | cut -f1)
        log_info "Size: $size"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    parse_args "$@"
    validate_target
    configure_target_triple
    ensure_target_installed
    configure_features
    configure_output
    configure_rustflags
    
    echo ""
    log_step "=== torchless Build Configuration ==="
    echo "  Target:      $TARGET"
    echo "  Features:    $FEATURES"
    echo "  Profile:     $PROFILE"
    echo "  PGO:         $PGO"
    echo "  Output:      $OUTPUT_DIR/$OUTPUT_NAME"
    echo ""
    
    if [ "$PGO" = true ]; then
        build_pgo
    else
        build_regular
    fi
    
    copy_output
    
    echo ""
    log_info "Build complete!"
}

main "$@"
