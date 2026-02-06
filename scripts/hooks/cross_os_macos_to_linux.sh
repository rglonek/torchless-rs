#!/bin/bash
#
# Cross-OS hook: macOS -> Linux (x86_64/aarch64) via zig.
# Requires: zig on PATH and rustup target installed by build.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="/usr/local/bin:$PATH"
if ! command -v zig >/dev/null 2>&1; then
    echo "[HOOK] zig not found; install zig to cross-compile Linux targets." >&2
    return 1
fi

case "$TARGET_ARCH" in
    x86_64)
        ZIG_TARGET="x86_64-linux-gnu"
        ;;
    aarch64)
        ZIG_TARGET="aarch64-linux-gnu"
        ;;
    *)
        echo "[HOOK] Unsupported Linux arch: $TARGET_ARCH" >&2
        return 1
        ;;
esac

export CC="zig cc -target $ZIG_TARGET"
export CXX="zig c++ -target $ZIG_TARGET"
export AR="zig ar"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$SCRIPT_DIR/zig_cc_linux_x86_64.sh"
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="$SCRIPT_DIR/zig_cc_linux_aarch64.sh"

# Cross-OS builds from macOS won't have Linux GPU libs available; build CPU-only.
export FEATURES_OVERRIDE="simd,parallel"

return 0
