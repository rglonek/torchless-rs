#!/bin/bash
#
# Cross-OS hook: macOS -> Windows x86_64 via zig.
# Uses GNU target; requires zig on PATH.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="/usr/local/bin:$PATH"
if ! command -v zig >/dev/null 2>&1; then
    echo "[HOOK] zig not found; install zig to cross-compile Windows targets." >&2
    return 1
fi

# Use GNU Windows target for cross-compilation from macOS
export RUST_TARGET_OVERRIDE="x86_64-pc-windows-gnu"
export CC="zig cc -target x86_64-windows-gnu"
export CXX="zig c++ -target x86_64-windows-gnu"
export AR="zig ar"
export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER="$SCRIPT_DIR/zig_cc_windows_x86_64.sh"

# Cross-OS builds from macOS won't have Windows GPU libs available; build CPU-only.
export FEATURES_OVERRIDE="simd,parallel"

return 0
