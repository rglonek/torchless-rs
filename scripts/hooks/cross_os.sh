#!/bin/bash
#
# Cross-OS build hook (optional).
# This script is invoked when --allow-cross-os is set and a target OS differs from the host.
#
# You can set environment variables here for toolchains, linkers, SDKs, etc.
# Example (Linux target using zig):
#   export CC="zig cc -target x86_64-linux-gnu"
#   export CXX="zig c++ -target x86_64-linux-gnu"
#   export AR="zig ar"
#   export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="zig cc -target x86_64-linux-gnu"
#
# Environment variables provided:
#   HOST_OS, HOST_TARGET, TARGET, TARGET_OS, TARGET_ARCH, RUST_TARGET,
#   FEATURES, OUTPUT_DIR, OUTPUT_NAME
#
# Return 0 to continue, non-zero to abort this platform build.
return 0
