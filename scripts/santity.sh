#!/bin/bash

set -euxo pipefail

# if we are in scripts, cd ..
if [ "$(basename "$(pwd)")" == "scripts" ]; then
    cd ..
fi

# run cargo clippy
echo "Running cargo clippy"
cargo clippy -- --deny warnings

# run cargo fmt
echo "Running cargo fmt"
cargo fmt --all -- --check

# run cargo test
echo "Running cargo test"
cargo test --verbose
