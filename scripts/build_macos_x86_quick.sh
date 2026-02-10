#!/bin/bash

# if we are in scripts, cd ..
if [ "$(basename "$(pwd)")" == "scripts" ]; then
    cd ..
fi

set -ex
cargo build --release --verbose --features "simd,parallel,opencl,accelerate,webgpu" --target x86_64-apple-darwin
cp target/x86_64-apple-darwin/release/torchless releases/torchless-0.1.0-macos-x86_64
cd releases
./torchless-0.1.0-macos-x86_64 ../models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf --debug --max-seq-len 32768 --chat --system "You are a helpful assistant. But you do not hold back if you need to say I am wrong or tell me off."
