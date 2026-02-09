#!/bin/bash

echo "Checking for huggingface-cli or hf..."
HF_CMD=""
if command -v huggingface-cli &> /dev/null; then
    HF_CMD="huggingface-cli"
elif command -v hf &> /dev/null; then
    HF_CMD="hf"
fi

if [ -z "$HF_CMD" ]; then
    echo "huggingface-cli/hf not found, installing..."
    if [ "$(uname)" == "Darwin" ]; then
        brew install huggingface-cli || exit 1
    fi

    if [ "$(uname)" == "Linux" ]; then
        sudo apt-get install huggingface-cli || exit 1
    fi

    if [ "$(uname)" == "Windows" ]; then
        scoop install huggingface-cli || exit 1
    fi
    if command -v huggingface-cli &> /dev/null; then
        HF_CMD="huggingface-cli"
    elif command -v hf &> /dev/null; then
        HF_CMD="hf"
    else
        echo "ERROR: Installation failed. Please install manually: pip install huggingface_hub"
        exit 1
    fi
fi

echo "Using: $HF_CMD"

if [ "$1" == "q4_k_m" ]; then
    # Q4_K_M GGUF (~4.5 GB)
    echo "Downloading Q4_K_M DeepSeek-R1-Distill-Llama-8B-GGUF (~4.5 GB)"
    $HF_CMD download bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF --include "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf" --local-dir . || exit 1
elif [ "$1" == "f16" ]; then
    # FP16 GGUF (~16 GB)
    echo "Downloading FP16 DeepSeek-R1-Distill-Llama-8B-GGUF (~16 GB)"
    $HF_CMD download bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF --include "DeepSeek-R1-Distill-Llama-8B-f16.gguf" --local-dir . || exit 1
else
    # both at the same time
    echo "Downloading FP16 and Q4_k_m DeepSeek-R1-Distill-Llama-8B-GGUF (~20.5 GB)"
    $HF_CMD download bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF --include "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf" --local-dir . || exit 1
    $HF_CMD download bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF --include "DeepSeek-R1-Distill-Llama-8B-f16.gguf" --local-dir . || exit 1
fi
echo "Done"
