#!/usr/bin/env python3
"""
Generate a tiny test model binary in the same format as export_mistral.py
This allows testing without needing a 27GB Mistral model.
"""
import json
import struct
import numpy as np

# Tiny model config for testing
CONFIG = {
    "hidden_size": "32",
    "intermediate_size": "64",
    "n_layers": "2",
    "n_heads": "4",
    "n_kv_heads": "2",
    "vocab_size": "300",  # Need at least 266 for byte fallback tokens (10 + 256)
    "max_position_embeddings": "128",
    "sliding_window": "32",
    "rope_theta": "10000.0",
    "norm_eps": "1e-5",
    "act_type": "silu",
    "quant": "f32"
}

# Minimal vocab for testing
VOCAB = {
    "<s>": 0,
    "▁": 1,
    "Hello": 2,
    "World": 3,
    "Paris": 4,
    "is": 5,
    "the": 6,
    "capital": 7,
    "of": 8,
    "France": 9,
}

# Add byte fallback tokens
for i in range(256):
    VOCAB[f"<0x{i:02X}>"] = 10 + i

# Minimal merges
MERGES = [
    "▁ H",
    "H e",
    "l l",
    "o o",
]

def pad_to_64(offset):
    """Pad offset to 64-byte alignment"""
    r = offset % 64
    if r == 0:
        return 0
    return 64 - r

def create_tensor_info(shape, dtype="f32"):
    """Create tensor metadata"""
    return {
        "dtype": dtype,
        "shape": shape,
        "offset": 0,  # Will be filled in later
    }

def generate_test_model(output_path):
    """Generate a tiny test model binary"""

    hidden_size = int(CONFIG["hidden_size"])
    intermediate_size = int(CONFIG["intermediate_size"])
    n_layers = int(CONFIG["n_layers"])
    vocab_size = int(CONFIG["vocab_size"])

    # Build header
    header = {
        "metadata": CONFIG,
        "tokenizer": {
            "vocab": VOCAB,
            "merges": MERGES,
        },
        "tensors": {}
    }

    # Define all tensors with random weights
    tensors = {}

    # Global tensors
    tensors["model.embed_tokens.weight"] = np.random.randn(vocab_size, hidden_size).astype(np.float32)
    tensors["model.norm.weight"] = np.random.randn(hidden_size).astype(np.float32)
    tensors["lm_head.weight"] = np.random.randn(vocab_size, hidden_size).astype(np.float32)

    # Layer tensors
    for i in range(n_layers):
        prefix = f"model.layers.{i}"

        # Norms
        tensors[f"{prefix}.input_layernorm.weight"] = np.random.randn(hidden_size).astype(np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.random.randn(hidden_size).astype(np.float32)

        # Attention projections
        tensors[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        tensors[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(hidden_size // 2, hidden_size).astype(np.float32)
        tensors[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(hidden_size // 2, hidden_size).astype(np.float32)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(hidden_size, hidden_size).astype(np.float32)

        # MLP projections
        tensors[f"{prefix}.mlp.gate_proj.weight"] = np.random.randn(intermediate_size, hidden_size).astype(np.float32)
        tensors[f"{prefix}.mlp.up_proj.weight"] = np.random.randn(intermediate_size, hidden_size).astype(np.float32)
        tensors[f"{prefix}.mlp.down_proj.weight"] = np.random.randn(hidden_size, intermediate_size).astype(np.float32)

    # Calculate offsets
    offset = 0
    for name, tensor in tensors.items():
        header["tensors"][name] = {
            "dtype": "f32",
            "shape": list(tensor.shape),
            "offset": offset
        }
        offset += tensor.nbytes
        offset += pad_to_64(offset)

    # Write binary file
    with open(output_path, "wb") as f:
        # Serialize header
        header_bytes = json.dumps(header).encode("utf-8")
        padding_size = pad_to_64(8 + len(header_bytes))

        # Write header size
        header_size = len(header_bytes) + padding_size
        f.write(struct.pack("<Q", header_size))

        # Write header
        f.write(header_bytes)

        # Write padding
        f.write(b'\x00' * padding_size)

        base_offset = f.tell()

        # Write tensors
        for name in header["tensors"]:
            tensor_offset = header["tensors"][name]["offset"]
            f.seek(base_offset + tensor_offset, 0)
            f.write(tensors[name].tobytes())

        file_size = f.tell()

    print(f"✓ Generated test model: {output_path}")
    print(f"  Config: {hidden_size}d hidden, {n_layers} layers, {vocab_size} vocab")
    print(f"  Size: {file_size} bytes ({file_size / 1024:.1f} KB)")

if __name__ == "__main__":
    import os

    # Create fixtures directory
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    output_path = os.path.join(fixtures_dir, "test_model.bin")
    generate_test_model(output_path)
