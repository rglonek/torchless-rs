# Library API

Using torchless-rs as a Rust library.

## Basic Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
torchless = { path = "../torchless-rs" }
# Or when published: torchless = "0.1"
```

## Model Loading

### Eager Mode (Fast Inference, ~25GB RAM)

Loads all tensors into memory for fastest inference:

```rust
use torchless::{Mistral, InferenceState, Parameters, generate};

fn main() -> anyhow::Result<()> {
    // Load model (copies all weights to RAM)
    let params = Parameters::load("mistral.bin")?;
    let model = Mistral::load(params)?;
    
    // Create inference state
    let mut state = InferenceState::new(model.config.clone());
    
    // Run inference
    let token: u32 = 1; // BOS token
    model.fast_forward(&mut state, token, false);
    
    // Generate next token
    let next_token = generate(&model, &mut state, token, 0.7, false);
    println!("Next token: {}", next_token);
    
    Ok(())
}
```

### Lazy Mode (Memory-Efficient, <2GB RAM)

Keeps weights memory-mapped, loads on demand:

```rust
use torchless::{LazyMistral, InferenceState, Parameters, generate_lazy};

fn main() -> anyhow::Result<()> {
    // Load parameters (memory-mapped, not copied)
    let params = Parameters::load("mistral.bin")?;
    
    // Create lazy model (borrows params)
    let model = LazyMistral::load(&params)?;
    
    // Create inference state
    let mut state = InferenceState::new(model.config.clone());
    
    // Run inference (tensors loaded from mmap on-demand)
    let token: u32 = 1;
    model.forward(&mut state, token, false);
    
    // Generate next token
    let next_token = generate_lazy(&model, &mut state, token, 0.7, false);
    println!("Next token: {}", next_token);
    
    Ok(())
}
```

### Dynamic Model Loading (Auto-Detect Architecture)

```rust
use torchless::{DynamicModel, InferenceState, Parameters};

let params = Parameters::load("model.bin")?;
let model = DynamicModel::load_auto(params)?;

println!("Detected architecture: {}", model.architecture());

let mut state = InferenceState::new(model.config().clone());
model.forward(&mut state, token, false);
```

## Tokenization

```rust
use torchless::Parameters;

let params = Parameters::load("mistral.bin")?;
let tokenizer = &params.tokenizer;

// Encode text to tokens
let tokens = tokenizer.encode("Hello, world!");
println!("Tokens: {:?}", tokens);

// Decode tokens to text
let text = tokenizer.decode(&tokens);
println!("Text: {}", text);
```

## Generation Loop

```rust
use torchless::{Mistral, InferenceState, Parameters, generate};

let params = Parameters::load("mistral.bin")?;
let model = Mistral::load(params)?;
let mut state = InferenceState::new(model.config.clone());

let prompt = "The capital of France is";
let tokens = model.tokenizer.encode(prompt);

// Process prompt tokens
for &token in &tokens {
    model.fast_forward(&mut state, token, false);
    state.pos += 1;
}

// Generate new tokens
let temperature = 0.7;
let max_tokens = 50;

for _ in 0..max_tokens {
    let next_token = generate(&model, &mut state, tokens.last().copied().unwrap_or(1), temperature, false);
    
    // Decode and print
    let text = model.tokenizer.decode(&[next_token]);
    print!("{}", text);
    
    // Advance state
    model.fast_forward(&mut state, next_token, false);
    state.pos += 1;
    
    // Stop on EOS
    if next_token == 2 { break; }
}
```

## Direct Tensor Access

For custom operations, access tensors directly via `TensorView`:

```rust
use torchless::{Parameters, TensorView, TensorDtype};

let params = Parameters::load("model.bin")?;

// Get lazy tensor view (no copy)
let view: TensorView = params.get_tensor_view("model.layers.0.self_attn.q_proj.weight")?;

// Inspect tensor
println!("Shape: {:?}", view.shape);
println!("Dtype: {:?}", view.dtype); // F32, F16, Int8, Int4, etc.
println!("Rows: {}, Cols: {}", view.nrows(), view.ncols());

// Access single row (dequantizes on-demand)
let row: Vec<f32> = view.get_row(0);

// Matrix-vector multiplication (fused dequant + matmul)
let input: Vec<f32> = vec![0.0; view.ncols()];
let output: Vec<f32> = view.matmul_vec(&input);

// Pre-allocated output
let mut output = vec![0.0; view.nrows()];
view.matmul_vec_into(&input, &mut output);
```

## Optimized Kernels

When built with feature flags, use optimized kernel APIs:

```rust
use torchless::kernels::{fast_rmsnorm, fast_softmax, fast_silu, fast_matmul_vec};

// These auto-select SIMD/parallel implementations when available
fast_rmsnorm(&mut hidden_state, &weight, 1e-5);
fast_softmax(&mut logits);
let activated = fast_silu(&gate_output);
let output = fast_matmul_vec(&weights, &input);
```

## Backend Selection

```rust
use torchless::{init_backend, BackendPreference, discover_backends, print_backend_summary};

// Print available backends
print_backend_summary();

// Auto-select best available
let backend = init_backend(BackendPreference::Auto)?;
println!("Using: {}", backend.name());

// Or explicitly request a backend
let backend = init_backend(BackendPreference::Metal)?;

// Use backend for operations
let result = backend.matmul_vec(&weights, &input);
```

## Quantization

```rust
use torchless::{QuantizedTensor, QuantFormat, MixedPrecisionConfig};

// Quantize data
let data: Vec<f32> = vec![...];
let q4 = QuantizedTensor::from_f32(&data, vec![1024, 4096], QuantFormat::Q4_0);

println!("Compression: {}x", q4.compression_ratio());

// Mixed precision configuration
let config = MixedPrecisionConfig::balanced(); // INT8 weights, FP16 activations
```

## Flash Attention

```rust
use torchless::{FlashAttentionConfig, flash_attention_multi_head};

let config = FlashAttentionConfig::default();

let output = flash_attention_multi_head(
    &q_state,
    k_cache.view(),
    v_cache.view(),
    n_heads,
    n_kv_heads,
    seq_len,
    &config,
);
```

## Speculative Decoding

```rust
use torchless::{SpeculativeDecoder, SpeculativeConfig};

let config = SpeculativeConfig::default();

let mut decoder = SpeculativeDecoder::new(
    |state, token| main_model.forward(state, token, false),
    |state, token| draft_model.forward(state, token, false),
    config,
);

let accepted_tokens = decoder.generate_step(&mut main_state, &mut draft_state, context_token);
println!("Acceptance rate: {:.1}%", decoder.stats().acceptance_rate() * 100.0);
```

## Configuration Access

```rust
let params = Parameters::load("mistral.bin")?;
let config = &params.config;

println!("Hidden size: {}", config.hidden_size);
println!("Layers: {}", config.n_layers);
println!("Heads: {}", config.n_heads);
println!("KV heads: {}", config.n_kv_heads);
println!("Vocab size: {}", config.vocab_size);
```
