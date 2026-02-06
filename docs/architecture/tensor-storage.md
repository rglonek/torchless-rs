# Tensor Storage

Unified tensor interface with multi-dtype support.

## Data Types

Support for multiple precision formats:

```rust
pub enum Dtype {
    F32,   // 32-bit float (4 bytes)
    F16,   // 16-bit float (2 bytes) - IEEE 754
    BF16,  // 16-bit brain float (2 bytes)
    Int8,  // 8-bit quantized (1 byte + scales)
    Int4,  // 4-bit quantized (0.5 bytes + scales)
}
```

### Memory Savings

| Dtype | Bytes/Element | 7B Model Size |
|-------|---------------|---------------|
| F32   | 4.0           | ~28 GB        |
| F16   | 2.0           | ~14 GB        |
| Int8  | ~1.1          | ~8 GB         |
| Int4  | ~0.6          | ~4 GB         |

## Device Types

```rust
pub enum Device {
    Cpu,
    Cuda(usize),   // With device index
    Metal,
    OpenCL(usize), // With device index
}
```

## CPU Storage Types

Four storage implementations for CPU:

```rust
pub struct CpuF32Storage { data: Vec<f32> }
pub struct CpuF16Storage { data: Vec<f16> }
pub struct CpuInt8Storage { data: Vec<i8>, scales: Vec<f32>, group_size: usize }
pub struct CpuInt4Storage { data: Vec<u8>, scales: Vec<f32>, group_size: usize }
```

## Unified Tensor Storage

```rust
pub enum TensorStorage {
    CpuF32(CpuF32Storage),
    CpuF16(CpuF16Storage),
    CpuInt8(CpuInt8Storage),
    CpuInt4(CpuInt4Storage),
    // GPU variants added by features
}
```

## UnifiedTensor

High-level tensor type with shape tracking:

```rust
pub struct UnifiedTensor {
    pub storage: TensorStorage,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}
```

### Creation Methods

```rust
// From existing data
UnifiedTensor::from_f32(data, shape)
UnifiedTensor::from_f32_as_f16(data, shape)

// With quantization
UnifiedTensor::from_f32_quantize_int8(data, shape, group_size)
UnifiedTensor::from_f32_quantize_int4(data, shape, group_size)
```

### Conversion to ndarray

```rust
tensor.to_array1()
tensor.to_array2()
tensor.to_array3()
tensor.to_array4()
```

## Device Transfer Trait

Prepared for GPU support:

```rust
pub trait DeviceTransfer {
    fn to_device(&self, device: Device) -> anyhow::Result<UnifiedTensor>;
    fn to_cpu(&self) -> anyhow::Result<UnifiedTensor>;
}
```

## TensorView API

Lazy tensor access for memory-efficient loading:

```rust
use torchless::{Parameters, TensorView, TensorDtype};

let params = Parameters::load("model.bin")?;

// Get lazy view (no copy)
let view: TensorView = params.get_tensor_view("layer.0.weight")?;

// Check dtype
match view.dtype {
    TensorDtype::F32 => println!("Full precision"),
    TensorDtype::F16 => println!("Half precision"),
    TensorDtype::Int8 => println!("8-bit quantized"),
    TensorDtype::Int4 => println!("4-bit quantized"),
    _ => {}
}

// Lazy operations (dequantize on-demand)
let row = view.get_row(0);
let output = view.matmul_vec(&input);
```

## Memory Layout

All tensors use row-major layout with optional alignment:

```rust
#[repr(C, align(64))]
struct AlignedStorage {
    data: Vec<f32>,
}
```

### Cache Line Alignment

- `CACHE_LINE_SIZE = 64` bytes
- `SIMD_ALIGNMENT = 64` bytes (AVX-512)
- `CACHE_LINE_F32S = 16` floats per cache line

## Quantization Storage

Quantized tensors store weights and scales separately:

```rust
// INT8: 64 weights share 1 scale
pub struct CpuInt8Storage {
    data: Vec<i8>,      // Quantized weights
    scales: Vec<f32>,   // One scale per group
    group_size: usize,  // Typically 64
}

// INT4: 32 weights share 1 scale (packed in nibbles)
pub struct CpuInt4Storage {
    data: Vec<u8>,      // Packed 4-bit values (2 per byte)
    scales: Vec<f32>,   // One scale per group
    group_size: usize,  // Typically 32
}
```

Dequantization:
- INT8: `weight_f32 = weight_int8 / scale`
- INT4: `weight_f32 = (nibble - 8) * scale`

## GPU Tensor Types

Each GPU backend provides its own tensor type:

```rust
#[cfg(feature = "cuda")]
pub struct CudaTensor {
    data: CudaSlice<f32>,
    shape: Vec<usize>,
}

#[cfg(feature = "metal")]
pub struct MetalTensor {
    buffer: metal::Buffer,
    shape: Vec<usize>,
}
```

GPU tensors integrate with the unified `TensorStorage` enum through the appropriate feature flags.
