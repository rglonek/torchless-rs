use ndarray::{Array, ArrayView, Ix1, Ix2, Ix3, Ix4};

pub mod storage;

// Re-export storage types
pub use storage::{
    Device, Dtype, TensorStorage, UnifiedTensor, DeviceTransfer,
    CpuF32Storage, CpuF16Storage, CpuBF16Storage, CpuInt8Storage, CpuInt4Storage,
    MixedPrecisionConfig, ModelSizeParams,
};

// Original ndarray type aliases for backward compatibility
pub type Tensor1 = Array<f32, Ix1>;
pub type Tensor2 = Array<f32, Ix2>;
pub type Tensor3 = Array<f32, Ix3>;
pub type Tensor4 = Array<f32, Ix4>;

pub type TensorView1<'a> = ArrayView<'a, f32, Ix1>;
pub type TensorView2<'a> = ArrayView<'a, f32, Ix2>;
pub type TensorView3<'a> = ArrayView<'a, f32, Ix3>;
pub type TensorView4<'a> = ArrayView<'a, f32, Ix4>;
