//! Weight matrix storage in native format with on-the-fly dequantization during matmul.
//!
//! This module provides the `WeightMatrix` type that stores model weights in their native
//! format (FP16, Q8_0, Q4_K_M, etc.) and dequantizes on-the-fly during operations.

use ndarray::Array2;
use std::fmt;

use super::parameters::{TensorDtype, TensorView};

/// Weight matrix that stores data in native format for memory efficiency.
/// Dequantizes on-the-fly during matmul operations by delegating to TensorView.
pub enum WeightMatrix {
    /// Native format: raw bytes kept as-is, dequantized on-the-fly during matmul
    Native {
        data: Vec<u8>,
        shape: [usize; 2],
        dtype: TensorDtype,
        scales: Option<Vec<u8>>,
        group_size: usize,
    },
    /// Pre-dequantized f32 (fallback for fused/stacked tensors)
    F32(Array2<f32>),
}

impl WeightMatrix {
    /// Returns the shape as [nrows, ncols]
    pub fn shape(&self) -> [usize; 2] {
        match self {
            WeightMatrix::Native { shape, .. } => *shape,
            WeightMatrix::F32(arr) => {
                let (r, c) = arr.dim();
                [r, c]
            }
        }
    }

    /// Number of rows (shape[0])
    pub fn nrows(&self) -> usize {
        self.shape()[0]
    }

    /// Number of columns (shape[1])
    pub fn ncols(&self) -> usize {
        self.shape()[1]
    }

    /// Creates a TensorView borrowing from self (Native variant only).
    /// Panics if called on F32 variant.
    fn as_view(&self) -> TensorView<'_> {
        match self {
            WeightMatrix::Native {
                data,
                shape,
                dtype,
                scales,
                group_size,
            } => TensorView {
                data: data.as_slice(),
                shape: vec![shape[0], shape[1]],
                dtype: *dtype,
                scales: scales.as_deref(),
                group_size: *group_size,
            },
            WeightMatrix::F32(_) => panic!("as_view not supported for F32 variant"),
        }
    }

    /// Matrix-vector multiplication: result = self @ x
    pub fn matmul_vec(&self, x: &[f32]) -> Vec<f32> {
        match self {
            WeightMatrix::Native { .. } => self.as_view().matmul_vec(x),
            WeightMatrix::F32(arr) => {
                let x_arr = ndarray::ArrayView1::from(x);
                let result = arr.dot(&x_arr);
                result.to_vec()
            }
        }
    }

    /// Matrix-vector multiplication into pre-allocated buffer: out = self @ x
    pub fn matmul_vec_into(&self, x: &[f32], out: &mut [f32]) {
        match self {
            WeightMatrix::Native { .. } => self.as_view().matmul_vec_into(x, out),
            WeightMatrix::F32(arr) => {
                let x_arr = ndarray::ArrayView1::from(x);
                let result = arr.dot(&x_arr);
                out.copy_from_slice(result.as_slice().unwrap());
            }
        }
    }

    /// Get a single row as f32 (used by Embedding)
    pub fn get_row(&self, row: usize) -> Vec<f32> {
        match self {
            WeightMatrix::Native { .. } => self.as_view().get_row(row),
            WeightMatrix::F32(arr) => arr.row(row).iter().cloned().collect(),
        }
    }

    /// Convenience constructor for the F32 variant
    pub fn from_f32(data: Array2<f32>) -> Self {
        WeightMatrix::F32(data)
    }
}

impl fmt::Debug for WeightMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightMatrix::Native { shape, dtype, .. } => write!(
                f,
                "WeightMatrix::Native {{ shape: {:?}, dtype: {:?} }}",
                shape, dtype
            ),
            WeightMatrix::F32(arr) => {
                let (r, c) = arr.dim();
                write!(f, "WeightMatrix::F32 {{ shape: [{}, {}] }}", r, c)
            }
        }
    }
}

impl Clone for WeightMatrix {
    fn clone(&self) -> Self {
        match self {
            WeightMatrix::Native {
                data,
                shape,
                dtype,
                scales,
                group_size,
            } => WeightMatrix::Native {
                data: data.clone(),
                shape: *shape,
                dtype: *dtype,
                scales: scales.clone(),
                group_size: *group_size,
            },
            WeightMatrix::F32(arr) => WeightMatrix::F32(arr.clone()),
        }
    }
}
