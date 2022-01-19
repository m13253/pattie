use crate::structs::axis::Axis;
use crate::structs::tensor::{COOTensor, COOTensorInner};
use crate::structs::vec::SmallVec;
use crate::traits::{IdxType, RawParts, ValType};
use anyhow::Result;
use ndarray::{Array2, Array3};
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal};
use ndarray_rand::RandomExt;
use num::Float;

/// Create a random dense matrix.
///
/// The matrix is filled with random values drawn from a normal distribution.
pub struct CreateRandomDenseMatrix<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    pub shape: (Axis<IT>, Axis<IT>),
    pub mean: VT,
    pub std_dev: VT,
}

impl<IT, VT> CreateRandomDenseMatrix<IT, VT>
where
    IT: IdxType,
    VT: ValType + Float,
    StandardNormal: Distribution<VT>,
{
    /// Create a new `CreateRandomDenseMatrix` task.
    #[must_use]
    pub fn new(shape: (Axis<IT>, Axis<IT>), mean: VT, std_dev: VT) -> Self {
        Self {
            shape,
            mean,
            std_dev,
        }
    }

    /// Perform the generation.
    ///
    /// Returns `Err` if the random generator fails.
    pub fn execute(self) -> Result<COOTensor<IT, VT>> {
        let shape = [&self.shape.0, &self.shape.1];
        let shape_size = (1, self.shape.0.len(), self.shape.1.len());
        let matrix = Array3::random(shape_size, Normal::new(self.mean, self.std_dev)?);

        let raw_parts = COOTensorInner {
            name: None,
            shape: shape.into_iter().cloned().collect(),
            sparse_axes: SmallVec::new(),
            dense_axes: shape.into_iter().cloned().collect(),
            indices: Array2::zeros((1, 0)),
            values: matrix.into_dyn(),
            sparse_is_sorted: true,
            sparse_sort_order: SmallVec::new(),
        };
        Ok(
            // # Safety
            // We have checked the data integrity
            unsafe { COOTensor::from_raw_parts(raw_parts) },
        )
    }
}
