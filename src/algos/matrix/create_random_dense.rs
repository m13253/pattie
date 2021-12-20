use crate::structs::tensor::{COOTensor, COOTensorInner};
use crate::structs::vec::SmallVec;
use crate::traits::{IdxType, IntoAxis, RawParts, ValType};
use anyhow::Result;
use ndarray::{Array2, Array3};
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal};
use ndarray_rand::RandomExt;
use num::Float;

pub fn create_random_dense_matrix<IT, VT, Ax1, Ax2>(
    shape: (Ax1, Ax2),
    mean: VT,
    std_dev: VT,
) -> Result<COOTensor<IT, VT>>
where
    IT: IdxType,
    VT: ValType + Float,
    StandardNormal: Distribution<VT>,
    Ax1: IntoAxis<IT>,
    Ax2: IntoAxis<IT>,
{
    let shape = [shape.0.into_axis(), shape.1.into_axis()];
    let shape_size = (1, shape[0].size(), shape[1].size());
    let matrix = Array3::random(shape_size, Normal::new(mean, std_dev)?);

    let raw_parts = COOTensorInner {
        name: None,
        shape: shape.iter().cloned().collect(),
        sparse_axes: SmallVec::new(),
        dense_axes: shape.iter().cloned().collect(),
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
