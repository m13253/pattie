use crate::structs::axis::Axis;
use crate::structs::tensor::{COOTensor, COOTensorInner};
use crate::structs::vec::SmallVec;
use crate::traits::{IdxType, RawParts, ValType};
use anyhow::Result;
use ndarray::{Array2, Array3};
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal};
use ndarray_rand::RandomExt;
use num::Float;

pub fn create_random_dense_matrix<IT, VT>(
    shape: (Axis<IT>, Axis<IT>),
    mean: VT,
    std_dev: VT,
) -> Result<COOTensor<IT, VT>>
where
    IT: IdxType,
    VT: ValType + Float,
    StandardNormal: Distribution<VT>,
{
    let shape = [shape.0, shape.1];
    let shape_size = (1, shape[0].len(), shape[1].len());
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
