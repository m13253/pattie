use crate::structs::tensor::COOTensor;
use crate::traits::{IdxType, ValType};
use anyhow::Result;
use ndarray::Array2;
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal};
use ndarray_rand::RandomExt;
use num::Float;

pub fn create_random_dense_matrix<IT, VT>(
    rows: IT,
    cols: IT,
    mean: VT,
    std_dev: VT,
) -> Result<COOTensor<IT, VT>>
where
    IT: IdxType,
    VT: ValType + Float,
    StandardNormal: Distribution<VT>,
{
    let shape = (rows.to_usize().unwrap(), cols.to_usize().unwrap());
    let matrix = Array2::random(shape, Normal::new(mean, std_dev)?);
    Ok(COOTensor::from_ndarray(matrix))
}
