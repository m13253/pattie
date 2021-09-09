use crate::traits;
use ndarray::{Array1, Dimension, Ix2};

pub struct SparseMatrixCOO<ValType: traits::ValType, IdxType: traits::IdxType> {
    dim: Ix2,
    indices: Array1<[IdxType; 2]>,
    values: Array1<ValType>,
}

impl<'a, ValType, IdxType> traits::Matrix<'a> for SparseMatrixCOO<ValType, IdxType>
where
    ValType: 'a + traits::ValType,
    IdxType: 'a + traits::IdxType,
{
    type Dim = Ix2;
}

impl<'a, ValType, IdxType> traits::Tensor<'a> for SparseMatrixCOO<ValType, IdxType>
where
    ValType: 'a + traits::ValType,
    IdxType: 'a + traits::IdxType,
{
    type Dim = Ix2;
    type ValType = ValType;

    fn ndim(&self) -> usize {
        2
    }

    fn num_non_zeros(&self) -> usize {
        self.values.len()
    }

    fn raw_dim(&self) -> Self::Dim {
        self.dim
    }

    fn shape(&self) -> &[ndarray::Ix] {
        self.dim.slice()
    }
}
