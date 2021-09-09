use crate::traits;
use ndarray::{Array1, Dimension, Ix2};

pub struct SparseMatrixCOO<ValType: traits::ValType, IdxType: traits::IdxType> {
    dim: Ix2,
    indices: Array1<[IdxType; 2]>,
    values: Array1<ValType>,
}

impl<ValType, IdxType> traits::Matrix for SparseMatrixCOO<ValType, IdxType>
where
    ValType: traits::ValType,
    IdxType: traits::IdxType,
{
    type Dim = Ix2;
}

impl<ValType, IdxType> traits::Tensor for SparseMatrixCOO<ValType, IdxType>
where
    ValType: traits::ValType,
    IdxType: traits::IdxType,
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
