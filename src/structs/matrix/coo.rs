use crate::traits;
use ndarray::{Array1, Dimension, Ix, Ix2};

#[derive(Clone, Debug)]
pub struct COOMatrix<VT, IT>
where
    VT: traits::ValType,
    IT: traits::IdxType,
{
    dim: Ix2,
    indices: Array1<[IT; 2]>,
    values: Array1<VT>,
}

impl<VT, IT> traits::Tensor<VT> for COOMatrix<VT, IT>
where
    VT: traits::ValType,
    IT: traits::IdxType,
{
    type Dim = Ix2;

    fn ndim(&self) -> usize {
        2
    }

    fn num_non_zeros(&self) -> usize {
        debug_assert_eq!(self.indices.len(), self.values.len());
        self.values.len()
    }

    fn raw_dim(&self) -> Self::Dim {
        self.dim.clone()
    }

    fn shape(&self) -> &[Ix] {
        &self.dim.slice()
    }
}
