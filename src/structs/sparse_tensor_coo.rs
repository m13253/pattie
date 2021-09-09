use crate::traits;
use ndarray::{Array1, Array2, Dimension, IxDyn};

pub struct SparseTensorCOO<ValType: traits::ValType, IdxType: traits::IdxType> {
    dim: IxDyn,
    sort_order: Array1<usize>,
    indices: Array2<IdxType>,
    values: Array1<ValType>,
}

impl<'a, ValType, IdxType> traits::Tensor<'a> for SparseTensorCOO<ValType, IdxType>
where
    ValType: 'a + traits::ValType,
    IdxType: 'a + traits::IdxType,
{
    type ValType = ValType;
    type Dim = IxDyn;

    fn ndim(&self) -> usize {
        self.dim.size()
    }

    fn num_non_zeros(&self) -> usize {
        self.values.len()
    }

    fn raw_dim(&self) -> Self::Dim {
        self.dim.clone()
    }

    fn shape(&self) -> &[ndarray::Ix] {
        self.dim.slice()
    }
}
