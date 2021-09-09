use crate::traits;
use ndarray::{Array2, Ix, Ix2};

type DenseMatrix<ValType> = Array2<ValType>;

impl<ValType> traits::Matrix for DenseMatrix<ValType>
where
    ValType: traits::ValType,
{
    type Dim = Ix2;
}

impl<ValType> traits::Tensor for DenseMatrix<ValType>
where
    ValType: traits::ValType,
{
    type ValType = ValType;
    type Dim = Ix2;

    fn ndim(&self) -> usize {
        self.ndim()
    }

    fn num_non_zeros(&self) -> usize {
        let (x, y) = self.dim();
        x * y
    }

    fn raw_dim(&self) -> Self::Dim {
        self.raw_dim()
    }

    fn shape(&self) -> &[Ix] {
        self.shape()
    }
}
