use crate::traits;
use ndarray::{iter::IndexedIter, Array2, Ix, Ix2};

type DenseMatrix<ValType> = Array2<ValType>;

impl<'a, ValType> traits::Matrix<'a> for DenseMatrix<ValType>
where
    ValType: 'a + traits::ValType,
{
    type Dim = Ix2;
}

impl<'a, ValType> traits::Tensor<'a> for DenseMatrix<ValType>
where
    ValType: 'a + traits::ValType,
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
