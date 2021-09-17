use crate::traits::{Tensor, ValType};
use ndarray::{Array2, Dimension, Ix2, ShapeBuilder};
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub struct DenseMatrix<VT>
where
    VT: ValType,
{
    data: Array2<VT>,
}

impl<VT> DenseMatrix<VT>
where
    VT: ValType,
{
    pub fn zeros<Sh>(shape: Sh) -> DenseMatrix<VT>
    where
        Sh: ShapeBuilder<Dim = Ix2>,
    {
        DenseMatrix {
            data: Array2::zeros(shape),
        }
    }
}

impl<VT> Tensor<VT> for DenseMatrix<VT>
where
    VT: ValType,
{
    type Dim = Ix2;

    fn ndim(&self) -> usize {
        2
    }

    fn num_non_zeros(&self) -> usize {
        self.data.raw_dim().size_checked().unwrap()
    }

    fn raw_dim(&self) -> Self::Dim {
        self.data.raw_dim()
    }

    fn shape(&self) -> &[usize] {
        self.data.shape()
    }
}

impl<VT> Deref for DenseMatrix<VT>
where
    VT: ValType,
{
    type Target = Array2<VT>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
impl<VT> DerefMut for DenseMatrix<VT>
where
    VT: ValType,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<VT> From<Array2<VT>> for DenseMatrix<VT>
where
    VT: ValType,
{
    fn from(a: Array2<VT>) -> Self {
        DenseMatrix { data: a }
    }
}
