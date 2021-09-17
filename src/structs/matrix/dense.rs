use crate::traits;
use ndarray::{Array2, Dimension, Ix2, ShapeBuilder};
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub struct Dense<VT>
where
    VT: traits::ValType,
{
    data: Array2<VT>,
}

impl<VT> Dense<VT>
where
    VT: traits::ValType,
{
    pub fn zeros<Sh>(shape: Sh) -> Dense<VT>
    where
        Sh: ShapeBuilder<Dim = Ix2>,
    {
        Dense {
            data: Array2::zeros(shape),
        }
    }
}

impl<VT> traits::Tensor<VT> for Dense<VT>
where
    VT: traits::ValType,
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
impl<VT> traits::Matrix<VT> for Dense<VT> where VT: traits::ValType {}

impl<VT> Deref for Dense<VT>
where
    VT: traits::ValType,
{
    type Target = Array2<VT>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}
impl<VT> DerefMut for Dense<VT>
where
    VT: traits::ValType,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<VT> From<Array2<VT>> for Dense<VT>
where
    VT: traits::ValType,
{
    fn from(a: Array2<VT>) -> Self {
        Dense { data: a }
    }
}
