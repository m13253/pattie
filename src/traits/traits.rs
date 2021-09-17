use ndarray::{Dimension, Ix, Ix2};
use num::{Integer, Num, NumCast};
use std::fmt::{Debug, Display};
use std::str::FromStr;

pub trait ValType: Num + Clone + Debug + Display + FromStr + Send + Sync {}
impl<T> ValType for T where T: Num + Clone + Debug + Display + FromStr + Send + Sync {}

pub trait IdxType: Integer + NumCast + Clone + Debug + Display + Send + Sync {}
impl<T> IdxType for T where T: Integer + NumCast + Clone + Debug + Display + Send + Sync {}

pub trait Tensor<VT>: Clone + Debug + Send + Sync
where
    VT: self::ValType,
{
    type Dim: Dimension;

    fn dim(&self) -> <Self::Dim as Dimension>::Pattern {
        self.raw_dim().into_pattern()
    }
    fn ndim(&self) -> usize;
    fn num_non_zeros(&self) -> usize;
    fn raw_dim(&self) -> Self::Dim;
    fn shape(&self) -> &[Ix];
    // Unable to provide default implementation due to lifetime restriction
    // fn shape(&self) -> &[Ix] {
    //     self.raw_dim().slice()
    // }
}

pub trait Matrix<VT>: Tensor<VT>
where
    VT: self::ValType,
    Self::Dim: Dimension<Pattern = <Ix2 as Dimension>::Pattern>,
{
}

impl<T, VT> Matrix<VT> for T
where
    Self: Tensor<VT>,
    VT: self::ValType,
    Self::Dim: Dimension<Pattern = <Ix2 as Dimension>::Pattern>,
{
}

pub trait TensorIter<'a, VT, IT>: Iterator<Item = (&'a [IT], &'a VT)>
where
    IT: 'a,
    VT: 'a,
{
}

pub trait TensorIterMut<'a, VT, IT>: Iterator<Item = (&'a [IT], &'a mut VT)>
where
    IT: 'a,
    VT: 'a,
{
}

pub trait TensorIntoIter<'a, VT, IT>: Iterator<Item = (&'a [IT], VT)>
where
    IT: 'a,
{
}
