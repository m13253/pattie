use ndarray::{Dimension, Ix};

pub trait Tensor {
    type ValType: super::ValType;
    type Dim: Dimension;

    fn dim(&self) -> <Self::Dim as Dimension>::Pattern {
        self.raw_dim().into_pattern()
    }
    fn ndim(&self) -> usize;
    fn num_non_zeros(&self) -> usize;
    fn raw_dim(&self) -> Self::Dim;
    fn shape(&self) -> &[Ix];
}
