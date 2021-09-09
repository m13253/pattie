use super::Tensor;
use ndarray::{Dimension, Ix};

pub trait Matrix<'a>: Tensor<'a> {
    type Dim: Dimension<Pattern = (Ix, Ix)>;
}
