use super::Tensor;
use ndarray::{Dimension, Ix};

pub trait Matrix: Tensor {
    type Dim: Dimension<Pattern = (Ix, Ix)>;
}
