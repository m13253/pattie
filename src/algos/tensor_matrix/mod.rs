//! Algorithms related to tensors and matrices.

mod coo_mul_dense;
mod scoo_mul_dense;

pub use coo_mul_dense::COOTensorMulDenseMatrix;
pub use scoo_mul_dense::SemiCOOTensorMulDenseMatrix;
