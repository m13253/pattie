//! Rust traits for scalars, matrices, tensors, iterators, etc.

mod idxtype;
mod tensor;
mod tensor_iter;
mod valtype;

pub use idxtype::IdxType;
pub use tensor::Tensor;
pub use tensor_iter::{TensorIntoIter, TensorIter, TensorIterMut};
pub use valtype::ValType;
