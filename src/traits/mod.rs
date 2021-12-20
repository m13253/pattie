//! Rust traits for scalars, matrices, tensors, iterators, etc.

mod axis;
mod idxtype;
mod raw_parts;
mod tensor;
mod tensor_iter;
mod valtype;

pub use axis::IntoAxis;
pub use idxtype::IdxType;
pub use raw_parts::RawParts;
pub use tensor::Tensor;
pub use tensor_iter::{TensorIntoIter, TensorIter, TensorIterMut};
pub use valtype::ValType;
