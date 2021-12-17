use super::{IdxType, ValType};
use crate::structs::axis::Axis;
use std::fmt::Debug;

/// Rust trait for a tensor.
///
/// `IT` is the type of indices used in the tensor.
/// `VT` is the type of the values inside the tensor.
///
/// This trait itself does not describe how the tensor is stored, nor their sparsity, refer to [`crate::structs::tensor`] for concrete types of tensors.
pub trait Tensor<IT, VT>: Clone + Debug + Send + Sync
where
    IT: IdxType,
    VT: ValType,
{
    /// The name of the tensor (optional).
    fn name(&self) -> Option<&str>;

    /// Change the name of the tensor.
    fn name_mut(&mut self) -> &mut Option<String>;

    /// The number of dimensions of the tensor.
    ///
    /// For example, 0 means scalar, 1 means vector, 2 means matrix, etc.
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// The number of elements in this tensor taking storage space.
    ///
    /// Note that this method also counts if elements are zero but taking storage space.
    /// This means that for dense tensors, `num_non_zeros` means the total number of elements.
    fn num_non_zeros(&self) -> usize;

    /// The dimensions of the tensor.
    ///
    /// Each axis is an [`Axis`] object.
    /// To access the lower and upper bounds of the axis, use [`Axis::lower_bound`] and [`Axis::upper_bound`].
    fn shape(&self) -> &[Axis<IT>];
}
