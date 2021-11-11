use crate::structs::axis::Axis;
use num::{Integer, Num, NumCast};
use std::fmt::{Debug, Display};

/// The type of a value inside a tensor.
///
/// Integers, floats, [`num::Complex`] are all supported types.
/// Moreover, any type that satisfies this trait can be used.
///
/// ```
/// use pattie::traits::ValType;
///
/// fn my_func(_x: impl ValType) {
/// }
/// my_func(42.0);
/// ```
pub trait ValType: Num + Clone + Debug + Display + Send + Sync {}
impl<T> ValType for T where T: Num + Clone + Debug + Display + Send + Sync {}

/// The type of indices used in a tensor.
///
/// [`u16`], [`u32`], [`u64`], [`usize`] are all supported types.
/// However, if the scale of the tensor is too large, narrow types may cause overflow and trigger panics.
///
/// ```
/// use pattie::traits::IdxType;
///
/// fn my_func(_x: impl IdxType) {
/// }
/// my_func(42);
/// ```
pub trait IdxType: Integer + NumCast + Clone + Debug + Display + Send + Sync {}
impl<T> IdxType for T where T: Integer + NumCast + Clone + Debug + Display + Send + Sync {}

/// Rust trait for a tensor.
///
/// `IT` is the type of indices used in the tensor.
/// `VT` is the type of the values inside the tensor.
///
/// This trait itself does not describe how the tensor is stored, nor their sparsity, refer to [`crate::structs::tensor`] for concrete types of tensors.
pub trait Tensor<IT, VT>: Clone + Debug + Send + Sync
where
    IT: self::IdxType,
    VT: self::ValType,
{
    /// The name of the tensor (optional).
    fn name(&self) -> Option<&str>;

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

/// An iterator over a tensor.
///
/// `'a` is the lifetime of the tensor from which the iterator is created.
/// `VT` is the type of the values inside the tensor.
/// `IT` is the type of the indices of the tensor.
///
/// Each element of the iterator is a tuple of the index and the value.
pub trait TensorIter<'a, VT, IT>: Iterator<Item = (&'a [IT], &'a VT)>
where
    IT: 'a,
    VT: 'a,
{
}

/// A mutable iterator over a tensor.
///
/// `'a` is the lifetime of the tensor from which the iterator is created.
/// `VT` is the type of the values inside the tensor.
/// `IT` is the type of the indices of the tensor.
///
/// A mutable iterator is used to modify each element of the tensor.
/// Each element of the iterator is a tuple of the index and the mutable value.
pub trait TensorIterMut<'a, VT, IT>: Iterator<Item = (&'a [IT], &'a mut VT)>
where
    IT: 'a,
    VT: 'a,
{
}

/// A moved iterator over a tensor.
///
/// `'a` is the lifetime of the tensor from which the iterator is created.
/// `VT` is the type of the values inside the tensor.
/// `IT` is the type of the indices of the tensor.
///
/// A moved iterator is used to consume each element of the tensor, meanwhile releasing the ownership of the tensor.
/// Each element of the iterator is a tuple of the index and the moved value.
pub trait TensorIntoIter<'a, VT, IT>: Iterator<Item = (&'a [IT], VT)>
where
    IT: 'a,
{
}
