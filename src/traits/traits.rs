use ndarray::{Dimension, Ix, Ix2};
use num::{Integer, Num, NumCast};
use std::fmt::{Debug, Display};
use std::str::FromStr;

/// The type of a value inside a tensor.
///
/// Integers, floats, `num::Complex` are all supported types.
/// Moreover, any type that satisfies this trait can be used.
///
/// ```
/// use pattie::traits::ValType;
///
/// fn my_func(_x: impl ValType) {
/// }
/// my_func(42.0);
/// ```
pub trait ValType: Num + Clone + Debug + Display + FromStr + Send + Sync {}
impl<T> ValType for T where T: Num + Clone + Debug + Display + FromStr + Send + Sync {}

/// The type of indices used in a tensor.
///
/// `u16`, `u32`, `u64`, usize are all supported types.
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
/// `VT` is the type of the values inside the tensor.
///
/// This trait itself does not describe how the tensor is stored, nor their sparsity, refer to `pattie::structs::tensor` for concrete types of tensors.
pub trait Tensor<VT>: Clone + Debug + Send + Sync
where
    VT: self::ValType,
{
    /// `Dim` describes the compile-time dimension of the tensor.
    ///
    /// If the dimension is known at compile-time, you can use `ndarray::Ix0`, `ndarray::Ix1`, etc.
    /// If the dimension is known only at runtime, you can use `ndarray::IxDyn`.
    ///
    /// Refer to `ndarray::Dim` for more information.
    ///
    /// # Example
    /// ```ignore
    /// use ndarray::Ix2;
    /// use pattie::traits::{Tensor, IdxType, ValType};
    ///
    /// impl<VT, IT> Tensor<VT> for SomeMatrix<VT, IT>
    /// where
    ///     VT: ValType,
    ///     IT: IdxType,
    /// {
    ///     type Dim = Ix2;
    /// }
    /// ```
    type Dim: Dimension;

    /// The pattern-match friendly version of the dimension.
    ///
    /// If the dimension is known at compile-time, you will get `()`, `len`, `(rows, cols)`, etc.
    /// If the dimension is known only at runtime, you will get `ndarray::IxDyn`.
    ///
    /// If you want to access other versions, you can call `raw_dim` or `shape`.
    fn dim(&self) -> <Self::Dim as Dimension>::Pattern {
        self.raw_dim().into_pattern()
    }

    /// The number of dimensions of the tensor.
    ///
    /// For example, 0 means scalar, 1 means vector, 2 means matrix, etc.
    fn ndim(&self) -> usize {
        self.raw_dim().ndim()
    }

    /// The number of elements in this tensor taking storage space.
    ///
    /// Note that this method also counts if elements are zero but taking storage space.
    /// This means that for dense tensors, `num_non_zeros` means the total number of elements.
    fn num_non_zeros(&self) -> usize;

    /// The dimension as `ndarray::Dim`.
    ///
    /// If the dimension is known at compile-time, you will get `ndarray::Ix0`, `ndarray::Ix1`, etc.
    /// If the dimension is known only at runtime, you will get `ndarray::IxDyn`.
    ///
    /// If you want to access other versions, you can call `dim` or `shape`.
    fn raw_dim(&self) -> Self::Dim;

    /// The dimension as `&[ndarray::Ix]`, aka `&[usize]`.
    ///
    /// No matter whether the dimension is known at compile-time or runtime, you will get `&[ndarray::Ix]`.
    ///
    /// If you want to access other versions, you can call `dim` or `raw_dim`.
    ///
    /// Note to implementors:
    ///
    /// This method is not provided by the trait itself due to lifetime restriction.
    /// You need to copy the implementation below into the `impl` block.
    /// ```ignore
    /// fn shape(&self) -> &[Ix] {
    ///     self.raw_dim().slice()
    /// }
    /// ```
    fn shape(&self) -> &[Ix];
}

/// Rust trait for a matrix.
///
/// `VT` is the type of the values inside the matrix.
///
/// All matrices are defined as tensor with 2 dimensions.
///
/// This trait itself does not describe how the matrix is stored, nor their sparsity, refer to `pattie::structs::matrix` for concrete types of tensors.
pub trait Matrix<VT>: Tensor<VT>
where
    VT: self::ValType,
    Self::Dim: Dimension<Pattern = <Ix2 as Dimension>::Pattern>,
{
}

/// All `Tensor<VT>` with 2 dimensions automatically implements the `Martix<VT>` trait.
impl<T, VT> Matrix<VT> for T
where
    Self: Tensor<VT>,
    VT: self::ValType,
    Self::Dim: Dimension<Pattern = <Ix2 as Dimension>::Pattern>,
{
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
