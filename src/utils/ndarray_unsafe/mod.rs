//! Access [`ndarray::Array`] faster by skipping safety checks.

pub mod ix1;
pub mod ix2;

use ndarray::{ArrayBase, Dimension, RawData, RawDataMut};

/// Write an [`ndarray::Array`] to an [`UncheckedArray`], so you can use the fast methods to unsafely access its contents.
///
/// # Example
/// ```
/// use ndarray::array;
/// use pattie::utils::ndarray_unsafe::*;
///
/// let mut array = array![1, 2, 3, 4];
/// array[1] = 42;
/// let value = unsafe { uncheck_arr(&mut array).get(1) };
/// assert_eq!(*value, 42);
/// ```
#[inline]
pub fn uncheck_arr<S, D>(array: &ArrayBase<S, D>) -> UncheckedArray<S, D>
where
    S: RawData,
    D: Dimension,
{
    UncheckedArray { array }
}

/// Write an [`ndarray::Array`] to an [`UncheckedArrayMut`], so you can use the fast methods to unsafely access its contents.
///
/// # Example
/// ```
/// use ndarray::array;
/// use pattie::utils::ndarray_unsafe::*;
///
/// let mut array = array![1, 2, 3, 4];
/// *unsafe { uncheck_arr_mut(&mut array).get(1) } = 42;
/// assert_eq!(array[1], 42);
/// ```
#[inline]
pub fn uncheck_arr_mut<S, D>(array: &mut ArrayBase<S, D>) -> UncheckedArrayMut<S, D>
where
    S: RawDataMut,
    D: Dimension,
{
    UncheckedArrayMut { array }
}

/// A wrapper to [`ndarray::Array`] so you can use the fast methods to unsafely access its contents.
#[repr(transparent)]
pub struct UncheckedArray<'a, S, D>
where
    S: RawData,
    D: Dimension,
{
    pub array: &'a ArrayBase<S, D>,
}

/// A wrapper to [`ndarray::Array`] so you can use the fast methods to unsafely access its contents.
#[repr(transparent)]
pub struct UncheckedArrayMut<'a, S, D>
where
    S: RawDataMut,
    D: Dimension,
{
    pub array: &'a mut ArrayBase<S, D>,
}
