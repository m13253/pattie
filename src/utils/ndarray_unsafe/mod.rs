//! Access [`ndarray::Array`] faster by skipping safety checks.

pub mod ix1;
pub mod ix2;

use ndarray::{ArrayBase, Dimension, RawData, RawDataMut};

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
pub fn uncheck_arr<'a, S, D>(array: &'a ArrayBase<S, D>) -> UncheckedArray<'a, S, D>
where
    S: RawData,
    D: Dimension,
{
    UncheckedArray { array }
}

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
pub fn uncheck_arr_mut<'a, S, D>(array: &'a mut ArrayBase<S, D>) -> UncheckedArrayMut<'a, S, D>
where
    S: RawDataMut,
    D: Dimension,
{
    UncheckedArrayMut { array }
}

#[repr(transparent)]
pub struct UncheckedArray<'a, S, D>
where
    S: RawData,
    D: Dimension,
{
    pub array: &'a ArrayBase<S, D>,
}

#[repr(transparent)]
pub struct UncheckedArrayMut<'a, S, D>
where
    S: RawDataMut,
    D: Dimension,
{
    pub array: &'a mut ArrayBase<S, D>,
}
