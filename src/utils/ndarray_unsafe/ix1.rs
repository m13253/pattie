use super::{UncheckedArray, UncheckedArrayMut};
use ndarray::{Dimension, Ix1, RawData, RawDataMut};
use std::slice;

impl<'a, S> UncheckedArray<'a, S, Ix1>
where
    S: RawData,
{
    /// Get a reference to the element at the given index.
    ///
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure idx < array.len()
    #[inline]
    pub unsafe fn get(self, idx: usize) -> &'a S::Elem {
        &*self.array.as_ptr().add(idx)
    }

    /// Slice the 1D array from the given start index to the given end index.
    ///
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure from <= to < array.len()
    #[inline]
    pub unsafe fn slice(self, from: usize, to: usize) -> &'a [S::Elem] {
        slice::from_raw_parts(self.array.as_ptr().add(from), to - from)
    }

    /// Convert the 1D array to a slice.
    ///
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    #[inline]
    pub unsafe fn as_slice(self) -> &'a [S::Elem] {
        let shape = self.array.raw_dim().into_pattern();
        slice::from_raw_parts(self.array.as_ptr(), shape)
    }
}

impl<'a, S> UncheckedArrayMut<'a, S, Ix1>
where
    S: RawDataMut,
{
    /// Get a mutable reference to the element at the given index.
    ///
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure idx < array.len()
    #[inline]
    pub unsafe fn get(self, idx: usize) -> &'a mut S::Elem {
        &mut *self.array.as_mut_ptr().add(idx)
    }

    /// Slice the 1D array from the given start index to the given end index.
    ///
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure from <= to < array.len()
    #[inline]
    pub unsafe fn slice(self, from: usize, to: usize) -> &'a mut [S::Elem] {
        slice::from_raw_parts_mut(self.array.as_mut_ptr().add(from), to - from)
    }

    /// Convert the 1D array to a mutable slice.
    ///
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    #[inline]
    pub unsafe fn as_slice(self) -> &'a mut [S::Elem] {
        let shape = self.array.raw_dim().into_pattern();
        slice::from_raw_parts_mut(self.array.as_mut_ptr(), shape)
    }
}
