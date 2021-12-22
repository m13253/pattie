use super::{UncheckedArray, UncheckedArrayMut};
use ndarray::{Dimension, Ix1, RawData, RawDataMut};
use std::slice;

impl<'a, S> UncheckedArray<'a, S, Ix1>
where
    S: RawData,
{
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure idx < array.len()
    pub unsafe fn get(self, idx: usize) -> &'a S::Elem {
        &*self.array.as_ptr().offset(idx as isize)
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure from <= to < array.len()
    pub unsafe fn slice(self, from: usize, to: usize) -> &'a [S::Elem] {
        slice::from_raw_parts(self.array.as_ptr().offset(from as isize), to - from)
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    pub unsafe fn as_slice(self) -> &'a [S::Elem] {
        let shape = self.array.raw_dim().into_pattern();
        slice::from_raw_parts(self.array.as_ptr(), shape)
    }
}

impl<'a, S> UncheckedArrayMut<'a, S, Ix1>
where
    S: RawDataMut,
{
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure idx < array.len()
    pub unsafe fn get(self, idx: usize) -> &'a mut S::Elem {
        &mut *self.array.as_mut_ptr().offset(idx as isize)
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure from <= to < array.len()
    pub unsafe fn slice(self, from: usize, to: usize) -> &'a mut [S::Elem] {
        slice::from_raw_parts_mut(self.array.as_mut_ptr().offset(from as isize), to - from)
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    pub unsafe fn as_slice(self) -> &'a mut [S::Elem] {
        let shape = self.array.raw_dim().into_pattern();
        slice::from_raw_parts_mut(self.array.as_mut_ptr(), shape)
    }
}
