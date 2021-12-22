use super::{UncheckedArray, UncheckedArrayMut};
use ndarray::{ArrayView2, ArrayViewMut2, Dimension, Ix2, RawData, RawDataMut};
use std::slice;

impl<'a, S> UncheckedArray<'a, S, Ix2>
where
    S: RawData,
{
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure idx.0 < array.nrows() && idx.1 < array.ncols()
    #[inline]
    pub unsafe fn get(self, idx: (usize, usize)) -> &'a S::Elem {
        let shape = self.array.raw_dim().into_pattern();
        let offset = idx.0 * shape.1 + idx.1;
        &*self.array.as_ptr().offset(offset as isize)
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure from.0 * array.ncols() + from.1 <= to.0 * array.ncols() + to.1 < array.len()
    #[inline]
    pub unsafe fn slice(self, from: (usize, usize), to: (usize, usize)) -> &'a [S::Elem] {
        let shape = self.array.raw_dim().into_pattern();
        let offset_from = from.0 * shape.1 + from.1;
        let offset_to = to.0 * shape.1 + to.1;
        slice::from_raw_parts(
            self.array.as_ptr().offset(offset_from as isize),
            offset_to - offset_from,
        )
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure idx < array.nrows()
    #[inline]
    pub unsafe fn row(self, idx: usize) -> &'a [S::Elem] {
        let shape = self.array.raw_dim().into_pattern();
        let offset = idx * shape.1;
        slice::from_raw_parts(self.array.as_ptr().offset(offset as isize), shape.1)
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure from <= to < array.nrows()
    #[inline]
    pub unsafe fn row_slice(self, from: usize, to: usize) -> ArrayView2<'a, S::Elem> {
        let shape = self.array.raw_dim().into_pattern();
        let offset = from * shape.1;
        ArrayView2::from_shape_ptr(
            (to - from, shape.1),
            self.array.as_ptr().offset(offset as isize),
        )
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    #[inline]
    pub unsafe fn as_slice(self) -> &'a [S::Elem] {
        slice::from_raw_parts(self.array.as_ptr(), self.array.len())
    }
}

impl<'a, S> UncheckedArrayMut<'a, S, Ix2>
where
    S: RawDataMut,
{
    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure idx.0 < array.nrows() && idx.1 < array.ncols()
    #[inline]
    pub unsafe fn get(self, idx: (usize, usize)) -> &'a mut S::Elem {
        let shape = self.array.raw_dim().into_pattern();
        let offset = idx.0 * shape.1 + idx.1;
        &mut *self.array.as_mut_ptr().offset(offset as isize)
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure from.0 * array.ncols() + from.1 <= to.0 * array.ncols() + to.1 < array.len()
    #[inline]
    pub unsafe fn slice(self, from: (usize, usize), to: (usize, usize)) -> &'a mut [S::Elem] {
        let shape = self.array.raw_dim().into_pattern();
        let offset_from = from.0 * shape.1 + from.1;
        let offset_to = to.0 * shape.1 + to.1;
        slice::from_raw_parts_mut(
            self.array.as_mut_ptr().offset(offset_from as isize),
            offset_to - offset_from,
        )
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure idx < array.nrows()
    #[inline]
    pub unsafe fn row(self, idx: usize) -> &'a mut [S::Elem] {
        let shape = self.array.raw_dim().into_pattern();
        let offset = idx * shape.1;
        slice::from_raw_parts_mut(self.array.as_mut_ptr().offset(offset as isize), shape.1)
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    /// Make sure from <= to < array.nrows()
    #[inline]
    pub unsafe fn row_slice(self, from: usize, to: usize) -> ArrayViewMut2<'a, S::Elem> {
        let shape = self.array.raw_dim().into_pattern();
        let offset = from * shape.1;
        ArrayViewMut2::from_shape_ptr(
            (to - from, shape.1),
            self.array.as_mut_ptr().offset(offset as isize),
        )
    }

    /// # Safety
    /// Make sure array.is_standard_layout() == true
    #[inline]
    pub unsafe fn as_slice(self) -> &'a mut [S::Elem] {
        slice::from_raw_parts_mut(self.array.as_mut_ptr(), self.array.len())
    }
}
