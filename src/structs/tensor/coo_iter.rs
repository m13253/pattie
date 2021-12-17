use super::COOTensor;
use crate::structs::axis::{map_axes_unwrap, Axis};
use crate::structs::vec::{smallvec, SmallVec};
use crate::traits::{IdxType, RawParts, ValType};
use ndarray::{ArrayView2, ArrayViewD, Ix};
use num::{NumCast, ToPrimitive};
use streaming_iterator::{DoubleEndedStreamingIterator, StreamingIterator};

/// Mutable iterator for [`COOTensor`].
pub struct COOIter<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
    dense_axes: &'a [Axis<IT>],
    indices: ArrayView2<'a, IT>,
    values: ArrayViewD<'a, VT>,

    dense_strides_sorted: SmallVec<(usize, isize)>,

    dense_index_to_logic: SmallVec<usize>,
    sparse_index_to_logic: SmallVec<usize>,

    dense_index_buffer: SmallVec<Ix>,
    logic_index_buffer: SmallVec<IT>,
    result_buffer: Option<(&'a [IT], &'a VT)>,
}

impl<'a, IT, VT> COOIter<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
    #[inline]
    pub(super) fn new(tensor: &'a COOTensor<IT, VT>) -> Self {
        let raw_parts = tensor.raw_parts();

        let shape = raw_parts.shape.as_slice();
        let sparse_axes = raw_parts.sparse_axes.as_slice();
        let dense_axes = raw_parts.dense_axes.as_slice();
        let indices = raw_parts.indices.view();
        let values = raw_parts.values.view();

        let mut dense_strides = values
            .strides()
            .iter()
            .copied()
            .enumerate()
            .collect::<SmallVec<_>>();
        dense_strides.sort_unstable_by(|(a_index, a_stride), (b_index, b_stride)| {
            a_stride.cmp(b_stride).then(b_index.cmp(a_index))
        });
        let dense_strides_sorted = dense_strides;

        let dense_index_to_logic = map_axes_unwrap(dense_axes, shape).collect::<SmallVec<_>>();
        let sparse_index_to_logic = map_axes_unwrap(sparse_axes, shape).collect::<SmallVec<_>>();

        let dense_index_buffer = smallvec![0; values.ndim()];
        let logic_index_buffer = smallvec![IT::zero(); shape.len()];

        Self {
            dense_axes,
            indices,
            values,
            dense_strides_sorted,
            dense_index_to_logic,
            sparse_index_to_logic,
            dense_index_buffer,
            logic_index_buffer,
            result_buffer: None,
        }
    }

    #[inline]
    fn calc_result(&mut self) -> (&'a [IT], &'a VT) {
        let sparse_block_idx = self.dense_index_buffer[0];
        let sparse_index_buffer = self.indices.row(sparse_block_idx);
        for (index, &axis_idx) in sparse_index_buffer
            .iter()
            .zip(self.sparse_index_to_logic.iter())
        {
            self.logic_index_buffer[axis_idx].clone_from(index);
        }

        for ((&index, axis), &axis_idx) in self
            .dense_index_buffer
            .iter()
            .skip(1)
            .zip(self.dense_axes.iter())
            .zip(self.dense_index_to_logic.iter())
        {
            // Checking overflow, since we are converting usize into IT
            self.logic_index_buffer[axis_idx] = <IT as NumCast>::from(
                index
                    .to_isize()
                    .unwrap()
                    .checked_add(axis.lower().to_isize().unwrap())
                    .unwrap(),
            )
            .unwrap();
        }

        let result: (&[IT], &VT) = (
            &self.logic_index_buffer,
            &self.values[self.dense_index_buffer.as_slice()],
        );
        // # Safety
        // TODO: I can't pass the borrow checker.
        // There must be a better way to do this.
        unsafe {
            (
                (result.0 as *const [IT]).as_ref::<'a>().unwrap_unchecked(),
                (result.1 as *const VT).as_ref::<'a>().unwrap_unchecked(),
            )
        }
    }
}

impl<'a, IT, VT> StreamingIterator for COOIter<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
    type Item = (&'a [IT], &'a VT);

    #[inline]
    fn advance(&mut self) {
        if self.result_buffer.is_some() {
            for &(dense_axis, _stride) in self.dense_strides_sorted.iter() {
                self.dense_index_buffer[dense_axis] += 1;
                if self.dense_index_buffer[dense_axis] >= self.values.shape()[dense_axis] {
                    self.dense_index_buffer[dense_axis] = 0;
                } else {
                    self.result_buffer = Some(self.calc_result());
                    return;
                }
            }
            self.result_buffer = None;
        } else {
            self.dense_index_buffer.fill(0);
            for &(dense_axis, _stride) in self.dense_strides_sorted.iter() {
                self.dense_index_buffer[dense_axis] = 0;
                if self.dense_index_buffer[dense_axis] >= self.values.shape()[dense_axis] {
                    return;
                }
            }
            self.result_buffer = Some(self.calc_result());
        }
    }

    #[inline]
    fn get(&self) -> Option<&Self::Item> {
        self.result_buffer.as_ref()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size_hint = self.values.len();
        (size_hint, Some(size_hint))
    }
}

impl<'a, IT, VT> DoubleEndedStreamingIterator for COOIter<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
    #[inline]
    fn advance_back(&mut self) {
        if self.result_buffer.is_some() {
            for &(dense_axis, _stride) in self.dense_strides_sorted.iter() {
                if self.dense_index_buffer[dense_axis] == 0 {
                    let axis_len = self.values.shape()[dense_axis];
                    if axis_len != 0 {
                        self.dense_index_buffer[dense_axis] = axis_len - 1;
                    }
                } else {
                    self.dense_index_buffer[dense_axis] -= 1;
                    self.result_buffer = Some(self.calc_result());
                    return;
                }
            }
            self.result_buffer = None;
        } else {
            for &(dense_axis, _stride) in self.dense_strides_sorted.iter() {
                let axis_len = self.values.shape()[dense_axis];
                if axis_len == 0 {
                    return;
                }
                self.dense_index_buffer[dense_axis] = axis_len - 1;
            }
            self.result_buffer = Some(self.calc_result());
        }
    }
}
