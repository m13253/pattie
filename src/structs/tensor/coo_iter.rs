use super::COOTensor;
use crate::structs::axis::Axis;
use crate::structs::vec::{smallvec, SmallVec};
use crate::traits::{IdxType, Tensor, TensorIter, ValType};
use ndarray::{ArrayView2, ArrayViewD, Ix};
use num::NumCast;
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
    pub(super) fn new(tensor: &'a COOTensor<IT, VT>) -> Self {
        let shape = tensor.shape();
        let sparse_axes = tensor.sparse_axes();
        let dense_axes = tensor.dense_axes();
        let (indices, values) = tensor.raw_data();

        let mut dense_strides = values
            .strides()
            .iter()
            .copied()
            .enumerate()
            .collect::<SmallVec<_>>();
        dense_strides.sort_unstable_by(|(a_index, a_stride), (b_index, b_stride)| {
            a_stride.cmp(&b_stride).then(b_index.cmp(&a_index))
        });
        let dense_strides_sorted = dense_strides;

        let dense_index_to_logic = dense_axes
            .iter()
            .map(|dense_axis| shape.iter().position(|axis| axis == dense_axis))
            .collect::<Option<SmallVec<_>>>()
            .expect("dense axis not found in shape");
        let sparse_index_to_logic = sparse_axes
            .iter()
            .map(|sparse_axis| shape.iter().position(|axis| axis == sparse_axis))
            .collect::<Option<SmallVec<_>>>()
            .expect("sparse axis not found in shape");

        let dense_index_buffer = smallvec![0; values.ndim()];
        let logic_index_buffer = smallvec![IT::zero(); shape.len()];

        Self {
            dense_axes,
            indices: indices.view(),
            values: values.view(),
            dense_strides_sorted,
            dense_index_to_logic,
            sparse_index_to_logic,
            dense_index_buffer,
            logic_index_buffer,
            result_buffer: None,
        }
    }

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
            self.logic_index_buffer[axis_idx] =
                <IT as NumCast>::from(index).unwrap() + axis.lower();
        }

        let result: (&[IT], &VT) = (
            &self.logic_index_buffer,
            &self.values[self.dense_index_buffer.as_slice()],
        );
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

    fn get(&self) -> Option<&Self::Item> {
        self.result_buffer.as_ref()
    }

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

impl<'a, IT, VT> TensorIter<'a, IT, VT> for COOIter<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
}
