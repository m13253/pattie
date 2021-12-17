use super::coo_iter::COOIter;
use super::coo_iter_mut::COOIterMut;
use crate::structs::axis::{Axes, Axis};
use crate::traits::{IdxType, RawParts, Tensor, ValType};
use ndarray::{self, Array2, ArrayD, ArrayView1, ArrayViewD};
use std::fmt::Debug;
use std::iter;

#[derive(Clone, Debug)]
pub struct COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    inner: COOTensorInner<IT, VT>,
}

#[derive(Clone, Debug)]
pub struct COOTensorInner<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    pub name: Option<String>,

    /// The logical shape of this tensor, expressed as an array of axes.
    /// Each axis contains a lower bound and an upper bound.
    pub shape: Axes<IT>,

    /// All sparse axes.
    pub sparse_axes: Axes<IT>,
    /// All dense axes.
    pub dense_axes: Axes<IT>,

    /// First index is each non-zero block, second index is each sparse axis, in the order of `sparse_axes`.
    pub indices: Array2<IT>,
    /// First index is each non-zero block, remaining indices are each dense axis.
    pub values: ArrayD<VT>,

    /// Are all sparse block sorted lexicographically?
    pub sparse_is_sorted: bool,
    /// Which sparse axis is the outermost?
    pub sparse_sort_order: Axes<IT>,
}

impl<IT, VT> COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    #[inline]
    pub fn zeros(shape: &[Axis<IT>], is_axis_dense: &[bool]) -> Self {
        assert_eq!(shape.len(), is_axis_dense.len());
        let mut sparse_axes = Axes::new();
        let mut dense_axes = Axes::new();
        let mut sparse_sort_order = Axes::new();
        for (axis, is_dense) in shape.iter().zip(is_axis_dense.iter()) {
            if *is_dense {
                dense_axes.push(axis.clone());
            } else {
                sparse_axes.push(axis.clone());
                sparse_sort_order.push(axis.clone());
            }
        }
        let indices = Array2::zeros((0, sparse_axes.len()));
        let values = ArrayD::zeros(
            iter::once(0)
                .chain(
                    dense_axes
                        .iter()
                        .map(|axis| axis.size().to_usize().unwrap()),
                )
                .collect::<Vec<_>>(),
        );
        Self {
            inner: COOTensorInner {
                name: None,
                shape: Axes::from(shape),
                sparse_axes,
                dense_axes,
                indices,
                values,
                sparse_is_sorted: true,
                sparse_sort_order,
            },
        }
    }

    #[inline]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }

    #[inline]
    fn name_mut(&mut self) -> &mut Option<String> {
        &mut self.inner.name
    }

    #[inline]
    pub fn sparse_axes(&self) -> &[Axis<IT>] {
        &self.inner.sparse_axes
    }

    #[inline]
    pub fn dense_axes(&self) -> &[Axis<IT>] {
        &self.inner.dense_axes
    }

    #[inline]
    pub fn sparse_sort_order(&self) -> Option<&[Axis<IT>]> {
        if self.inner.sparse_is_sorted {
            Some(&self.inner.sparse_sort_order)
        } else {
            None
        }
    }

    #[inline]
    pub fn clear_sparse_sort_order(&mut self) {
        self.inner.sparse_is_sorted = false;
    }

    #[inline]
    pub fn push_block(&mut self, sparse_index: ArrayView1<IT>, value: ArrayViewD<VT>) {
        assert_eq!(sparse_index.len(), self.inner.sparse_axes.len());
        assert!(sparse_index
            .iter()
            .zip(self.inner.sparse_axes.iter())
            .all(|(idx, axis)| axis.range().contains(idx)));
        assert_eq!(value.shape(), &self.inner.values.shape()[1..]);

        self.clear_sparse_sort_order();
        self.inner.indices.push_row(sparse_index).unwrap();
        self.inner.values.push(ndarray::Axis(0), value).unwrap();
    }

    #[inline]
    pub fn iter(&self) -> COOIter<'_, IT, VT> {
        COOIter::new(self)
    }

    #[inline]
    pub fn iter_mut(&mut self) -> COOIterMut<'_, IT, VT> {
        COOIterMut::new(self)
    }
}

impl<IT, VT> Tensor<IT, VT> for COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    #[inline]
    fn name(&self) -> Option<&str> {
        self.name()
    }

    #[inline]
    fn name_mut(&mut self) -> &mut Option<String> {
        self.name_mut()
    }

    #[inline]
    fn num_non_zeros(&self) -> usize {
        self.inner.values.len()
    }

    #[inline]
    fn shape(&self) -> &[Axis<IT>] {
        &self.inner.shape
    }
}

impl<IT, VT> RawParts for COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    type Inner = COOTensorInner<IT, VT>;

    #[inline]
    unsafe fn from_raw_parts(raw_parts: Self::Inner) -> Self {
        Self { inner: raw_parts }
    }

    #[inline]
    fn into_raw_parts(self) -> Self::Inner {
        self.inner
    }

    #[inline]
    fn raw_parts(&self) -> &Self::Inner {
        &self.inner
    }

    #[inline]
    unsafe fn raw_parts_mut(&mut self) -> &mut Self::Inner {
        &mut self.inner
    }
}
