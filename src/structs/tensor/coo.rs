use super::coo_iter::COOIter;
use super::coo_iter_mut::COOIterMut;
use crate::structs::axis::{Axes, Axis};
use crate::traits::{IdxType, Tensor, ValType};
use ndarray::{self, Array2, ArrayD, ArrayView1, ArrayViewD};
use num::NumCast;
use std::fmt::Debug;
use std::iter::once;

#[derive(Clone, Debug)]
pub struct COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    pub(super) name: Option<String>,

    pub(super) shape: Axes<IT>,
    pub(super) sparse_axes: Axes<IT>,
    pub(super) dense_axes: Axes<IT>,

    // First index is each non-zero block, second index is each sparse axis.
    pub(super) indices: Array2<IT>,
    // First index is each non-zero block, remaining indices are each dense axis.
    pub(super) values: ArrayD<VT>,

    pub(super) sparse_is_sorted: bool,
    pub(super) sparse_sort_order: Axes<IT>,
}

impl<IT, VT> COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
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
            once(0)
                .chain(
                    dense_axes
                        .iter()
                        .map(|axis| NumCast::from(axis.size()).unwrap()),
                )
                .collect::<Vec<_>>(),
        );
        Self {
            name: None,
            shape: Axes::from(shape),
            sparse_axes,
            dense_axes,
            indices,
            values,
            sparse_is_sorted: true,
            sparse_sort_order,
        }
    }

    pub fn name_mut(&mut self) -> &mut Option<String> {
        &mut self.name
    }

    pub unsafe fn shape_mut(&mut self) -> &mut Axes<IT> {
        &mut self.shape
    }

    pub fn sparse_axes(&self) -> &[Axis<IT>] {
        &self.sparse_axes
    }

    pub unsafe fn sparse_axes_mut(&mut self) -> &mut Axes<IT> {
        &mut self.sparse_axes
    }

    pub fn dense_axes(&self) -> &[Axis<IT>] {
        &self.dense_axes
    }

    pub unsafe fn dense_axes_mut(&mut self) -> &mut Axes<IT> {
        &mut self.dense_axes
    }

    pub fn raw_data(&self) -> (&Array2<IT>, &ArrayD<VT>) {
        (&self.indices, &self.values)
    }

    pub unsafe fn raw_data_mut(&mut self) -> (&mut Array2<IT>, &mut ArrayD<VT>) {
        (&mut self.indices, &mut self.values)
    }

    pub fn sparse_sort_order(&self) -> Option<&[Axis<IT>]> {
        if self.sparse_is_sorted {
            Some(&self.sparse_sort_order)
        } else {
            None
        }
    }

    pub unsafe fn sparse_sort_order_mut(&mut self) -> &mut Axes<IT> {
        // Since nobody can read sort_order before it is returned, we can safely set is_sorted aforehand.
        self.sparse_is_sorted = true;
        &mut self.sparse_sort_order
    }

    pub fn clear_sparse_sort_order(&mut self) {
        self.sparse_is_sorted = false;
    }

    pub fn push_block(&mut self, sparse_index: ArrayView1<IT>, value: ArrayViewD<VT>) {
        assert_eq!(sparse_index.len(), self.sparse_axes.len());
        assert!(sparse_index
            .iter()
            .zip(self.sparse_axes.iter())
            .all(|(idx, axis)| axis.range().contains(idx)));
        assert_eq!(value.shape(), &self.values.shape()[1..]);

        self.clear_sparse_sort_order();
        self.indices.push_row(sparse_index).unwrap();
        self.values.push(ndarray::Axis(0), value).unwrap();
    }

    pub fn iter(&self) -> COOIter<'_, IT, VT> {
        COOIter::new(self)
    }

    pub fn iter_mut(&mut self) -> COOIterMut<'_, IT, VT> {
        COOIterMut::new(self)
    }
}

impl<IT, VT> Tensor<IT, VT> for COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn num_non_zeros(&self) -> usize {
        self.values.len()
    }

    fn shape(&self) -> &[Axis<IT>] {
        &self.shape
    }
}
