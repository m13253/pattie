use super::coo_iter::COOIter;
use super::coo_iter_mut::COOIterMut;
use crate::structs::axis::{Axes, Axis};
use crate::traits::{IdxType, Tensor, ValType};
use ndarray::{aview1, Array2};
use num::NumCast;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    name: Option<String>,
    shape: Axes<IT>,
    sparse_axes: Axes<IT>,
    sparse_storage_is_sorted: bool,
    sparse_storage_order: Axes<IT>,
    dense_storage_order: Axes<IT>,
    // First index is each non-zero block, second index is each sparse axis.
    indices: Array2<IT>,
    // First index is each non-zero block, second index is the C-format offset for dense axes.
    values: Array2<VT>,
}

impl<IT, VT> COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    pub fn zeros(shape: &[Axis<IT>], is_axis_dense: &[bool]) -> Self {
        assert_eq!(shape.len(), is_axis_dense.len());
        let mut sparse_axes = Axes::new();
        let mut sparse_storage_order = Axes::new();
        let mut dense_storage_order = Axes::new();
        let mut dense_block_size = 1_usize;
        for (axis, is_dense) in shape.iter().zip(is_axis_dense.iter()) {
            if *is_dense {
                dense_storage_order.push(axis.clone());
                dense_block_size = dense_block_size
                    .checked_mul(NumCast::from(axis.size()).unwrap())
                    .unwrap();
            } else {
                sparse_axes.push(axis.clone());
                sparse_storage_order.push(axis.clone());
            }
        }
        let num_sparse_axes = sparse_axes.len();
        Self {
            name: None,
            shape: Axes::from(shape),
            sparse_axes,
            sparse_storage_is_sorted: true,
            sparse_storage_order,
            dense_storage_order,
            indices: Array2::zeros((0, num_sparse_axes)),
            values: Array2::zeros((0, dense_block_size)),
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

    pub fn sparse_storage_order(&self) -> Option<&[Axis<IT>]> {
        if self.sparse_storage_is_sorted {
            Some(&self.sparse_storage_order)
        } else {
            None
        }
    }

    pub unsafe fn sparse_storage_order_mut(&mut self) -> &mut Axes<IT> {
        // Since nobody can read sort_order before it is returned, we can safely set is_sorted aforehand.
        self.sparse_storage_is_sorted = true;
        &mut self.sparse_storage_order
    }

    pub fn clear_sparse_storage_order(&mut self) {
        self.sparse_storage_is_sorted = false;
    }

    pub fn dense_storage_order(&self) -> &[Axis<IT>] {
        &self.dense_storage_order
    }

    pub unsafe fn dense_storage_order_mut(&mut self) -> &mut Axes<IT> {
        &mut self.dense_storage_order
    }

    pub fn dense_block_size(&self) -> usize {
        self.values.ncols()
    }

    pub fn raw_data(&self) -> (&Array2<IT>, &Array2<VT>) {
        (&self.indices, &self.values)
    }

    pub unsafe fn raw_data_mut(&mut self) -> (&mut Array2<IT>, &mut Array2<VT>) {
        (&mut self.indices, &mut self.values)
    }

    pub fn push_block(&mut self, index: &[IT], value: &[VT]) {
        assert_eq!(index.len(), self.sparse_axes.len());
        assert!(index
            .iter()
            .zip(self.sparse_axes.iter())
            .all(|(idx, axis)| axis.range().contains(idx)));
        self.clear_sparse_storage_order();
        self.indices.push_row(aview1(index)).unwrap();
        self.values.push_row(aview1(value)).unwrap();
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
