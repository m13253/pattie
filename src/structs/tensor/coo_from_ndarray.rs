use super::coo::{COOTensor, COOTensorInner};
use crate::structs::axis::{Axes, AxisBuilder};
use crate::traits::{IdxType, RawParts, ValType};
use ndarray::{self, Array2, ArrayBase, Data, Dimension, RawData};
use num::NumCast;

impl<IT, VT> COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    #[inline]
    pub fn from_ndarray<S, D>(array: ArrayBase<S, D>) -> Self
    where
        S: Data + RawData<Elem = VT>,
        D: Dimension,
    {
        let shape = array
            .shape()
            .iter()
            .map(|&x| {
                AxisBuilder::new()
                    .range(IT::zero()..<IT as NumCast>::from(x).unwrap())
                    .build()
            })
            .collect::<Axes<_>>();
        let dense_axes = shape.clone();
        let indices = Array2::zeros((1, 0));
        let values = array.insert_axis(ndarray::Axis(0)).into_dyn().into_owned();
        let sparse_sort_order = shape.clone();

        let raw_parts = COOTensorInner {
            name: None,
            shape,
            sparse_axes: Axes::new(),
            dense_axes,
            indices,
            values,
            sparse_is_sorted: true,
            sparse_sort_order,
        };
        // # Safety
        // We have checked the data integrity
        unsafe { Self::from_raw_parts(raw_parts) }
    }
}

impl<S, D, IT> From<ArrayBase<S, D>> for COOTensor<IT, S::Elem>
where
    S: Data,
    D: Dimension,
    IT: IdxType,
    S::Elem: ValType,
{
    #[inline]
    fn from(array: ArrayBase<S, D>) -> Self {
        Self::from_ndarray(array)
    }
}
