use super::coo::COOTensor;
use crate::structs::axis::{Axes, AxisBuilder};
use crate::traits::ValType;
use ndarray::{self, Array2, ArrayBase, Data, Dimension, Ix, RawData};

impl<VT> COOTensor<Ix, VT>
where
    VT: ValType,
{
    pub fn from_ndarray<S, D>(array: ArrayBase<S, D>) -> Self
    where
        S: Data + RawData<Elem = VT>,
        D: Dimension,
    {
        let shape = array
            .shape()
            .into_iter()
            .map(|&x| AxisBuilder::new().range(0..x).build())
            .collect::<Axes<_>>();
        let dense_axes = shape.clone();
        let indices = Array2::zeros((1, 0));
        let values = array.insert_axis(ndarray::Axis(0)).into_dyn().into_owned();
        let sparse_sort_order = shape.clone();
        Self {
            name: None,
            shape,
            sparse_axes: Axes::new(),
            dense_axes,
            indices,
            values,
            sparse_is_sorted: true,
            sparse_sort_order,
        }
    }
}

impl<S, D> From<ArrayBase<S, D>> for COOTensor<Ix, S::Elem>
where
    S: Data,
    D: Dimension,
    S::Elem: ValType,
{
    fn from(array: ArrayBase<S, D>) -> Self {
        Self::from_ndarray(array)
    }
}
