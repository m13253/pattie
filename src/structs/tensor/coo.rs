use super::coo_iter::COOIter;
use super::coo_iter_mut::COOIterMut;
use crate::traits::{self, Tensor};
use ndarray::{aview0, aview1, Array1, Array2, Axis, Dimension, Ix, IxDyn, ShapeBuilder};
use num::NumCast;

#[derive(Clone, Debug)]
pub struct COO<VT, IT>
where
    VT: traits::ValType,
    IT: traits::IdxType,
{
    dim: IxDyn,
    is_sorted: bool,
    sort_order: Array1<Axis>,
    indices: Array2<IT>,
    values: Array1<VT>,
}

impl<VT, IT> COO<VT, IT>
where
    VT: traits::ValType,
    IT: traits::IdxType,
{
    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = IxDyn>,
    {
        let shape = shape.into_shape();
        let dim = shape.raw_dim();
        let ndim = dim.ndim();
        Self {
            dim: dim.clone(),
            is_sorted: true,
            sort_order: (0..ndim).map(Axis).collect(),
            indices: Array2::zeros((0, ndim)),
            values: Array1::zeros(0),
        }
    }

    pub fn sort_order(&self) -> Option<&Array1<Axis>> {
        if self.is_sorted {
            Some(&self.sort_order)
        } else {
            None
        }
    }

    pub unsafe fn set_sort_order(&mut self) -> &mut Array1<Axis> {
        // Since nobody can read sort_order before it is returned, we can safely set is_sorted aforehand.
        self.is_sorted = true;
        &mut self.sort_order
    }

    pub fn clear_sort_order(&mut self) {
        self.is_sorted = false;
    }

    pub fn raw_data(&self) -> (&Array2<IT>, &Array1<VT>) {
        (&self.indices, &self.values)
    }

    pub unsafe fn raw_data_mut(&mut self) -> (&mut Array2<IT>, &mut Array1<VT>) {
        (&mut self.indices, &mut self.values)
    }

    pub fn push(&mut self, index: &[IT], value: VT) {
        assert_eq!(index.len(), self.dim.ndim());
        assert!(index
            .iter()
            .zip(self.shape().iter())
            .all(|(i, s)| <usize as NumCast>::from(i.clone())
                .map(|i| i < *s)
                .unwrap_or(false)));
        self.indices.push_row(aview1(index)).unwrap();
        self.values.push(Axis(0), aview0(&value)).unwrap();
    }

    pub fn iter(&self) -> COOIter<'_, VT, IT> {
        let (indices, values) = self.raw_data();
        COOIter {
            indices: indices.rows().into_iter(),
            values: values.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> COOIterMut<'_, VT, IT> {
        let (indices, values) = unsafe { self.raw_data_mut() };
        COOIterMut {
            indices: indices.rows().into_iter(),
            values: values.iter_mut(),
        }
    }
}

impl<VT, IT> traits::Tensor<VT> for COO<VT, IT>
where
    VT: traits::ValType,
    IT: traits::IdxType,
{
    type Dim = IxDyn;

    fn ndim(&self) -> usize {
        self.dim.size()
    }

    fn num_non_zeros(&self) -> usize {
        debug_assert_eq!(self.indices.nrows(), self.values.len());
        self.values.len()
    }

    fn raw_dim(&self) -> Self::Dim {
        self.dim.clone()
    }

    fn shape(&self) -> &[Ix] {
        self.dim.slice()
    }
}
