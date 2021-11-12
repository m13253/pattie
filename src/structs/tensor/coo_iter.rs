use super::COOTensor;
use crate::traits::{IdxType, Tensor, TensorIter, ValType};
use ndarray::{iter, Ix1};
use smallvec::{smallvec, SmallVec};
use std::slice;

/// Iterator for [`COOTensor`].
pub struct COOIter<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
    tensor: &'a COOTensor<IT, VT>,
    // Extracted from indices_iter.
    sparse_index: SmallVec<[IT; 4]>,
    // Incrementing dense index.
    dense_index: SmallVec<[IT; 4]>,
    // `sparse_index` and `dense_index` combined.
    index: SmallVec<[IT; 4]>,
    // The map from `index` to `sparse_index` and `dense_matrix`.
    index_map: SmallVec<[*const IT; 4]>,
    size_hint: usize,
    indices_iter: iter::LanesIter<'a, IT, Ix1>,
    values_block_iter: iter::LanesIter<'a, VT, Ix1>,
    values_iter: Option<iter::Iter<'a, VT, Ix1>>,
}

impl<'a, IT, VT> COOIter<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
    pub(super) fn new(tensor: &'a COOTensor<IT, VT>) -> Self {
        let sparse_index = smallvec![IT::zero(); tensor.sparse_axes().len()];
        let dense_index = smallvec![IT::zero(); tensor.dense_storage_order().len()];
        let index = smallvec![IT::zero(); tensor.ndim()];
        let index_map = tensor
            .shape()
            .iter()
            .map(|axis| {
                tensor
                    .sparse_axes()
                    .iter()
                    .zip(sparse_index.iter())
                    .chain(tensor.dense_storage_order().iter().zip(dense_index.iter()))
                    .find(|(ax, _)| *ax == axis)
                    .map(|(_, idx)| idx as *const _)
                    .expect("cannot determine the storage order of axis")
            })
            .collect();
        let (indices, values) = tensor.raw_data();
        assert_eq!(sparse_index.len(), indices.ncols());
        let size_hint = values.len();
        Self {
            tensor,
            sparse_index,
            dense_index,
            index,
            index_map,
            size_hint,
            indices_iter: indices.rows().into_iter(),
            values_block_iter: values.rows().into_iter(),
            values_iter: None,
        }
    }

    fn next_block(&mut self) -> Option<()> {
        let next_sparse_indices = self.indices_iter.next()?;
        self.sparse_index
            .iter_mut()
            .zip(next_sparse_indices.iter())
            .for_each(|(x, y)| x.clone_from(y));
        self.dense_index
            .iter_mut()
            .zip(self.tensor.shape().iter())
            .for_each(|(x, y)| *x = y.lower());
        self.values_iter = Some(self.values_block_iter.next().unwrap().into_iter());
        Some(())
    }

    fn next_dense_index(&mut self) -> Option<()> {
        for (i, x) in self.dense_index.iter_mut().enumerate().rev() {
            if *x < self.tensor.dense_storage_order()[i].upper() {
                *x = x.clone() + IT::one();
                return Some(());
            } else {
                *x = self.tensor.dense_storage_order()[i].lower();
            }
        }
        None
    }
}

impl<'a, IT, VT> TensorIter<'a, IT, VT> for COOIter<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
}

impl<'a, IT, VT> Iterator for COOIter<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
    type Item = (&'a [IT], &'a VT);

    fn next(&mut self) -> Option<Self::Item> {
        if self.values_iter.is_none() || self.next_dense_index().is_none() {
            self.next_block()?;
        }

        let mut value: Option<&'a VT> = None;
        while value.is_none() {
            self.next_block()?;
            value = self.values_iter.as_mut().unwrap().next();
        }

        self.index
            .iter_mut()
            .zip(self.index_map.iter())
            .for_each(|(x, &y)| x.clone_from(unsafe { &*y }));

        // Unsolvable using safe Rust, let's just force the compiler to listen to us.
        // https://stackoverflow.com/a/30422716
        let index_unchecked =
            unsafe { slice::from_raw_parts(self.index.as_ptr(), self.index.len()) } as &'a _;

        Some((index_unchecked, value.unwrap()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size_hint, Some(self.size_hint))
    }
}

impl<'a, IT, VT> ExactSizeIterator for COOIter<'a, IT, VT>
where
    VT: 'a + ValType,
    IT: 'a + IdxType,
{
    fn len(&self) -> usize {
        self.size_hint
    }
}
