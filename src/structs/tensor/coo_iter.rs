use ndarray::{iter::Iter, iter::LanesIter, Ix1};

use crate::traits;

pub struct COOIter<'a, VT, IT>
where
    VT: 'a + traits::ValType,
    IT: 'a + traits::IdxType,
{
    pub(super) indices: LanesIter<'a, IT, Ix1>,
    pub(super) values: Iter<'a, VT, Ix1>,
}

impl<'a, VT, IT> traits::TensorIter<'a, VT, IT> for COOIter<'a, VT, IT>
where
    VT: 'a + traits::ValType,
    IT: 'a + traits::IdxType,
{
}

impl<'a, VT, IT> Iterator for COOIter<'a, VT, IT>
where
    VT: 'a + traits::ValType,
    IT: 'a + traits::IdxType,
{
    type Item = (&'a [IT], &'a VT);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.indices.next()?;
        let value = self.values.next().unwrap();
        Some((index.to_slice().unwrap(), value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, VT, IT> ExactSizeIterator for COOIter<'a, VT, IT>
where
    VT: 'a + traits::ValType,
    IT: 'a + traits::IdxType,
{
    fn len(&self) -> usize {
        let len1 = self.indices.len();
        let len2 = self.values.len();
        debug_assert_eq!(len1, len2);
        len2
    }
}
