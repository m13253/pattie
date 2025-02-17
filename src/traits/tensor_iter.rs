use super::{IdxType, ValType};
use streaming_iterator::StreamingIterator;

/// An iterator over a tensor.
///
/// `'a` is the lifetime of the tensor from which the iterator is created.
/// `VT` is the type of the values inside the tensor.
/// `IT` is the type of the indices of the tensor.
///
/// Each element of the iterator is a tuple of the index and the value.
pub trait TensorIter<'a, IT, VT>: StreamingIterator<Item = (&'a [IT], &'a VT)>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
}

impl<'a, T, IT, VT> TensorIter<'a, IT, VT> for T
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
    T: StreamingIterator<Item = (&'a [IT], &'a VT)>,
{
}

/// A mutable iterator over a tensor.
///
/// `'a` is the lifetime of the tensor from which the iterator is created.
/// `VT` is the type of the values inside the tensor.
/// `IT` is the type of the indices of the tensor.
///
/// A mutable iterator is used to modify each element of the tensor.
/// Each element of the iterator is a tuple of the index and the mutable value.
pub trait TensorIterMut<'a, IT, VT>: StreamingIterator<Item = (&'a [IT], &'a mut VT)>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
}

impl<'a, T, IT, VT> TensorIterMut<'a, IT, VT> for T
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
    T: StreamingIterator<Item = (&'a [IT], &'a mut VT)>,
{
}

/// A moved iterator over a tensor.
///
/// `'a` is the lifetime of the tensor from which the iterator is created.
/// `VT` is the type of the values inside the tensor.
/// `IT` is the type of the indices of the tensor.
///
/// A moved iterator is used to consume each element of the tensor, meanwhile releasing the ownership of the tensor.
/// Each element of the iterator is a tuple of the index and the moved value.
pub trait TensorIntoIter<'a, IT, VT>: StreamingIterator<Item = (&'a [IT], VT)>
where
    IT: 'a + IdxType,
    VT: ValType,
{
}

impl<'a, T, IT, VT> TensorIntoIter<'a, IT, VT> for T
where
    IT: 'a + IdxType,
    VT: ValType,
    T: StreamingIterator<Item = (&'a [IT], VT)>,
{
}
