use crate::structs::axis::{map_axes_unwrap, Axis};
use crate::structs::tensor::COOTensor;
use crate::structs::vec::SmallVec;
use crate::traits::{IdxType, RawParts, ValType};
use ndarray::{Array2, ArrayView1, ArrayViewMut2};

/// Sort the storage order of elements inside a `COOTensor`.
pub struct SortCOOTensor<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
    pub tensor: &'a mut COOTensor<IT, VT>,
    pub order: &'a [Axis<IT>],
}

impl<'a, IT, VT> SortCOOTensor<'a, IT, VT>
where
    IT: 'a + IdxType,
    VT: 'a + ValType,
{
    /// Create a new `SortCOOTensor` task.
    #[must_use]
    pub fn new(tensor: &'a mut COOTensor<IT, VT>, order: &'a [Axis<IT>]) -> Self {
        Self { tensor, order }
    }

    /// Perform the sorting.
    ///
    /// The sort operation uses quick-sort algorithm, but may change in future versions.
    pub fn execute(self) {
        let tensor = self.tensor;
        let raw_parts = unsafe { tensor.raw_parts_mut() };
        let order = self.order;
        let order_index = map_axes_unwrap(order, &raw_parts.sparse_axes).collect::<SmallVec<_>>();

        // Reshape values into 2D array for more efficient indexing.
        let num_blocks = raw_parts.indices.nrows();
        let values_shape = raw_parts.values.shape().split_first().unwrap();
        assert_eq!(num_blocks, *values_shape.0);
        let values_2d_shape = (*values_shape.0, values_shape.1.iter().product());
        let mut values_2d = raw_parts
            .values
            .view_mut()
            .into_shape(values_2d_shape)
            .unwrap();

        // Call the quick sort algorithm.
        Self::sort_subtensor(
            &mut raw_parts.indices,
            &mut values_2d,
            &order_index,
            0,
            num_blocks,
        );

        // Mark the tensor as sorted.
        raw_parts.sparse_sort_order.clone_from_slice(order);
        raw_parts.sparse_is_sorted = true;
    }

    fn sort_subtensor(
        indices: &mut Array2<IT>,
        values: &mut ArrayViewMut2<VT>,
        order: &[usize],
        from: usize,
        to: usize,
    ) {
        if to - from < 2 {
            return;
        }
        let mut pivot = (from + to) / 2;
        let mut i = from;
        let mut j = to - 1;
        loop {
            while Self::index_less_than(order, &indices.row(i), &indices.row(pivot)) {
                i += 1;
            }
            while Self::index_less_than(order, &indices.row(j), &indices.row(pivot)) {
                j -= 1;
            }
            if i >= j {
                break;
            }
            Self::swap_block(indices, values, i, j);
            if i == pivot {
                pivot = j;
            } else if j == pivot {
                pivot = i;
            }
            i += 1;
            j -= 1;
        }
        Self::sort_subtensor(indices, values, order, from, j);
        Self::sort_subtensor(indices, values, order, i, to);
    }

    fn index_less_than(order: &[usize], a: &ArrayView1<IT>, b: &ArrayView1<IT>) -> bool {
        for &i in order.iter() {
            if a[i] >= b[i] {
                return false;
            }
        }
        true
    }

    fn swap_block(indices: &mut Array2<IT>, values: &mut ArrayViewMut2<VT>, a: usize, b: usize) {
        for offset in 0..indices.ncols() {
            unsafe {
                indices.uswap((a, offset), (b, offset));
            }
        }
        for offset in 0..values.ncols() {
            unsafe {
                values.uswap((a, offset), (b, offset));
            }
        }
    }
}
