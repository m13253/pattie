use crate::structs::axis::{map_axes_unwrap, Axes, Axis};
use crate::structs::tensor::COOTensor;
use crate::structs::vec::SmallVec;
use crate::traits::{IdxType, RawParts, ValType};
use ndarray::{Array2, ArrayView1, ArrayViewMut2};

/// Task builder to sort the storage order of elements inside a `COOTensor`.
pub struct SortCOOTensor<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    tensor: Option<&'a mut COOTensor<IT, VT>>,
    order: Option<Axes<IT>>,
}

impl<'a, IT, VT> SortCOOTensor<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    /// Create a new `SortCOOTensor` task builder.
    /// Use `prepare` to specify the order of axis to sort tensors into.
    /// After configuring the task builder, use `execute` to perform the sort.
    pub fn new() -> Self {
        Self {
            tensor: None,
            order: None,
        }
    }

    /// Specify the order of axis to sort tensors into.
    pub fn prepare(mut self, tensor: &'a mut COOTensor<IT, VT>, order: &[Axis<IT>]) -> Self {
        assert_eq!(tensor.sparse_axes().len(), order.len());
        self.tensor = Some(tensor);
        self.order = Some(SmallVec::from(order));
        self
    }

    /// Perform the sorting.
    ///
    /// The sort operation uses quick-sort algorithm, but may change in future versions.
    pub fn execute(self) {
        let tensor = self.tensor.unwrap();
        let raw_parts = unsafe { tensor.raw_parts_mut() };
        let order = self.order.as_ref().unwrap().as_slice();
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
        sort_subtensor(
            &mut raw_parts.indices,
            &mut values_2d,
            &order_index,
            0,
            num_blocks,
        );

        // Mark the tensor as sorted.
        raw_parts.sparse_sort_order = self.order.unwrap().clone();
        raw_parts.sparse_is_sorted = true;

        fn sort_subtensor<IT, VT>(
            indices: &mut Array2<IT>,
            values: &mut ArrayViewMut2<VT>,
            order: &[usize],
            from: usize,
            to: usize,
        ) where
            IT: IdxType,
            VT: ValType,
        {
            if to - from < 2 {
                return;
            }
            let mut pivot = (from + to) / 2;
            let mut i = from;
            let mut j = to - 1;
            loop {
                while index_less_than(order, &indices.row(i), &indices.row(pivot)) {
                    i += 1;
                }
                while index_less_than(order, &indices.row(j), &indices.row(pivot)) {
                    j -= 1;
                }
                if i >= j {
                    break;
                }
                swap_block(indices, values, i, j);
                if i == pivot {
                    pivot = j;
                } else if j == pivot {
                    pivot = i;
                }
                i += 1;
                j -= 1;
            }
            sort_subtensor(indices, values, order, from, j);
            sort_subtensor(indices, values, order, i, to);
        }

        fn index_less_than<IT>(order: &[usize], a: &ArrayView1<IT>, b: &ArrayView1<IT>) -> bool
        where
            IT: IdxType,
        {
            for &i in order.iter() {
                if a[i] >= b[i] {
                    return false;
                }
            }
            true
        }

        fn swap_block<IT, VT>(
            indices: &mut Array2<IT>,
            values: &mut ArrayViewMut2<VT>,
            a: usize,
            b: usize,
        ) where
            IT: IdxType,
            VT: ValType,
        {
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
}

impl<'a, IT, VT> Default for SortCOOTensor<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    fn default() -> Self {
        Self::new()
    }
}
