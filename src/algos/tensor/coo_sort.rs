use crate::structs::axis::{map_axes_unwrap, Axes, Axis};
use crate::structs::tensor::COOTensor;
use crate::structs::vec::SmallVec;
use crate::traits::{IdxType, RawParts, ValType};
use ndarray::{Array2, ArrayD, ArrayView1, IxDyn};
use std::mem;

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
        Default::default()
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
    ///
    /// # Example
    /// ```
    /// use pattie::structs::tensor::COOTensor;
    /// use pattie::algos::tensor::SortCOOTensor;
    ///
    /// let mut tensor = COOTensor::<f32, usize>::zeros(vec![3, 2, 3, 4]);
    /// let axes = tensor.sparse_axes();
    /// let order = [0, 1, 3, 2].iter().map(|i| axes[i].clone()).collect::<Vec<_>>();
    /// let task = SortCOOTensor::new().with_order(tensor, &order);
    /// task.execute();
    /// ```
    pub fn execute(self) {
        let tensor = self.tensor.unwrap();
        let raw_parts = unsafe { tensor.raw_parts_mut() };
        let order = self.order.as_ref().unwrap().as_slice();
        let order_index = map_axes_unwrap(order, &raw_parts.sparse_axes).collect::<SmallVec<_>>();

        let num_blocks = raw_parts.indices.nrows();
        let values_shape = raw_parts.values.raw_dim();
        let values_2d_shape = (
            raw_parts.values.shape()[0],
            raw_parts.values.shape()[1..].iter().product::<usize>(),
        );
        assert_eq!(num_blocks, values_2d_shape.0);
        let mut values_2d = mem::replace(&mut raw_parts.values, ArrayD::zeros(IxDyn(&[])))
            .into_shape(values_2d_shape)
            .unwrap();

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

        fn sort_subtensor<IT, VT>(
            indices: &mut Array2<IT>,
            values: &mut Array2<VT>,
            order: &[usize],
            from: usize,
            to: usize,
        ) where
            VT: ValType,
            IT: IdxType,
        {
            if from >= to {
                return;
            }
            let mut pivot = (from + to) / 2;
            let mut i = from;
            let mut j = to - 1;
            while i < j {
                while index_less_than(order, &indices.row(i), &indices.row(pivot)) {
                    i += 1;
                }
                while index_less_than(order, &indices.row(j), &indices.row(pivot)) {
                    j -= 1;
                }
                if i < j {
                    if pivot == i {
                        pivot = j;
                    } else if pivot == j {
                        pivot = i;
                    }
                    for d in 0..indices.ncols() {
                        unsafe {
                            indices.uswap((i, d), (j, d));
                        }
                    }
                    for v in 0..values.ncols() {
                        unsafe {
                            values.uswap((i, v), (j, v));
                        }
                    }
                    i += 1;
                    j -= 1;
                }
            }
            sort_subtensor(indices, values, order, from, j);
            sort_subtensor(indices, values, order, i, to);
        }

        sort_subtensor(
            &mut raw_parts.indices,
            &mut values_2d,
            &order_index,
            0,
            num_blocks,
        );

        raw_parts.values = values_2d.into_shape(values_shape).unwrap();
    }
}

impl<'a, IT, VT> Default for SortCOOTensor<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    fn default() -> Self {
        Self {
            tensor: None,
            order: None,
        }
    }
}
