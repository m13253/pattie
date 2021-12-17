use crate::structs::axis::Axes;
use crate::structs::tensor;
use crate::traits::{IdxType, Tensor, ValType};
use ndarray::{Array1, Array2, ArrayView1};

/// Task builder to sort the storage order of elements inside a `COOTensor`.
pub struct SortCOOTensor<IT>
where
    IT: IdxType,
{
    order: Axes<IT>,
}

impl<IT> SortCOOTensor<IT>
where
    IT: IdxType,
{
    /// Create a new `SortCOOTensor` task builder.
    /// Use `with_order` or `with_last_order` to specify the order of axis to sort tensors into.
    /// After configuring the task builder, use `execute` to perform the sort.
    pub fn new() -> Self {
        Self { order: Axes::new() }
    }

    /// Specify the order of axis to sort tensors into.
    pub fn with_order(mut self, order: ArrayView1<Axis>) -> Self {
        self.order = order.to_owned();
        for i in 0..order.len() {
            assert!(self.order.iter().find(|x| x.index() == i).is_some())
        }
        self
    }

    /// Perform the sorting.
    ///
    /// The sort operation uses quick-sort algorithm, but may change in future versions.
    ///
    /// # Example
    /// ```
    /// use ndarray::{ArrayView1, Axis};
    /// use pattie::structs::tensor::COOTensor;
    /// use pattie::algos::tensor::SortCOOTensor;
    ///
    /// let mut tensor = COOTensor::<f32, usize>::zeros(vec![3, 2, 3, 4]);
    /// let task = SortCOOTensor::new().with_last_order(4, Axis(2));
    /// task.execute(&mut tensor);
    /// ```
    pub fn execute<IT, VT>(&self, tensor: &mut tensor::COOTensor<IT, VT>)
    where
        IT: IdxType,
        VT: ValType,
    {
        fn index_less_than<IT, VT>(
            order: &Array1<Axis>,
            a: &ArrayView1<IT>,
            b: &ArrayView1<IT>,
        ) -> bool
        where
            VT: ValType,
            IT: IdxType,
        {
            for i in order.iter() {
                if a[i.index()] >= b[i.index()] {
                    return false;
                }
            }
            true
        }

        fn sort_subtensor<IT, VT>(
            indices: &mut Array2<IT>,
            values: &mut Array1<VT>,
            order: &Array1<Axis>,
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
                while index_less_than::<IT, VT>(&order, &indices.row(i), &indices.row(pivot)) {
                    i += 1;
                }
                while index_less_than::<IT, VT>(&order, &indices.row(j), &indices.row(pivot)) {
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
                    unsafe {
                        values.uswap(i, j);
                    }
                    i += 1;
                    j -= 1;
                }
            }
            sort_subtensor(indices, values, order, from, j);
            sort_subtensor(indices, values, order, i, to);
        }

        assert_eq!(tensor.ndim(), self.order.len());
        unsafe { tensor.set_sort_order() }.clone_from(&self.order);
        let (indices, values) = unsafe { tensor.raw_data_mut() };

        sort_subtensor(indices, values, &self.order, 0, values.len());
    }
}
