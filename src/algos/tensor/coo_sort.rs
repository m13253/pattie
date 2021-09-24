use crate::structs::tensor;
use crate::traits::{IdxType, Tensor, ValType};
use ndarray::{Array1, Array2, ArrayView1, Axis};

/// Task builder to sort the storage order of elements inside a `COOTensor`.
pub struct SortCOOTensor {
    order: Array1<Axis>,
}

impl SortCOOTensor {
    /// Create a new `SortCOOTensor` task builder.
    /// Use `with_order` or `with_last_order` to specify the order of axis to sort tensors into.
    /// After configuring the task builder, use `execute` to perform the sort.
    pub fn new() -> Self {
        Self {
            order: Array1::from_elem(0, Axis(0)),
        }
    }

    /// Specify the order of axis to sort tensors into.
    ///
    /// # Example
    /// ```
    /// use ndarray::{ArrayView1, Axis};
    /// use pattie::structs::tensor::COOTensor;
    /// use pattie::algos::tensor::SortCOOTensor;
    ///
    /// let mut tensor = COOTensor::<f32, usize>::zeros(vec![3, 2, 3, 4]);
    /// let task = SortCOOTensor::new().with_order(ArrayView1::from(&[
    ///     Axis(0), Axis(2), Axis(3), Axis(1),
    /// ]));
    /// task.execute(&mut tensor);
    /// ```
    pub fn with_order(mut self, order: ArrayView1<Axis>) -> Self {
        self.order = order.to_owned();
        for i in 0..order.len() {
            assert!(self.order.iter().find(|x| x.index() == i).is_some())
        }
        self
    }

    /// `with_last_order` is a shortcut to `with_order` to put `last_axis` at the end, while keeping other axes in increasing order.
    ///
    /// # Example
    /// ```
    /// use ndarray::{ArrayView1, Axis};
    /// use pattie::algos::tensor::SortCOOTensor;
    ///
    /// let task1 = SortCOOTensor::new().with_last_order(5, Axis(2));
    /// let task2 = SortCOOTensor::new().with_order(ArrayView1::from(&[
    ///     Axis(0), Axis(2), Axis(3), Axis(4), Axis(2),
    /// ]));
    /// // task1 is equivalent to task2.
    /// ```
    pub fn with_last_order(mut self, ndim: usize, last_axis: Axis) -> Self {
        assert!(last_axis.index() < ndim);
        self.order = Array1::from_iter((0..ndim).map(|i| {
            if i + 1 == ndim {
                last_axis
            } else if i < last_axis.index() {
                Axis(i)
            } else {
                Axis(i + 1)
            }
        }));
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
    pub fn execute<VT, IT>(&self, tensor: &mut tensor::COOTensor<VT, IT>)
    where
        VT: ValType,
        IT: IdxType,
    {
        fn index_less_than<VT, IT>(
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

        fn sort_subtensor<VT, IT>(
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
                while index_less_than::<VT, IT>(&order, &indices.row(i), &indices.row(pivot)) {
                    i += 1;
                }
                while index_less_than::<VT, IT>(&order, &indices.row(j), &indices.row(pivot)) {
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
