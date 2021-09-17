use crate::traits;
use crate::{structs::tensor, traits::Tensor};
use ndarray::{Array1, Array2, ArrayView1, Axis};

pub struct COOSort {
    order: Array1<Axis>,
}

impl COOSort {
    pub fn new() -> COOSort {
        COOSort {
            order: Array1::from_elem(0, Axis(0)),
        }
    }

    pub fn with_order(mut self, order: ArrayView1<Axis>) -> COOSort {
        self.order = order.to_owned();
        for i in 0..order.len() {
            assert!(self.order.iter().find(|x| x.index() == i).is_some())
        }
        self
    }

    pub fn with_last_order(mut self, ndim: usize, last_order: Axis) -> COOSort {
        assert!(last_order.index() < ndim);
        self.order = Array1::from_iter((0..ndim).map(|i| {
            if i + 1 == ndim {
                last_order
            } else if i < last_order.index() {
                Axis(i)
            } else {
                Axis(i + 1)
            }
        }));
        self
    }

    pub fn execute<VT, IT>(&self, tensor: &mut tensor::COOTensor<VT, IT>)
    where
        VT: traits::ValType,
        IT: traits::IdxType,
    {
        fn index_less_than<VT, IT>(
            order: &ArrayView1<Axis>,
            a: &ArrayView1<IT>,
            b: &ArrayView1<IT>,
        ) -> bool
        where
            VT: traits::ValType,
            IT: traits::IdxType,
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
            order: ArrayView1<Axis>,
            from: usize,
            to: usize,
        ) where
            VT: traits::ValType,
            IT: traits::IdxType,
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
        let (indices, values) = unsafe { tensor.raw_data_mut() };

        sort_subtensor(indices, values, self.order.view(), 0, values.len());
    }
}
