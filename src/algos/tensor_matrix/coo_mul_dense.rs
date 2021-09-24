use crate::algos::tensor::SortCOOTensor;
use crate::structs::matrix::DenseMatrix;
use crate::structs::tensor::COOTensor;
use crate::traits::{IdxType, Tensor, ValType};
use anyhow::Result;
use ndarray::{Axis, Dimension};

/// Task builder to multiply a `COOTensor` with a `DenseMatrix`.
pub struct COOTensorMulDenseMatrix {
    common_axis: Option<Axis>,
}

impl COOTensorMulDenseMatrix {
    /// Create a new `COOTensorMulDenseMatrix` task builder.
    /// Use `with_common_axis` to specify the common axis of multiplication.
    pub fn new() -> Self {
        Self { common_axis: None }
    }

    /// Set the common axis of multiplication.
    pub fn with_common_axis(mut self, common_axis: Axis) -> Self {
        self.common_axis = Some(common_axis);
        self
    }

    /// Sort the tensor according to the common axis.
    ///
    /// This method modifies element order of the tensor.
    pub fn prepare<VT, IT>(&self, tsr: &mut COOTensor<VT, IT>)
    where
        VT: ValType,
        IT: IdxType,
    {
        let common_axis = self.common_axis.expect("common_axis not set");
        let sort_task = SortCOOTensor::new().with_last_order(tsr.ndim(), common_axis);
        sort_task.execute(tsr);
    }

    /// Multiply a `COOTensor` with a `DenseMatrix`.
    ///
    /// This method assumes that the tensor is already sorted. Otherwise, call `prepare` first.
    pub fn execute<VT, IT>(
        &self,
        a: &COOTensor<VT, IT>,
        b: &DenseMatrix<VT>,
    ) -> Result<COOTensor<VT, IT>>
    where
        VT: ValType,
        IT: IdxType,
    {
        let common_axis = self.common_axis.expect("common_axis not set");
        let sort_order = a
            .sort_order()
            .expect("tensor is not sorted prior to multiplication");
        if sort_order.len() != 0 {
            assert_eq!(sort_order.get(sort_order.len() - 1), Some(&common_axis));
        }

        let mut c_dim = a.raw_dim();
        c_dim.as_array_view_mut()[common_axis.index()] = b.nrows();

        let mut _c = COOTensor::<VT, IT>::zeros(c_dim);

        todo!()
    }
}
