use crate::algos::tensor::SortCOOTensor;
use crate::structs::matrix::DenseMatrix;
use crate::structs::tensor::COOTensor;
use crate::traits::{IdxType, Tensor, ValType};
use anyhow::Result;
use ndarray::{Axis, Dimension};

pub struct COOTensorMulDenseMatrix {
    common_axis: Axis,
}

impl COOTensorMulDenseMatrix {
    pub fn new() -> Self {
        Self {
            common_axis: Axis(0),
        }
    }

    pub fn with_common_axis(mut self, common_axis: Axis) -> Self {
        self.common_axis = common_axis;
        self
    }

    pub fn prepare<VT, IT>(&self, tsr: &mut COOTensor<VT, IT>)
    where
        VT: ValType,
        IT: IdxType,
    {
        let sort_task = SortCOOTensor::new().with_last_order(tsr.ndim(), self.common_axis);
        sort_task.execute(tsr);
    }

    pub fn execute<VT, IT>(
        &self,
        a: &COOTensor<VT, IT>,
        b: &DenseMatrix<VT>,
    ) -> Result<COOTensor<VT, IT>>
    where
        VT: ValType,
        IT: IdxType,
    {
        let mut c_dim = a.raw_dim();
        c_dim.as_array_view_mut()[self.common_axis.index()] = b.nrows();

        let mut c = COOTensor::<VT, IT>::zeros(c_dim);

        todo!();

        Ok(c)
    }
}
