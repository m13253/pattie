use crate::algos::tensor::SortCOOTensor;
use crate::structs::{matrix, tensor};
use crate::traits::{IdxType, Tensor, ValType};
use anyhow::Result;
use ndarray::Axis;

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

    pub fn prepare<VT, IT>(&self, tsr: &mut tensor::COOTensor<VT, IT>) -> ()
    where
        VT: ValType,
        IT: IdxType,
    {
        let sort_task = SortCOOTensor::new().with_last_order(tsr.ndim(), self.common_axis);
        sort_task.execute(tsr);
    }

    pub fn execute<VT, IT>(
        &self,
        a: &tensor::COOTensor<VT, IT>,
        b: &matrix::DenseMatrix<VT>,
    ) -> Result<tensor::COOTensor<VT, IT>>
    where
        VT: ValType,
        IT: IdxType,
    {
        todo!()
    }
}
