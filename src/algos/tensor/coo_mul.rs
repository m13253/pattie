use crate::structs::tensor;
use crate::traits;
use anyhow::Result;

pub struct COOMulCOO {}

impl COOMulCOO {
    pub fn new() -> COOMulCOO {
        COOMulCOO {}
    }

    pub fn execute<VT, IT>(
        a: &tensor::COOTensor<VT, IT>,
        b: &tensor::COOTensor<VT, IT>,
    ) -> Result<tensor::COOTensor<VT, IT>>
    where
        VT: traits::ValType,
        IT: traits::IdxType,
    {
        todo!()
    }
}
