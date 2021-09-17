use crate::structs::tensor;
use crate::traits;
use anyhow::Result;

pub struct COOMulCOO {}

impl COOMulCOO {
    pub fn new() -> Self {
        Self {}
    }

    pub fn mul<VT, IT>(
        a: &tensor::COO<VT, IT>,
        b: &tensor::COO<VT, IT>,
    ) -> Result<tensor::COO<VT, IT>>
    where
        VT: traits::ValType,
        IT: traits::IdxType,
    {
        todo!()
    }
}
