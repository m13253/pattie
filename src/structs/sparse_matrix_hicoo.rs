use crate::traits;
use ndarray::Ix2;
use std::marker::PhantomData;

pub struct SparseMatrixHiCOO<
    ValType: traits::ValType,
    IdxType: traits::IdxType,
    const NUM_BITS_BLOCK: usize,
    const NUM_BITS_SUPERBLOCK: usize,
> {
    dim: Ix2,
    phantom: PhantomData<(ValType, IdxType)>,
}
