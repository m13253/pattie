use num::Float;
use num::Integer;

pub struct Matrix<T: Float> {
    pub rows: usize,
    pub cols: usize,
    pub stride: usize,
    pub data: Vec<T>,
}

pub struct SparseMatrix<ValType: Float, IdxType: Integer> {
    pub rows: usize,
    pub cols: usize,
    pub idx_rows: Vec<IdxType>,
    pub idx_cols: Vec<IdxType>,
    pub val_non_zeros: Vec<ValType>,
}

pub struct SparseMatrixHiCOO<
    ValType: Float,
    // IdxType: Integer,
    const NUM_BITS_BLOCK: usize,
    const NUM_BITS_SUPERBLOCK: usize,
> {
    pub rows: usize,
    pub cols: usize,
    pub num_non_zeros: usize,

    // TODO
    pub val_non_zeros: Vec<ValType>,
}
