use num::{Integer, Num, NumCast};

pub trait ValType: Num {}

pub trait IdxType: Integer + NumCast {}
