pub use smallvec::smallvec;

pub const SMALL_DIMS: usize = 4;
pub type SmallVec<T> = smallvec::SmallVec<[T; SMALL_DIMS]>;
