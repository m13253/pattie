pub use smallvec::smallvec;

/// The number of dimensions we consider enough for our [`SmallVec`].
pub const SMALL_DIMS: usize = 4;

/// Re-export [`smallbec::SmallVec`] with a fixed size of [`SMALL_DIMS`].
pub type SmallVec<T> = smallvec::SmallVec<[T; SMALL_DIMS]>;
