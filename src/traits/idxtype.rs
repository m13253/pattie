use num::{Integer, NumCast};
use std::fmt::{Debug, Display};

/// The type of indices used in a tensor.
///
/// [`u16`], [`u32`], [`u64`], [`usize`] are all supported types.
/// However, if the scale of the tensor is too large, narrow types may cause overflow and trigger panics.
///
/// ```
/// use pattie::traits::IdxType;
///
/// fn my_func(_x: impl IdxType) {
/// }
/// my_func(42);
/// ```
pub trait IdxType: Integer + NumCast + Clone + Debug + Display + Send + Sync {}
impl<T> IdxType for T where T: Integer + NumCast + Clone + Debug + Display + Send + Sync {}
