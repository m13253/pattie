use num::Num;
use std::fmt::{Debug, Display};

/// The type of a value inside a tensor.
///
/// Integers, floats, [`num::Complex`] are all supported types.
/// Moreover, any type that satisfies this trait can be used.
///
/// ```
/// use pattie::traits::ValType;
///
/// fn my_func(_x: impl ValType) {
/// }
/// my_func(42.0);
/// ```
pub trait ValType: Num + Clone + Debug + Display + Send + Sync {}
impl<T> ValType for T where T: Num + Clone + Debug + Display + Send + Sync + ?Sized {}
