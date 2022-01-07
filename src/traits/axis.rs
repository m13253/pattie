use super::IdxType;
use crate::structs::axis::Axis;

/// Represents a data type that can be converted into an [`Axis`].
///
/// This provides a shorthand if you only want to create a zero-indexed anonymous axis.
pub trait IntoAxis<IT>: Into<Axis<IT>>
where
    IT: IdxType,
{
    fn into_axis(self) -> Axis<IT> {
        self.into()
    }
}

impl<IT, T> IntoAxis<IT> for T
where
    IT: IdxType,
    Axis<IT>: From<T>,
{
}
