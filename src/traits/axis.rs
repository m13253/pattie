use super::IdxType;
use crate::structs::axis::Axis;

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
