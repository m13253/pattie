use super::Axis;
use crate::structs::vec::SmallVec;
use crate::traits::IdxType;
use thiserror::Error;

/// A special [`SmallVec`] that is used to store axes of tensors.
///
/// This vector allocates only if its size is greater than [`crate::structs::vec::SMALL_DIMS`].
pub type Axes<IT> = SmallVec<Axis<IT>>;

#[derive(Error, Debug)]
pub enum AxisMapError<IT>
where
    IT: IdxType,
{
    #[error("axis {} not found", axis)]
    AxisNotFound { axis: Axis<IT> },
}

/// Locate the index of each `from` axis in the array of `to`.
///
/// The returning iterator can be collected into a vector, and has the same length as `from`.
/// Panics if any axis is not found.
///
/// The operation takes O(n^2) time,
/// but considering the number of axes is small (3 or 4),
/// it shouldn't affect performance very much.
///
/// ```
/// use pattie::structs::axis::{AxisBuilder, map_axes};
///
/// let to = [
///     AxisBuilder::new().range(0..10).build(),
///     AxisBuilder::new().range(0..10).build(),
///     AxisBuilder::new().range(0..10).build(),
///     AxisBuilder::new().range(0..10).build(),
///     AxisBuilder::new().range(0..10).build(),
/// ];
/// let from = [
///     to[0].clone(), to[2].clone(), to[4].clone(),
/// ];
/// let map = map_axes(&from, &to).collect::<Result<Vec<_>, _>>().unwrap();
/// assert_eq!(&map, &[0, 2, 4]);
/// ```
#[inline]
pub fn map_axes<'a, IT>(
    from: &'a [Axis<IT>],
    to: &'a [Axis<IT>],
) -> impl Iterator<Item = Result<usize, AxisMapError<IT>>> + 'a
where
    IT: IdxType,
{
    from.iter().map(|axis| {
        to.iter()
            .position(|to_axis| to_axis == axis)
            .ok_or(AxisMapError::AxisNotFound { axis: axis.clone() })
    })
}

/// Same as [`map_axes`], but panics if any axis is not found.
///
/// ```
/// use pattie::structs::axis::{AxisBuilder, map_axes_unwrap};
///
/// let to = [
///     AxisBuilder::new().range(0..10).build(),
///     AxisBuilder::new().range(0..10).build(),
///     AxisBuilder::new().range(0..10).build(),
///     AxisBuilder::new().range(0..10).build(),
///     AxisBuilder::new().range(0..10).build(),
/// ];
/// let from = [
///     to[0].clone(), to[2].clone(), to[4].clone(),
/// ];
/// let map = map_axes_unwrap(&from, &to).collect::<Vec<_>>();
/// assert_eq!(&map, &[0, 2, 4]);
/// ```
#[inline]
pub fn map_axes_unwrap<'a, IT>(
    from: &'a [Axis<IT>],
    to: &'a [Axis<IT>],
) -> impl Iterator<Item = usize> + 'a
where
    IT: IdxType,
{
    map_axes(from, to).map(Result::unwrap)
}
