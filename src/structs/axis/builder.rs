use super::Axis;
use crate::traits::IdxType;
use std::borrow::Cow;
use std::ops::Range;
use std::sync::atomic::{AtomicI64, Ordering};

/// A builder for [`Axis`].
#[derive(Clone, Debug)]
pub struct AxisBuilder<'a, IT = isize>
where
    IT: IdxType,
{
    label: Option<Cow<'a, str>>,
    range: Option<Range<IT>>,
}

static mut COUNTER: AtomicI64 = AtomicI64::new(0);

impl<'a, IT> AxisBuilder<'a, IT>
where
    IT: IdxType,
{
    /// Creates a new builder for [`Axis`].
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the label of the axis (optional).
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().label("x").range(0..10).build();
    /// assert_eq!(axis.label(), Some("x"));
    /// ```
    #[inline]
    pub fn label(self, label: impl Into<Cow<'a, str>>) -> Self {
        Self {
            label: Some(label.into()),
            ..self
        }
    }

    /// Sets the range of the axis.
    /// The range is half-inclusive, for example, `0..10` contains 0 but not 10.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// assert_eq!(axis.range(), 0..10);
    /// ```
    #[inline]
    pub fn range(self, range: Range<IT>) -> Self {
        Self {
            range: Some(range),
            ..self
        }
    }

    /// Builds the [`Axis`].
    ///
    /// Panics if the range is not set.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().label("x").range(0..10).build();
    /// assert_eq!(axis.label(), Some("x"));
    /// assert_eq!(axis.range(), 0..10);
    /// ```
    #[inline]
    pub fn build(self) -> Axis<IT> {
        Axis {
            // # Safety
            // `COUNTER` is a static variable, accessing it requires an unsafe block.
            // This access is safe because we are using an atomic operation.
            id: unsafe { COUNTER.fetch_add(1, Ordering::Relaxed) },
            label: self.label.map(Cow::into_owned),
            range: self.range.expect("range not set"),
        }
    }
}

impl<'a, IT> From<&'a Axis<IT>> for AxisBuilder<'a, IT>
where
    IT: IdxType,
{
    /// Creates a new builder from an existing [`Axis`].
    /// The new builder will have the same label and range.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis1 = AxisBuilder::new().label("x").range(0..10).build();
    /// let axis2 = AxisBuilder::from(&axis1).label("y").build();
    /// assert_eq!(axis2.label(), Some("y"));
    /// assert_eq!(axis2.range(), 0..10);
    /// ```
    #[inline]
    fn from(axis: &'a Axis<IT>) -> Self {
        Self {
            label: axis.label().map(Cow::Borrowed),
            range: Some(axis.range()),
        }
    }
}

impl<'a, IT> From<Range<IT>> for AxisBuilder<'a, IT>
where
    IT: IdxType,
{
    /// Creates a new builder with the given range.
    /// The range is half-inclusive, for example, `0..10` contains 0 but not 10.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::from(0..10).build();
    /// ```
    #[inline]
    fn from(range: Range<IT>) -> Self {
        AxisBuilder::new().range(range)
    }
}

impl<'a, IT> From<IT> for AxisBuilder<'a, IT>
where
    IT: IdxType,
{
    /// Creates a new builder with a range `0..upper`.
    /// The range is half-inclusive, for example, `0..10` contains 0 but not 10.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::from(10).build();
    /// assert_eq!(axis.range(), 0..10);
    /// ```
    #[inline]
    fn from(upper: IT) -> Self {
        AxisBuilder::new().range(IT::zero()..upper)
    }
}

impl<'a, IT> Default for AxisBuilder<'a, IT>
where
    IT: IdxType,
{
    #[inline]
    fn default() -> Self {
        Self {
            label: Option::default(),
            range: Option::default(),
        }
    }
}
