mod axes;
mod builder;

pub use self::axes::{axes_to_string, map_axes, map_axes_ok, map_axes_unwrap, Axes};
pub use self::builder::AxisBuilder;

use crate::traits::IdxType;
use std::borrow::Cow;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct Axis<IT>
where
    IT: IdxType,
{
    pub(super) id: i64,
    pub(super) label: Option<String>,
    pub(super) range: Range<IT>,
}

impl<IT> Axis<IT>
where
    IT: IdxType,
{
    /// Returns the label of the axis.
    /// If no label was set, `None` is returned.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().label("x").range(0..10).build();
    /// assert_eq!(axis.label(), Some("x"));
    /// ```
    #[inline]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns the range of the axis.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// assert_eq!(axis.range(), 0..10);
    /// ```
    #[inline]
    pub fn range(&self) -> Range<IT> {
        self.range.clone()
    }

    /// Returns whether the upper bound is not greater than the lower bound.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(10..0).build();
    /// assert!(axis.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Returns the lower bound (inclusive) of the axis.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// assert_eq!(axis.lower(), 0);
    /// ```
    #[inline]
    pub fn lower(&self) -> IT {
        self.range.start
    }

    /// Returns the upper bound (exclusive) of the axis.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// assert_eq!(axis.upper(), 10);
    /// ```
    #[inline]
    pub fn upper(&self) -> IT {
        self.range.end
    }

    /// Returns the size of the axis.
    /// If the upper bound is not greater than the lower bound, `0` is returned.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// assert_eq!(axis.size(), 10);
    /// ```
    #[inline]
    pub fn size(&self) -> usize {
        if self.range.start < self.range.end {
            (self.range.end - self.range.start).to_usize().unwrap()
        } else {
            0
        }
    }

    /// Creates a new axis, modifies it with a new label.
    /// The new axis is **not equal** to the old one.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// let new_axis = axis.clone_with_label("y");
    /// assert_eq!(new_axis.label(), Some("y"));
    /// ```
    #[inline]
    #[must_use]
    pub fn clone_with_label<'a>(&'a self, label: impl Into<Cow<'a, str>>) -> Self {
        AxisBuilder::from(self).label(label).build()
    }

    /// Creates a new axis, modifies it with a new range.
    /// The new axis is **not equal** to the old one.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// let new_axis = axis.clone_with_range(0..20);
    /// assert_eq!(new_axis.range(), 0..20);
    /// ```
    #[inline]
    #[must_use]
    pub fn clone_with_range(&self, range: Range<IT>) -> Self {
        AxisBuilder::from(self).range(range).build()
    }

    /// Creates a new axis that covers both two old axes.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis1 = AxisBuilder::new().range(0..10).build();
    /// let axis2 = AxisBuilder::new().range(20..30).build();
    /// let new_axis = axis1.extend(&axis2);
    /// assert_eq!(new_axis.range(), 0..30);
    /// ```
    #[inline]
    #[must_use]
    pub fn extend(&self, other: &Self) -> Self {
        let self_start = self.range.start;
        let self_end = self.range.end;
        let other_start = other.range.start;
        let other_end = other.range.end;
        AxisBuilder::new()
            .range(self_start.min(other_start)..self_end.max(other_end))
            .build()
    }

    /// Creates a new named axis that covers both two old axes.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis1 = AxisBuilder::new().label("x").range(0..10).build();
    /// let axis2 = AxisBuilder::new().label("y").range(20..30).build();
    /// let new_axis = axis1.extend_with_label(&axis2, "z");
    /// assert_eq!(new_axis.range(), 0..30);
    /// ```
    #[inline]
    #[must_use]
    pub fn extend_with_label<'a>(&'a self, other: &Self, label: impl Into<Cow<'a, str>>) -> Self {
        let self_start = self.range.start;
        let self_end = self.range.end;
        let other_start = other.range.start;
        let other_end = other.range.end;
        AxisBuilder::new()
            .label(label)
            .range(self_start.min(other_start)..self_end.max(other_end))
            .build()
    }

    /// Creates a new axis that contains only the common parts of two old axes.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis1 = AxisBuilder::new().range(0..20).build();
    /// let axis2 = AxisBuilder::new().range(10..30).build();
    /// let new_axis = axis1.intersect(&axis2);
    /// assert_eq!(new_axis.range(), 10..20);
    ///
    /// let axis2 = AxisBuilder::new().range(30..40).build();
    /// let new_axis = axis1.intersect(&axis2);
    /// assert!(new_axis.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        let self_start = self.range.start;
        let self_end = self.range.end;
        let other_start = other.range.start;
        let other_end = other.range.end;
        AxisBuilder::new()
            .range(self_start.max(other_start)..self_end.min(other_end))
            .build()
    }

    /// Creates a named new axis that contains only the common parts of two old axes.
    ///
    /// ```
    /// use pattie::structs::axis::{Axis, AxisBuilder};
    ///
    /// let axis1 = AxisBuilder::new().label("x").range(0..20).build();
    /// let axis2 = AxisBuilder::new().label("y").range(10..30).build();
    /// let new_axis = axis1.intersect_with_label(&axis2, "z");
    /// assert_eq!(new_axis.range(), 10..20);
    ///
    /// let axis2 = AxisBuilder::new().label("y").range(30..40).build();
    /// let new_axis = axis1.intersect_with_label(&axis2, "z");
    /// assert!(new_axis.is_empty());
    /// ```
    #[inline]
    #[must_use]
    pub fn intersect_with_label<'a>(
        &'a self,
        other: &Self,
        label: impl Into<Cow<'a, str>>,
    ) -> Self {
        let self_start = self.range.start;
        let self_end = self.range.end;
        let other_start = other.range.start;
        let other_end = other.range.end;
        AxisBuilder::new()
            .label(label)
            .range(self_start.max(other_start)..self_end.min(other_end))
            .build()
    }
}

impl<IT> Display for Axis<IT>
where
    IT: IdxType,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(label) = self.label() {
            write!(f, "{}({}..{})", label, self.lower(), self.upper())
        } else {
            write!(f, "ax#{}({}..{})", self.id, self.lower(), self.upper())
        }
    }
}

impl<IT> PartialEq for Axis<IT>
where
    IT: IdxType,
{
    /// Compares two axes.
    ///
    /// Two axes are equal if and only if one of them is cloned from the other,
    /// and no modifications happens after the clone.
    ///
    /// This is to ensure axes across different tensors are tracked correctly.
    /// If you want to associate two axes into one, use [`Axis::extend`] or [`Axis::intersect`].
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<IT> Eq for Axis<IT> where IT: IdxType {}

impl<IT> Hash for Axis<IT>
where
    IT: IdxType,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<IT> From<Range<IT>> for Axis<IT>
where
    IT: IdxType,
{
    /// Creates a new axis with the given range.
    /// The range is half-inclusive, for example, `0..10` contains 0 but not 10.
    ///
    /// ```
    /// use pattie::structs::axis::Axis;
    ///
    /// let axis = Axis::from(0..10);
    /// ```
    #[inline]
    fn from(range: Range<IT>) -> Self {
        AxisBuilder::new().range(range).build()
    }
}
