use super::AxisBuilder;
use crate::traits::IdxType;
use std::borrow::Cow;
use std::hash::{Hash, Hasher};
use std::marker::PhantomPinned;
use std::ops::Range;
use std::sync::Arc;

#[derive(Clone)]
pub struct Axis<IT = isize>
where
    IT: IdxType,
{
    pub(super) inner: Arc<AxisInner<IT>>,
}

#[derive(Clone)]
pub(super) struct AxisInner<IT>
where
    IT: IdxType,
{
    pub(super) label: Option<String>,
    pub(super) range: Range<IT>,
    pub(super) pin: PhantomPinned,
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
    pub fn label(&self) -> Option<&str> {
        self.inner.label.as_deref()
    }

    /// Returns the range of the axis.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// assert_eq!(axis.range(), 0..10);
    /// ```
    pub fn range(&self) -> Range<IT> {
        self.inner.range.clone()
    }

    /// Returns whether the upper bound is not greater than the lower bound.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(10..0).build();
    /// assert!(axis.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.inner.range.is_empty()
    }

    /// Returns the lower bound (inclusive) of the axis.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// assert_eq!(axis.lower(), 0);
    /// ```
    pub fn lower(&self) -> IT {
        self.inner.range.start.clone()
    }

    /// Returns the upper bound (exclusive) of the axis.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// assert_eq!(axis.upper(), 10);
    /// ```
    pub fn upper(&self) -> IT {
        self.inner.range.end.clone()
    }

    /// Returns the length of the axis.
    /// If the upper bound is not greater than the lower bound, `0` is returned.
    ///
    /// ```
    /// use pattie::structs::axis::AxisBuilder;
    ///
    /// let axis = AxisBuilder::new().range(0..10).build();
    /// assert_eq!(axis.len(), 10);
    /// ```
    pub fn len(&self) -> IT {
        if self.inner.range.start < self.inner.range.end {
            self.inner.range.end.clone() - self.inner.range.start.clone()
        } else {
            IT::zero()
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
    pub fn extend(&self, other: &Self) -> Self {
        let self_start = self.inner.range.start.clone();
        let self_end = self.inner.range.end.clone();
        let other_start = other.inner.range.start.clone();
        let other_end = other.inner.range.end.clone();
        AxisBuilder::new()
            .range(self_start.min(other_start)..self_end.max(other_end.clone()))
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
    pub fn extend_with_label<'a>(&'a self, other: &Self, label: impl Into<Cow<'a, str>>) -> Self {
        let self_start = self.inner.range.start.clone();
        let self_end = self.inner.range.end.clone();
        let other_start = other.inner.range.start.clone();
        let other_end = other.inner.range.end.clone();
        AxisBuilder::new()
            .label(label)
            .range(self_start.min(other_start.clone())..self_end.clone().max(other_end.clone()))
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
    pub fn intersect(&self, other: &Self) -> Self {
        let self_start = self.inner.range.start.clone();
        let self_end = self.inner.range.end.clone();
        let other_start = other.inner.range.start.clone();
        let other_end = other.inner.range.end.clone();
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
    pub fn intersect_with_label<'a>(
        &'a self,
        other: &Self,
        label: impl Into<Cow<'a, str>>,
    ) -> Self {
        let self_start = self.inner.range.start.clone();
        let self_end = self.inner.range.end.clone();
        let other_start = other.inner.range.start.clone();
        let other_end = other.inner.range.end.clone();
        AxisBuilder::new()
            .label(label)
            .range(self_start.max(other_start)..self_end.min(other_end))
            .build()
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
    fn eq(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.inner) == Arc::as_ptr(&other.inner)
    }
}

impl<IT> Eq for Axis<IT> where IT: IdxType {}

impl<IT> Hash for Axis<IT>
where
    IT: IdxType,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.inner).hash(state);
    }
}

impl<IT> From<IT> for Axis<IT>
where
    IT: IdxType,
{
    /// Creates a new axis with a range `0..upper`.
    /// The range is half-inclusive, for example, `0..10` contains 0 but not 10.
    ///
    /// ```
    /// use pattie::structs::axis::Axis;
    ///
    /// let axis = Axis::from(10);
    /// assert_eq!(axis.range(), 0..10);
    /// ```
    fn from(upper: IT) -> Self {
        AxisBuilder::new().range(IT::zero()..upper).build()
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
    fn from(range: Range<IT>) -> Self {
        AxisBuilder::new().range(range).build()
    }
}
