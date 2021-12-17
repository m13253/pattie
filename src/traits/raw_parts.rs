/// RawParts is a trait that allows a user to access the internals of a data structure,
/// which would otherwise be hidden by the public API.
///
/// Since a lot of algorithms relies on directly accessing private fields,
/// this mechanism can allow this while keeping the data integrity.
///
/// Be aware that the raw parts of a data structure do not guarantee API stability.
/// It is very common to break the API compatibility across multiple versions.
pub trait RawParts {
    type Inner: ?Sized;

    /// Construct a data structure from its raw parts.
    ///
    /// # Safety
    /// The caller must ensure that the data structure is valid.
    unsafe fn from_raw_parts(raw_parts: Self::Inner) -> Self;

    /// Extract the raw parts from the data structure.
    /// The original data structure is consumed.
    ///
    /// This way, the caller can access the internals of the data structure,
    /// which would otherwise be hidden by the public API.
    fn into_raw_parts(self) -> Self::Inner;

    /// Get a reference to the raw parts of a data structure.
    ///
    /// This way, the caller can access the internals of the data structure,
    /// which would otherwise be hidden by the public API.
    fn raw_parts(&self) -> &Self::Inner;

    /// Get a mutable reference to the raw parts of a data structure.
    ///
    /// This way, the caller can access the internals of the data structure,
    /// which would otherwise be hidden by the public API.
    ///
    /// # Safety
    /// The caller must keep the data integrity upon returning the reference.
    unsafe fn raw_parts_mut(&mut self) -> &mut Self::Inner;
}
