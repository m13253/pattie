//! This module contains I/O routines.
//!
//! This module is automatically imported when you import the `structs` module.
//!
//! Although this module contains code, the documentation browser shows empty. The actual contents are in the `structs` module.

mod read_coo;
mod write_coo;

pub use read_coo::*;
pub use write_coo::*;
