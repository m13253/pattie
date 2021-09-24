//! Pattie is the Rust rework of the original ParTI! library.
//!
//! This library provides data structures and algorithms for sparse tensor operations.
//!
//! It consists of the following modules:
//! * `algos`: contains algorithms.
//! * `io`: contains input/output routines. But it is automatically imported when you import the `structs` module.
//! * `structs`: contains the data structures.
//! * `traits`: contains Rust traits for scalars, matrices, tensors, iterators, etc.
//!
//! This library is still in early development stage. The API is not stable yet.

pub mod algos;
pub mod io;
pub mod structs;
pub mod traits;
