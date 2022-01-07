//! Algorithms related to vectors.

pub mod reorder;

pub use reorder::{
    reorder_backward, reorder_backward_inplace, reorder_forward, reorder_forward_inplace,
};
