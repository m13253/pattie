//! Write a tensor to a text file.
//!
//! The real documentation is at [`pattie::structs::tensor::COOTensor`].
//!
//! [`pattie::structs::tensor::COOTensor`]: ../../structs/tensor/struct.COOTensor.html#method.write_to_text

use crate::structs::tensor;
use crate::traits::{IdxType, Tensor, ValType};
use std::fmt;
use std::io;
use std::io::Write;
use streaming_iterator::StreamingIterator;

impl<IT, VT> tensor::COOTensor<IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    /// Write the tensor to a text file.
    ///
    /// # Arguments
    ///
    /// * `w` - A [`std::io::Write`] object. Examples are [`std::fs::File`], [`std::io::Stdout`], [`Vec<u8>`].
    ///
    /// # Example output
    ///
    /// ```text
    /// 3
    /// 0 0 0
    /// 3 2 3
    /// 0       0       0       1.000000e+00
    /// 0       0       1       2.000000e+00
    /// 0       1       0       3.000000e+00
    /// 0       1       2       4.000000e+00
    /// 1       0       2       5.000000e+00
    /// 1       1       0       6.000000e+00
    /// 2       0       1       7.000000e+00
    /// 2       1       1       8.000000e+00
    /// ```
    ///
    /// First line is the number of axes.
    /// Second line is the lower bound of each axis (inclusive).
    /// Third line is the upper bound of each axis (exclusive).
    /// The following lines are the elements of the tensor.
    #[inline]
    pub fn write_to_text<W>(&self, w: &mut W) -> io::Result<()>
    where
        W: io::Write,
        VT: fmt::LowerExp, // We can't specialize yet: https://github.com/rust-lang/rust/issues/31844
    {
        self.write_to_text_with_formatter(w, |value| format!("{:.6e}", value))
    }

    /// Similar to [`tensor::COOTensor::write_to_text`], but with a custom formatter for values.
    pub fn write_to_text_with_formatter<W, F>(&self, w: &mut W, formatter: F) -> io::Result<()>
    where
        W: io::Write,
        F: FnMut(&VT) -> String,
    {
        let mut formatter = formatter;
        let mut w = io::BufWriter::new(w);

        if let Some(name) = self.name() {
            writeln!(w, "# {}", name)?;
        }

        writeln!(w, "{}", self.ndim())?;
        if self.ndim() == 0 {
            return Ok(());
        }

        if self.shape().iter().any(|axis| axis.label().is_some()) {
            for (i, axis) in self.shape().iter().enumerate() {
                if let Some(label) = axis.label() {
                    writeln!(w, "# Axis {}: {}", i, label)?;
                } else {
                    writeln!(w, "# Axis {} has no label", i)?;
                }
            }
        }

        let mut shape_iter = self.shape().iter();
        if let Some(axis) = shape_iter.next() {
            write!(w, "{}", axis.lower())?;
        }
        for axis in shape_iter {
            write!(w, " {}", axis.lower())?;
        }
        writeln!(w)?;

        let mut shape_iter = self.shape().iter();
        if let Some(axis) = shape_iter.next() {
            write!(w, "{}", axis.upper())?;
        }
        for axis in shape_iter {
            write!(w, " {}", axis.upper())?;
        }
        writeln!(w)?;

        let mut tensor_iter = self.iter();
        while let Some(&(index, value)) = tensor_iter.next() {
            for index in index.iter() {
                write!(w, "{}\t", index)?;
            }
            write!(w, "{}", formatter(value))?;
            writeln!(w)?;
        }

        Ok(())
    }
}
