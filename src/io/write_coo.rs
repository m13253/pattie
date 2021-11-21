//! Write a tensor to a text file.
//!
//! The real documentation is at [`pattie::structs::tensor::COOTensor`].
//!
//! [`pattie::structs::tensor::COOTensor`]: ../../structs/tensor/struct.COOTensor.html#method.write_to_text

use crate::structs::tensor;
use crate::traits::{IdxType, Tensor, ValType};
use std::fmt;
use std::io::{self, Write};

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
    pub fn write_to_text<W>(&self, w: &mut W) -> io::Result<()>
    where
        W: io::Write,
        VT: fmt::LowerExp, // We can't specialize yet: https://github.com/rust-lang/rust/issues/31844
    {
        self.write_to_text_with_formatter(w, |value| format!("{:.6e}", value))
    }

    /// Similar to [`write_to_text`], but with a custom formatter for values.
    pub fn write_to_text_with_formatter<W, F>(&self, w: &mut W, formatter: F) -> io::Result<()>
    where
        W: io::Write,
        F: FnMut(&VT) -> String,
    {
        let mut formatter = formatter;
        let mut buf_writer = io::BufWriter::new(w);

        writeln!(buf_writer, "{}", self.ndim())?;
        if self.ndim() == 0 {
            return Ok(());
        }

        let mut shape_iter = self.shape().into_iter();
        if let Some(axis) = shape_iter.next() {
            write!(buf_writer, "{}", axis.lower())?;
        }
        for axis in shape_iter {
            write!(buf_writer, " {}", axis.lower())?;
        }
        writeln!(buf_writer)?;

        let mut shape_iter = self.shape().into_iter();
        if let Some(axis) = shape_iter.next() {
            write!(buf_writer, "{}", axis.upper())?;
        }
        for axis in shape_iter {
            write!(buf_writer, " {}", axis.upper())?;
        }
        writeln!(buf_writer)?;

        for (index, value) in self.iter() {
            for index in index.iter() {
                write!(buf_writer, "{}\t", index)?;
            }
            write!(buf_writer, "{}", formatter(value))?;
            writeln!(buf_writer)?;
        }

        Ok(())
    }
}
