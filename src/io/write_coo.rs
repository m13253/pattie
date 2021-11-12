//! Write a tensor to a text file.
//!
//! The real documentation is at [`pattie::structs::tensor::COOTensor`].
//!
//! [`pattie::structs::tensor::COOTensor`]: ../../structs/tensor/struct.COOTensor.html#method.write_to_text

use crate::structs::tensor;
use crate::traits::{self, Tensor};
use anyhow::{anyhow, Result};
use num::NumCast;
use std::fmt::LowerExp;
use std::io::{self, Write};

impl<IT, VT> tensor::COOTensor<IT, VT>
where
    IT: traits::IdxType,
    VT: traits::ValType,
{
    /// Write the tensor to a text file.
    ///
    /// # Arguments
    ///
    /// * `w` - A `std::io::Write` object. Examples are `std::fs::File`, `std::io::stdout()`, `Vec<u8>`.
    /// * `index_offset` - An offset to be added to the index. All indices in this library are zero-based, however, some MATLAB or Fortran programs expect one-based indices. Put an offset of 1 if you want to use one-based indices.
    ///
    /// # Example output
    ///
    /// ```text
    /// 3
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
    /// First line is the number of dimensions.
    /// Second line is the shape of the tensor.
    /// The following lines are the elements of the tensor.
    pub fn write_to_text<W>(&self, w: &mut W, index_offset: usize) -> Result<()>
    where
        W: io::Write,
        VT: LowerExp, // https://github.com/rust-lang/rust/issues/31844
    {
        let mut buf_writer = io::BufWriter::new(w);

        writeln!(buf_writer, "{}", self.ndim())?;

        let mut shape_iter = self.shape().into_iter();
        if let Some(axis) = shape_iter.next() {
            write!(buf_writer, "{}", axis.size())?;
        }
        for axis in shape_iter {
            write!(buf_writer, " {}", axis.size())?;
        }
        writeln!(buf_writer)?;

        for (index, value) in self.iter() {
            for index in index.iter() {
                write!(
                    buf_writer,
                    "{}\t",
                    <usize as NumCast>::from(index.clone())
                        .and_then(|index| index.checked_add(index_offset))
                        .ok_or(anyhow!("index value overflow"))?
                )?;
            }
            writeln!(buf_writer, "{:.6e}", value)?;
        }

        Ok(())
    }
}
