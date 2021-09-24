//! Read a tensor from a text file.
//!
//! The real documentation is at [`pattie::structs::tensor::COOTensor`].
//!
//! [`pattie::structs::tensor::COOTensor`]: ../../structs/tensor/struct.COOTensor.html#method.read_from_text

use crate::structs::tensor;
use crate::traits;
use anyhow::{anyhow, bail, Result};
use ndarray::{Dimension, Ix, Ix0};
use num::NumCast;
use std::io;

impl<VT, IT> tensor::COOTensor<VT, IT>
where
    VT: traits::ValType,
    IT: traits::IdxType,
{
    /// Read a tensor from the text file.
    ///
    /// # Arguments
    ///
    /// * `r` - A `std::io::Read` object. Examples are `std::fs::File`, `std::io::stdin()`, `Vec<u8>`.
    /// * `index_offset` - An offset to be added to the index. All indices in this library are zero-based, however, some MATLAB or Fortran programs provide one-based indices. Put an offset of 1 if you want to use one-based indices.
    ///
    /// # Example input
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
    pub fn read_from_text<R>(r: &mut R, index_offset: usize) -> Result<tensor::COOTensor<VT, IT>>
    where
        R: io::Read,
    {
        let buf_reader = io::BufReader::new(r);
        let mut lines = io::BufRead::lines(buf_reader);

        let ndim = lines
            .next()
            .ok_or(io::Error::new(io::ErrorKind::UnexpectedEof, "empty file"))??
            .trim()
            .parse::<usize>()
            .map_err(|_| anyhow!("line 1: invalid number of dimensions"))?;

        if ndim == 0 {
            return Ok(tensor::COOTensor::zeros(Ix0().into_dyn()));
        }

        let shape = lines
            .next()
            .ok_or(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "line 2: unexpected end of file",
            ))??
            .trim()
            .split_whitespace()
            .map(|s| {
                s.parse::<Ix>()
                    .map_err(|_| anyhow!("line 2: invalid shape"))
            })
            .collect::<Result<Vec<_>>>()?;
        if shape.len() != ndim {
            bail!("line 2: tensor shape does not match the header");
        }

        let mut tsr = tensor::COOTensor::<VT, IT>::zeros(shape);

        for (line_no, line) in lines.enumerate() {
            let line = line?;
            let mut fields = line.trim().split_whitespace();

            let index = fields
                .by_ref()
                .take(ndim)
                .map(|s| {
                    s.parse::<usize>().map_or(
                        Err(anyhow!("line {}: invalid index", line_no + 3)),
                        |v| {
                            v.checked_sub(index_offset)
                                .and_then(<IT as NumCast>::from)
                                .ok_or(anyhow!("line {}: index overflow", line_no + 3))
                        },
                    )
                })
                .collect::<Result<Vec<IT>>>()?;
            if index.len() != ndim {
                bail!(
                    "line {}: index length does not match the header",
                    line_no + 3
                );
            }

            let value = fields
                .next()
                .ok_or(anyhow!("line {}: missing value", line_no + 3))?
                .parse::<VT>()
                .map_err(|_| anyhow!("line {}: invalid value", line_no + 3))?;

            tsr.push(&index, value);
        }

        Ok(tsr)
    }
}
