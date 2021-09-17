use crate::structs::tensor;
use crate::traits;
use anyhow::{anyhow, bail, Result};
use ndarray::{Dimension, Ix, Ix0};
use num::NumCast;
use std::io;

impl<VT, IT> tensor::COO<VT, IT>
where
    VT: traits::ValType,
    IT: traits::IdxType,
{
    pub fn read_from_text<R>(r: &mut R, index_offset: usize) -> Result<tensor::COO<VT, IT>>
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
            return Ok(tensor::COO::zeros(Ix0().into_dyn()));
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

        let mut tsr = tensor::COO::<VT, IT>::zeros(shape);

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
