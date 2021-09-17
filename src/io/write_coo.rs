use crate::structs::tensor;
use crate::traits::{self, Tensor};
use anyhow::{anyhow, Result};
use num::NumCast;
use std::fmt::LowerExp;
use std::io::{self, Write};

impl<VT, IT> tensor::COO<VT, IT>
where
    VT: traits::ValType,
    IT: traits::IdxType,
{
    pub fn write_to_text<W>(&self, w: &mut W, index_offset: usize) -> Result<()>
    where
        W: io::Write,
        VT: LowerExp,
    {
        let mut buf_writer = io::BufWriter::new(w);

        writeln!(buf_writer, "{}", self.ndim())?;

        let mut dim_iter = self.shape().into_iter();
        if let Some(dim) = dim_iter.next() {
            write!(buf_writer, "{}", dim)?;
        }
        for dim in dim_iter {
            write!(buf_writer, " {}", dim)?;
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
            // Once specialization is available, we can remove the LowerExp requirement.
            // This macro ensures the compilation will fail after that day, so we can come back and change the code to use specialization.
            todo_or_die::issue_closed!("rust-lang", "rust", 31844);
            writeln!(buf_writer, "{:.6e}", value)?;
        }

        Ok(())
    }
}
