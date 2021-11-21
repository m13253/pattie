use std::io;

pub struct LineNumberReader<R>
where
    R: io::Read,
{
    inner: R,
    line_number: u64,
    column_number: u64,
    peek: Option<Peek>,
}

enum Peek {
    Byte(u8),
    EOF,
}

impl<R> LineNumberReader<R>
where
    R: io::Read,
{
    pub fn new(inner: R) -> LineNumberReader<R> {
        LineNumberReader {
            inner,
            line_number: 1,
            column_number: 1,
            peek: None,
        }
    }

    pub fn line_column(&self) -> (u64, u64) {
        if let Some(Peek::Byte(b)) = self.peek {
            if b == b'\n' {
                (self.line_number + 1, 1)
            } else {
                (self.line_number, self.column_number + 1)
            }
        } else {
            (self.line_number, self.column_number)
        }
    }

    pub fn peek_byte(&mut self) -> io::Result<Option<u8>> {
        match self.peek {
            Some(Peek::Byte(b)) => Ok(Some(b)),
            Some(Peek::EOF) => Ok(None),
            None => {
                let mut buf = [0; 1];
                let bytes_read = self.inner.read(&mut buf)?;
                if bytes_read != 0 {
                    self.peek = Some(Peek::Byte(buf[0]));
                    Ok(Some(buf[0]))
                } else {
                    self.peek = Some(Peek::EOF);
                    Ok(None)
                }
            }
        }
    }

    pub fn read_byte(&mut self) -> io::Result<Option<u8>> {
        match self.peek {
            Some(Peek::Byte(b)) => {
                self.peek = None;
                if b == b'\n' {
                    self.line_number += 1;
                    self.column_number = 1;
                } else {
                    self.column_number += 1;
                }
                Ok(Some(b))
            }
            Some(Peek::EOF) => {
                self.peek = None;
                Ok(None)
            }
            None => {
                let mut buf = [0; 1];
                let bytes_read = self.inner.read(&mut buf)?;
                if bytes_read != 0 {
                    Ok(Some(buf[0]))
                } else {
                    Ok(None)
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn inner(&mut self) -> &mut R {
        &mut self.inner
    }

    #[allow(dead_code)]
    pub fn into_inner(self) -> R {
        self.inner
    }
}

impl<R> io::Read for LineNumberReader<R>
where
    R: io::Read,
{
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self.peek {
            Some(Peek::Byte(b)) => {
                if let Some(buf) = buf.get_mut(0) {
                    self.peek = None;
                    if b == b'\n' {
                        self.line_number += 1;
                        self.column_number = 1;
                    } else {
                        self.column_number += 1;
                    }
                    *buf = b;
                    Ok(1)
                } else {
                    Ok(0)
                }
            }
            Some(Peek::EOF) => {
                self.peek = None;
                Ok(0)
            }
            None => {
                let bytes_read = self.inner.read(buf)?;
                for &b in buf[..bytes_read].iter() {
                    if b == b'\n' {
                        self.line_number += 1;
                        self.column_number = 1;
                    } else {
                        self.column_number += 1;
                    }
                }
                Ok(bytes_read)
            }
        }
    }
}
