//! Read a tensor from a text file.
//!
//! The real documentation is at [`pattie::structs::tensor::COOTensor`].
//!
//! [`pattie::structs::tensor::COOTensor`]: ../../structs/tensor/struct.COOTensor.html#method.read_from_text

use super::lineno_reader::LineNumberReader;
use crate::structs::axis::{Axes, AxisBuilder};
use crate::structs::tensor::{self, COOTensor};
use crate::structs::vec::{smallvec, SmallVec};
use crate::traits;
use ndarray::{aview0, aview1};
use std::ascii;
use std::borrow::Cow;
use std::fmt::Display;
use std::io;
use std::str::FromStr;
use std::string::FromUtf8Error;
use thiserror::Error;

enum Token {
    Eof,
    NewLine,
    Comment,
    Value(String),
}

#[derive(Copy, Clone, Debug, Default)]
struct TokenMask {
    eof: bool,
    new_line: bool,
    comment: bool,
    value: bool,
}

#[derive(Error, Debug)]
pub enum TensorReadError {
    #[error("line {line}, column {column}: {source}")]
    FromUtf8Error {
        line: u64,
        column: u64,
        source: FromUtf8Error,
    },
    #[error("line {line}, column {column}: index out of bound")]
    IndexOutOfBoundError { line: u64, column: u64 },
    #[error("line {line}, column {column}: {source}")]
    IOError {
        line: u64,
        column: u64,
        #[source]
        source: io::Error,
    },
    #[error("line {line}, column {column}: expect {expect}, but found {found}")]
    TokenizeError {
        line: u64,
        column: u64,
        expect: String,
        found: Cow<'static, str>,
    },
    #[error("line {line}, column {column}: unable to parse value {value:?}")]
    ValueError {
        line: u64,
        column: u64,
        value: Cow<'static, str>,
    },
}

impl<IT, VT> tensor::COOTensor<IT, VT>
where
    IT: traits::IdxType,
    VT: traits::ValType,
{
    /// Read a tensor from the text file.
    ///
    /// # Arguments
    ///
    /// * `r` - A `std::io::Read` object. Examples are `std::fs::File`, `std::io::stdin()`, `Vec<u8>`.
    ///
    /// # Example input
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
    pub fn read_from_text<R>(r: &mut R) -> Result<tensor::COOTensor<IT, VT>, TensorReadError>
    where
        R: io::Read,
        IT: FromStr,
        VT: FromStr,
    {
        Self::read_from_text_with_parser(r, |value| value.parse::<VT>().ok())
    }

    /// Similar to [`tensor::COOTensor::read_from_text`], but with a custom parser for values.
    pub fn read_from_text_with_parser<R, P>(
        r: &mut R,
        parser: P,
    ) -> Result<tensor::COOTensor<IT, VT>, TensorReadError>
    where
        R: io::Read,
        P: FnMut(&str) -> Option<VT>,
        IT: FromStr,
    {
        let mut parser = parser;
        let mut r = LineNumberReader::new(io::BufReader::new(r));

        // First line: number of axes
        let ndim = {
            let (line, column) = r.line_column();
            let token = read_until_token(
                &mut r,
                TokenMask {
                    eof: false,
                    new_line: false,
                    comment: false,
                    value: true, // !
                },
                TokenMask {
                    eof: false,
                    new_line: true, // !
                    comment: true,  // !
                    value: false,
                },
            )?;
            if let Token::Value(value) = token {
                value
                    .parse::<usize>()
                    .map_err(|_| TensorReadError::ValueError {
                        line,
                        column,
                        value: value.into(),
                    })?
            } else {
                unreachable!();
            }
        };

        // Read until EOL (or EOF if ndim == 0)
        let mut eof = is_token_eof(&read_until_token(
            &mut r,
            TokenMask {
                eof: ndim == 0,
                new_line: true, // !
                comment: false,
                value: false,
            },
            TokenMask {
                eof: false,
                new_line: true, // !
                comment: true,  // !
                value: false,
            },
        )?);

        // Second line: lower bound of each axis (inclusive)
        let lower_bound = (0..ndim)
            .map(|dim| {
                let (line, column) = r.line_column();
                let token = read_until_token(
                    &mut r,
                    TokenMask {
                        eof: false,
                        new_line: false,
                        comment: false,
                        value: true, // !
                    },
                    TokenMask {
                        eof: false,
                        new_line: dim == 0,
                        comment: true, // !
                        value: false,
                    },
                )?;
                if let Token::Value(value) = token {
                    Ok(value
                        .parse::<IT>()
                        .map_err(|_| TensorReadError::ValueError {
                            line,
                            column,
                            value: value.into(),
                        })?)
                } else {
                    unreachable!();
                }
            })
            .collect::<Result<SmallVec<_>, _>>()?;

        // Read until EOL
        eof = eof
            || is_token_eof(&read_until_token(
                &mut r,
                TokenMask {
                    eof: ndim == 0,
                    new_line: true, // !
                    comment: false,
                    value: false,
                },
                TokenMask {
                    eof: false,
                    new_line: false,
                    comment: true, // !
                    value: false,
                },
            )?);

        // Third line: upper bound of each axis (exclusive)
        let upper_bound = (0..ndim)
            .map(|dim| {
                let (line, column) = r.line_column();
                let token = read_until_token(
                    &mut r,
                    TokenMask {
                        eof: false,
                        new_line: false,
                        comment: false,
                        value: true, // !
                    },
                    TokenMask {
                        eof: false,
                        new_line: dim == 0,
                        comment: true, // !
                        value: true,
                    },
                )?;
                if let Token::Value(value) = token {
                    Ok(value
                        .parse::<IT>()
                        .map_err(|_| TensorReadError::ValueError {
                            line,
                            column,
                            value: value.into(),
                        })?)
                } else {
                    unreachable!();
                }
            })
            .collect::<Result<SmallVec<_>, _>>()?;

        // Read until EOL
        eof = eof
            || is_token_eof(&read_until_token(
                &mut r,
                TokenMask {
                    eof: ndim == 0,
                    new_line: true, // !
                    comment: false,
                    value: false,
                },
                TokenMask {
                    eof: false,
                    new_line: false,
                    comment: true, // !
                    value: false,
                },
            )?);

        // Create the tensor
        let shape = lower_bound
            .into_iter()
            .zip(upper_bound.into_iter())
            .map(|(lower, upper)| AxisBuilder::new().range(lower..upper).build())
            .collect::<Axes<_>>();
        let is_axis_dense: SmallVec<_> = smallvec![false; ndim];
        let mut tensor = COOTensor::zeros(&shape, &is_axis_dense);

        if eof {
            return Ok(tensor);
        }

        // Reuse index to save allocations
        let mut index: SmallVec<_> = smallvec![IT::zero(); ndim];

        // For each line, until EOF
        'a: loop {
            // Index
            for (dim, (idx, axis)) in index.iter_mut().zip(shape.iter()).enumerate() {
                let (line, column) = r.line_column();
                // EOF may be reached if dim == 0
                let token = read_until_token(
                    &mut r,
                    TokenMask {
                        eof: dim == 0,
                        new_line: false,
                        comment: false,
                        value: true,
                    },
                    TokenMask {
                        eof: false,
                        new_line: dim == 0,
                        comment: true,
                        value: false,
                    },
                )?;
                match token {
                    Token::Value(value) => {
                        *idx = value
                            .parse::<IT>()
                            .map_err(|_| TensorReadError::ValueError {
                                line,
                                column,
                                value: value.into(),
                            })?;
                        if !axis.range().contains(idx) {
                            return Err(TensorReadError::IndexOutOfBoundError { line, column });
                        }
                    }
                    Token::Eof => {
                        break 'a;
                    }
                    _ => unreachable!(),
                }
            }

            let value = {
                let (line, column) = r.line_column();
                let token = read_until_token(
                    &mut r,
                    TokenMask {
                        eof: false,
                        new_line: false,
                        comment: false,
                        value: true, // !
                    },
                    TokenMask {
                        eof: false,
                        new_line: false,
                        comment: true, // !
                        value: false,
                    },
                )?;
                if let Token::Value(value) = token {
                    parser(&value).ok_or_else(|| TensorReadError::ValueError {
                        line,
                        column,
                        value: value.into(),
                    })?
                } else {
                    unreachable!();
                }
            };

            tensor.push_block(aview1(&index), aview0(&value).into_dyn());
        }

        Ok(tensor)
    }
}

fn read_next_token<R>(
    r: &mut LineNumberReader<R>,
    expect: TokenMask,
) -> Result<Token, TensorReadError>
where
    R: io::Read,
{
    loop {
        let (line, column) = r.line_column();
        let look_ahead = r.peek_byte().map_err(|source| TensorReadError::IOError {
            line,
            column,
            source,
        })?;
        match look_ahead {
            Some(b'\t' | b' ') => {
                r.read_byte().map_err(|source| TensorReadError::IOError {
                    line,
                    column,
                    source,
                })?;
            }
            Some(b'\n') => {
                if expect.new_line {
                    r.read_byte().map_err(|source| TensorReadError::IOError {
                        line,
                        column,
                        source,
                    })?;
                    return Ok(Token::NewLine);
                } else {
                    return Err(TensorReadError::TokenizeError {
                        line,
                        column,
                        expect: format!("{}", expect),
                        found: "a new line".into(),
                    });
                }
            }
            Some(b'\r') => {
                r.read_byte().map_err(|source| TensorReadError::IOError {
                    line,
                    column,
                    source,
                })?;
                if let Ok(Some(b)) = r.peek_byte() {
                    if b == b'\n' {
                        r.read_byte().map_err(|source| TensorReadError::IOError {
                            line,
                            column,
                            source,
                        })?;
                    }
                }
                return Ok(Token::NewLine);
            }
            Some(b'#') => {
                if expect.comment {
                    read_next_comment(r)?;
                    return Ok(Token::Comment);
                } else {
                    return Err(TensorReadError::TokenizeError {
                        line,
                        column,
                        expect: format!("{}", expect),
                        found: "a comment".into(),
                    });
                }
            }
            Some(b) => {
                if expect.value {
                    return Ok(Token::Value(read_next_value(r)?));
                } else {
                    return Err(TensorReadError::TokenizeError {
                        line,
                        column,
                        expect: format!("{}", expect),
                        found: format!("'{}'", ascii::escape_default(b)).into(),
                    });
                }
            }
            None => {
                if expect.eof {
                    return Ok(Token::Eof);
                } else {
                    return Err(TensorReadError::TokenizeError {
                        line,
                        column,
                        expect: format!("{}", expect),
                        found: "the end of file".into(),
                    });
                }
            }
        }
    }
}

fn read_until_token<R>(
    r: &mut LineNumberReader<R>,
    expect: TokenMask,
    skip: TokenMask,
) -> Result<Token, TensorReadError>
where
    R: io::Read,
{
    let skip = TokenMask {
        eof: skip.eof || expect.eof,
        new_line: skip.new_line || expect.new_line,
        comment: skip.comment || expect.comment,
        value: skip.value || expect.value,
    };
    loop {
        let token = read_next_token(r, skip)?;
        match token {
            Token::Eof => {
                if expect.eof {
                    return Ok(token);
                }
            }
            Token::NewLine => {
                if expect.new_line {
                    return Ok(token);
                }
            }
            Token::Comment => {
                if expect.comment {
                    return Ok(token);
                }
            }
            Token::Value(_) => {
                if expect.value {
                    return Ok(token);
                }
            }
        }
    }
}

fn read_next_comment<R>(r: &mut LineNumberReader<R>) -> Result<(), TensorReadError>
where
    R: io::Read,
{
    loop {
        let (line, column) = r.line_column();
        let look_ahead = r.peek_byte().map_err(|source| TensorReadError::IOError {
            line,
            column,
            source,
        })?;
        match look_ahead {
            Some(b'\n' | b'\r') => {
                r.read_byte().map_err(|source| TensorReadError::IOError {
                    line,
                    column,
                    source,
                })?;
                return Ok(());
            }
            Some(_) => {
                r.read_byte().map_err(|source| TensorReadError::IOError {
                    line,
                    column,
                    source,
                })?;
            }
            None => {
                return Ok(());
            }
        }
    }
}

fn read_next_value<R>(r: &mut LineNumberReader<R>) -> Result<String, TensorReadError>
where
    R: io::Read,
{
    let mut result = Vec::new();
    let (line, column) = r.line_column();
    loop {
        let (line, column) = r.line_column(); // Shadow
        let look_ahead = r.peek_byte().map_err(|source| TensorReadError::IOError {
            line,
            column,
            source,
        })?;
        match look_ahead {
            Some(b'\t' | b'\n' | b'\r' | b' ' | b'#') => break,
            Some(b) => {
                result.push(b);
                r.read_byte().map_err(|source| TensorReadError::IOError {
                    line,
                    column,
                    source,
                })?;
            }
            None => break,
        }
    }
    String::from_utf8(result).map_err(|source| TensorReadError::FromUtf8Error {
        line,
        column,
        source,
    })
}

fn is_token_eof(token: &Token) -> bool {
    matches!(token, Token::Eof)
}

impl Display for TokenMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = [
            (self.value, "a value"),
            (self.comment, "a comment"),
            (self.new_line, "a new line"),
            (self.eof, "the end of file"),
        ]
        .iter()
        .filter_map(|&(b, s)| if b { Some(s) } else { None })
        .collect::<Vec<_>>();
        match s.len() {
            0 => write!(f, "nothing"),
            1 => write!(f, "{}", s[0]),
            2 => write!(f, "{} or {}", s[0], s[1]),
            len => {
                for &i in s[0..len - 1].iter() {
                    write!(f, "{}, ", i)?;
                }
                write!(f, "or {}", s[len - 1])
            }
        }
    }
}
