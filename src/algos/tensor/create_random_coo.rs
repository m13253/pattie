use crate::structs::axis::Axis;
use crate::structs::tensor::COOTensor;
use crate::structs::vec::{smallvec, SmallVec};
use crate::traits::{IdxType, RawParts, ValType};
use anyhow::Result;
use ndarray::{aview1, Array1, Array2};
use ndarray_rand::rand;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use num::{CheckedMul, Float, ToPrimitive};
use std::collections::HashMap;

/// Create a random COO sparse tensor.
///
/// The density of non-zero elements is controlled by `density`.
/// The tensor is filled with random values drawn from a normal distribution.
pub struct CreateRandomCOOTensor<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    pub shape: &'a [Axis<IT>],
    pub density: f64,
    pub mean: VT,
    pub std_dev: VT,
}

impl<'a, IT, VT> CreateRandomCOOTensor<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType + Float,
    StandardNormal: Distribution<VT>,
{
    /// Create a new `CreateRandomCOOTensor` task.
    #[must_use]
    pub fn new(shape: &'a [Axis<IT>], density: f64, mean: VT, std_dev: VT) -> Self {
        Self {
            shape,
            density,
            mean,
            std_dev,
        }
    }

    /// Perform the generation.
    ///
    /// Returns `Err` if the random generator fails.
    ///
    /// # Allocation
    /// This function requires `O(n)` auxiliary memory, where `n` is the number of non-zero elements.
    pub fn execute(self) -> Result<COOTensor<IT, VT>>
    where
        IT: IdxType + SampleUniform,
        VT: ValType + Float,
        StandardNormal: Distribution<VT>,
    {
        let ndim = self.shape.len();
        let is_axis_dense: SmallVec<bool> = smallvec![false; ndim];
        let mut tensor = COOTensor::<IT, VT>::zeros(self.shape, &is_axis_dense);

        let (strides, total_size) = Self::calc_strides(self.shape);
        let num_non_zeros = (total_size as f64 * self.density)
            .round()
            .to_usize()
            .unwrap();
        let mut index_hashtable = HashMap::<usize, ()>::with_capacity(num_non_zeros);

        let mut raw_parts = unsafe { tensor.raw_parts_mut() };
        let mut rng = rand::thread_rng();
        let index_distr = self
            .shape
            .iter()
            .map(|ax| Uniform::new(ax.lower(), ax.upper()))
            .collect::<SmallVec<_>>();

        let mut indices = Array2::uninit((num_non_zeros, ndim));
        let mut index_buffer: SmallVec<IT> = smallvec![IT::zero(); ndim];
        for row in indices.rows_mut() {
            loop {
                for ax in 0..ndim {
                    index_buffer[ax] = index_distr[ax].sample(&mut rng);
                }
                let offset = Self::index_to_offset(&index_buffer, self.shape, &strides);
                if index_hashtable.insert(offset, ()).is_none() {
                    aview1(&index_buffer).assign_to(row);
                    break;
                }
            }
        }
        let indices = unsafe { indices.assume_init() };

        raw_parts.indices = indices;
        raw_parts.values =
            Array1::random(num_non_zeros, Normal::new(self.mean, self.std_dev)?).into_dyn();
        Ok(tensor)
    }

    fn calc_strides(shape: &[Axis<IT>]) -> (SmallVec<usize>, usize)
    where
        IT: IdxType,
    {
        let mut strides = smallvec![0; shape.len()];
        let mut total_size = 1;
        for (axis, stride) in shape.iter().zip(strides.iter_mut()).rev() {
            *stride = total_size;
            total_size = total_size.checked_mul(&axis.len()).unwrap();
        }
        (strides, total_size)
    }

    fn index_to_offset(index: &[IT], shape: &[Axis<IT>], strides: &[usize]) -> usize
    where
        IT: IdxType,
    {
        index
            .iter()
            .zip(shape.iter())
            .zip(strides.iter())
            .map(|((&idx, axis), &stride)| {
                (idx - axis.lower())
                    .to_usize()
                    .unwrap()
                    .checked_mul(stride)
                    .unwrap()
            })
            .product()
    }
}
