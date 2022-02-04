use std::mem;

use crate::structs::axis::{map_axes, Axis};
use crate::structs::tensor::{COOTensor, COOTensorInner};
use crate::structs::vec::{smallvec, SmallVec};
use crate::traits::{IdxType, RawParts, Tensor, ValType};
use crate::utils::tracer::Tracer;
use anyhow::{anyhow, bail, Result};
use log::info;
use ndarray::{Array2, ArrayView1, ArrayView2, Ix1, Ix3};
use rayon::prelude::*;
use scopeguard::defer;
use ubyte::ToByteUnit;

/// Multiply a `COOTensor` with a `DenseMatrix`.
pub struct COOTensorMulDenseMatrix<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    pub tensor: &'a COOTensor<IT, VT>,
    pub matrix: &'a COOTensor<IT, VT>,

    pub tracer: Tracer,
    pub multi_thread: bool,
}

impl<'a, IT, VT> COOTensorMulDenseMatrix<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    /// Create a new `COOTensorMulDenseMatrix` task.
    #[must_use]
    pub fn new(tensor: &'a COOTensor<IT, VT>, matrix: &'a COOTensor<IT, VT>) -> Self {
        Self {
            tensor,
            matrix,
            tracer: Tracer::new_dummy(),
            multi_thread: false,
        }
    }

    /// Enable performance tracing.
    #[must_use]
    pub fn trace(mut self, tracer: &Tracer) -> Self {
        self.tracer.clone_from(tracer);
        self
    }

    /// Perform the multiplication.
    ///
    /// The last axis of the tensor's sparse_sort_order must be the same as the first axis of the matrix.
    pub fn execute(self) -> Result<COOTensor<IT, VT>>
    where
        IT: IdxType,
        VT: ValType,
    {
        let event = self.tracer.start();
        defer! {
            event.finish("COOTensorMulDenseMatrix");
        }

        // This algorithm only solves the case where the tensor is fully sparse.
        if !self.tensor.dense_axes().is_empty() {
            bail!("The tensor must be fully sparse.");
        }
        // Check if the matrix has 2 axes.
        if self.matrix.ndim() != 2 {
            bail!("The matrix must have 2 axes.");
        }
        // Check if the matrix is fully dense.
        if !self.matrix.sparse_axes().is_empty() {
            bail!("The matrix must be fully dense.");
        }

        // Map the common axis of the tensor and the matrix.
        let matrix_dense_axes = self.matrix.dense_axes();
        assert_eq!(matrix_dense_axes.len(), 2);
        let common_axis = matrix_dense_axes.first().unwrap();
        let matrix_dense_to_tensor_sparse =
            map_axes(self.matrix.dense_axes(), self.tensor.sparse_axes()).collect::<SmallVec<_>>();
        assert_eq!(matrix_dense_to_tensor_sparse.len(), 2);
        let common_axis_index = *matrix_dense_to_tensor_sparse[0]
            .as_ref()
            .map_err(|err| anyhow!("{}", err))?;
        if matrix_dense_to_tensor_sparse[1].is_ok() {
            bail!("There must be only one common axis.");
        };

        // Check if the tensor is sorted, and the trailing axis is the common axis.
        let sparse_sorting_order = self
            .tensor
            .sparse_sort_order()
            .ok_or(anyhow!("The tensor must be sorted"))?;
        if sparse_sorting_order.last() != self.matrix.dense_axes().first() {
            bail!("The tensor must be sorted along the common axis.");
        }

        // Extract the contents from the inputs.
        let tensor_indices = &self.tensor.raw_parts().indices;
        // Reshape the tensor into an ArrayView1.
        let tensor_values = self
            .tensor
            .raw_parts()
            .values
            .view()
            .into_dimensionality::<Ix1>()?;
        let matrix_shape = self.matrix.shape();
        // Reshape the matrix into an ArrayView2.
        let matrix_values = self
            .matrix
            .raw_parts()
            .values
            .view()
            .into_dimensionality::<Ix3>()?
            .index_axis_move(ndarray::Axis(0), 0);

        // Create the output tensor.
        let result_shape = self
            .tensor
            .shape()
            .iter()
            .map(|ax| {
                if ax == &matrix_shape[0] {
                    matrix_shape[1].clone()
                } else {
                    ax.clone()
                }
            })
            .collect::<SmallVec<_>>();
        let result_sparse_axes = self
            .tensor
            .sparse_axes()
            .iter()
            .filter(|&ax| ax != common_axis)
            .cloned()
            .collect::<SmallVec<_>>();
        let sparse_sort_order = self
            .tensor
            .sparse_sort_order()
            .unwrap()
            .iter()
            .filter(|&ax| ax != common_axis)
            .cloned()
            .collect::<SmallVec<_>>();

        // Calculate the result indices.
        let (result_indices, fiber_offsets) =
            self.compute_indices(tensor_indices, common_axis_index);

        let memory_burden = (tensor_values.len() as u64
            * (mem::size_of::<IT>() as u64
                + matrix_values.ncols() as u64 * mem::size_of::<VT>() as u64 * 4))
            .bytes();
        info!(target: "COOTensorMulDenseMatrix::compute_values", "Memory burden: {}", memory_burden);

        let result_values = if self.multi_thread {
            self.compute_values_multi_thread(
                tensor_indices,
                &tensor_values,
                &matrix_values,
                &result_indices,
                &fiber_offsets,
                common_axis,
                common_axis_index,
            )
        } else {
            self.compute_values(
                tensor_indices,
                &tensor_values,
                &matrix_values,
                &result_indices,
                &fiber_offsets,
                common_axis,
                common_axis_index,
            )
        };

        let result = COOTensorInner {
            name: None,
            shape: result_shape,
            sparse_axes: result_sparse_axes,
            dense_axes: smallvec![common_axis.clone()],
            indices: result_indices,
            values: result_values.into_dyn(),
            sparse_is_sorted: true,
            sparse_sort_order,
        };

        Ok(
            // # Safety
            // We make sure the tensor is in valid state.
            unsafe { COOTensor::from_raw_parts(result) },
        )
    }

    fn compute_indices(
        &self,
        tensor_indices: &Array2<IT>,
        common_axis_index: usize,
    ) -> (Array2<IT>, Vec<usize>)
    where
        IT: IdxType,
    {
        let event = self.tracer.start();
        defer! {
            event.finish("COOTensorMulDenseMatrix::compute_indices");
        }

        let num_blocks = tensor_indices.nrows();
        let num_axes = tensor_indices.ncols();
        assert_ne!(num_axes, 0);

        let mut last_index = None;
        let mut semi_sparse_indices = Vec::new();
        let mut fiber_offsets = Vec::new();
        let mut index_buffer: SmallVec<_> = smallvec![IT::zero(); num_axes-1];

        for i in 0..num_blocks {
            if last_index
                .map(|last_index| unsafe {
                    // # Safety
                    // Both `last_index` and `i` are lower than `num_blocks`.
                    !self.index_eq_except_axis(tensor_indices, last_index, i, common_axis_index)
                })
                .unwrap_or(true)
            {
                unsafe {
                    self.copy_index_except_axis(
                        tensor_indices,
                        i,
                        common_axis_index,
                        index_buffer.as_mut_slice(),
                    )
                }
                semi_sparse_indices.extend_from_slice(index_buffer.as_slice());
                fiber_offsets.push(i);
                last_index = Some(i);
            }
        }
        fiber_offsets.push(num_blocks);

        let result_indices =
            Array2::from_shape_vec((fiber_offsets.len() - 1, num_axes - 1), semi_sparse_indices)
                .unwrap();
        (result_indices, fiber_offsets)
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_values(
        &self,
        tensor_indices: &Array2<IT>,
        tensor_values: &ArrayView1<VT>,
        matrix_values: &ArrayView2<VT>,
        result_indices: &Array2<IT>,
        fiber_offsets: &[usize],
        common_axis: &Axis<IT>,
        common_axis_index: usize,
    ) -> Array2<VT>
    where
        IT: IdxType,
        VT: ValType,
    {
        let event = self.tracer.start();
        defer! {
            event.finish("COOTensorMulDenseMatrix::compute_values");
        }

        let num_fibers = result_indices.nrows();
        let matrix_free_axis_len = matrix_values.ncols();
        let mut result_values = Array2::<VT>::zeros((num_fibers, matrix_free_axis_len));

        for i in 0..num_fibers {
            // # Safety
            // fiber_offests.len() == num_fibers + 1
            let inz_begin = *unsafe { fiber_offsets.get_unchecked(i) };
            let inz_end = *unsafe { fiber_offsets.get_unchecked(i + 1) };
            for j in inz_begin..inz_end {
                // # Safety
                // j < inz_end <= indices.nrows()
                // 0 <= r < common_axis.size()
                let r = unsafe {
                    (*tensor_indices.uget((j, common_axis_index)) - common_axis.lower())
                        .to_usize()
                        .unwrap_unchecked()
                };
                for k in 0..matrix_free_axis_len {
                    // # Safety
                    // i < num_fibers
                    // j < indices.nrows()
                    // r < common_axis.len() == matrix_values.nrows()
                    // k < matrix_free_axis_len
                    unsafe {
                        let value = result_values.uget_mut((i, k));
                        *value = value.clone()
                            + tensor_values.uget(j).clone() * matrix_values.uget((r, k)).clone();
                    }
                }
            }
        }

        result_values
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_values_multi_thread(
        &self,
        tensor_indices: &Array2<IT>,
        tensor_values: &ArrayView1<VT>,
        matrix_values: &ArrayView2<VT>,
        result_indices: &Array2<IT>,
        fiber_offsets: &[usize],
        common_axis: &Axis<IT>,
        common_axis_index: usize,
    ) -> Array2<VT>
    where
        IT: IdxType,
        VT: ValType,
    {
        let event = self.tracer.start();
        defer! {
            event.finish("COOTensorMulDenseMatrix::compute_values_multi_thread");
        }

        let num_fibers = result_indices.nrows();
        let matrix_free_axis_len = matrix_values.ncols();
        let mut result_values = Array2::<VT>::zeros((num_fibers, matrix_free_axis_len));

        result_values
            .outer_iter_mut()
            .into_par_iter()
            .enumerate()
            //.with_max_len(num_fibers / 1024 + 1)
            .for_each(|(i, mut result_row)| {
                // # Safety
                // fiber_offests.len() == num_fibers + 1
                let inz_begin = *unsafe { fiber_offsets.get_unchecked(i) };
                let inz_end = *unsafe { fiber_offsets.get_unchecked(i + 1) };
                for j in inz_begin..inz_end {
                    // # Safety
                    // j < inz_end <= indices.nrows()
                    // 0 <= r < common_axis.size()
                    let r = unsafe {
                        (*tensor_indices.uget((j, common_axis_index)) - common_axis.lower())
                            .to_usize()
                            .unwrap_unchecked()
                    };
                    for k in 0..matrix_free_axis_len {
                        // # Safety
                        // i < num_fibers
                        // j < indices.nrows()
                        // r < common_axis.len() == matrix_values.nrows()
                        // k < matrix_free_axis_len
                        unsafe {
                            let value = result_row.uget_mut(k);
                            *value = value.clone()
                                + tensor_values.uget(j).clone()
                                    * matrix_values.uget((r, k)).clone();
                        }
                    }
                }
            });

        result_values
    }

    /// Compare fiber index at row `row_a` and `row_b`, except for one common axis.
    ///
    /// # Safety
    /// Make sure row_{a,b} < indices.nrows()
    unsafe fn index_eq_except_axis(
        &self,
        indices: &Array2<IT>,
        row_a: usize,
        row_b: usize,
        except_axis_index: usize,
    ) -> bool
    where
        IT: IdxType,
    {
        for i in (0..indices.ncols()).rev() {
            // Performance concern:
            // Why we don't use `indices.row()`?
            // Because it creates an `ndarray::NdProducer`, which is painfully slow.
            if i != except_axis_index && indices.uget((row_a, i)) != indices.uget((row_b, i)) {
                return false;
            }
        }
        true
    }

    /// Copy the index of the fiber at row `row` to `index_buffer`.
    ///
    /// # Safety
    /// Make sure row < indices.nrows()
    /// Make sure except_axis_index < indices.ncols()
    /// Make sure index_buffer.len() == indices.ncols() - 1
    unsafe fn copy_index_except_axis(
        &self,
        indices: &Array2<IT>,
        row: usize,
        except_axis_index: usize,
        index_buffer: &mut [IT],
    ) where
        IT: IdxType,
    {
        for i in 0..except_axis_index {
            index_buffer
                .get_unchecked_mut(i)
                .clone_from(indices.uget((row, i)));
        }
        for i in except_axis_index + 1..indices.ncols() {
            index_buffer
                .get_unchecked_mut(i - 1)
                .clone_from(indices.uget((row, i)));
        }
    }
}
