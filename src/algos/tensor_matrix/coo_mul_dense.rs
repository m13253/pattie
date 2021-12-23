use crate::structs::axis::{map_axes, Axis};
use crate::structs::tensor::{COOTensor, COOTensorInner};
use crate::structs::vec::{smallvec, SmallVec};
use crate::traits::{IdxType, RawParts, Tensor, ValType};
use crate::utils::ndarray_unsafe::{uncheck_arr, uncheck_arr_mut};
use anyhow::{anyhow, bail, Result};
use ndarray::{Array2, ArrayView1, ArrayView2, Ix1, Ix3};

/// Task builder to multiply a `COOTensor` with a `DenseMatrix`.
pub struct COOTensorMulDenseMatrix<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    tensor: Option<&'a COOTensor<IT, VT>>,
    matrix: Option<&'a COOTensor<IT, VT>>,
    tensor_values: Option<ArrayView1<'a, VT>>,
    matrix_values: Option<ArrayView2<'a, VT>>,
    common_axis: Option<Axis<IT>>,
    /// The index of the common axis in `tensor.sparse_axes()`
    common_axis_index: Option<usize>,
}

impl<'a, IT, VT> COOTensorMulDenseMatrix<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    /// Create a new `COOTensorMulDenseMatrix` task builder.
    pub fn new() -> Self {
        Self {
            tensor: None,
            matrix: None,
            tensor_values: None,
            matrix_values: None,
            common_axis: None,
            common_axis_index: None,
        }
    }

    pub fn prepare(
        mut self,
        tensor: &'a COOTensor<IT, VT>,
        matrix: &'a COOTensor<IT, VT>,
    ) -> Result<Self> {
        self.tensor = Some(tensor);
        self.matrix = Some(matrix);

        // Check if the tensor is fully sparse.
        if !tensor.dense_axes().is_empty() {
            bail!("The tensor must be fully sparse.");
        }
        // Check if the matrix has 2 axes.
        if matrix.ndim() != 2 {
            bail!("The matrix must have 2 axes.");
        }
        // Check if the matrix is fully dense.
        if !matrix.sparse_axes().is_empty() {
            bail!("The matrix must be fully dense.");
        }

        // Map the common axis of the tensor and the matrix.
        let matrix_dense_axes = matrix.dense_axes();
        assert_eq!(matrix_dense_axes.len(), 2);
        let common_axis = &matrix_dense_axes[0];
        let matrix_dense_to_tensor_sparse =
            map_axes(matrix.dense_axes(), tensor.sparse_axes()).collect::<SmallVec<_>>();
        assert_eq!(matrix_dense_to_tensor_sparse.len(), 2);
        let common_axis_index = *matrix_dense_to_tensor_sparse[0]
            .as_ref()
            .map_err(|err| anyhow!("{}", err))?;
        if matrix_dense_to_tensor_sparse[1].is_ok() {
            bail!("There must be only one common axis.");
        };

        // Check if the tensor is sorted, and the trailing axis is the common axis.
        let sparse_sorting_order = tensor
            .sparse_sort_order()
            .ok_or(anyhow!("The tensor must be sorted"))?;
        if sparse_sorting_order.last() != matrix.dense_axes().first() {
            bail!("The tensor must be sorted along the common axis.");
        }

        // Store the tensor into the task builder.
        self.tensor_values = Some(
            tensor
                .raw_parts()
                .values
                .view()
                .into_dimensionality::<Ix1>()?,
        );
        // Extract the matrix into an ArrayView2.
        self.matrix_values = Some(
            matrix
                .raw_parts()
                .values
                .view()
                .into_dimensionality::<Ix3>()?
                .index_axis_move(ndarray::Axis(0), 0),
        );
        self.common_axis = Some(common_axis.clone());
        self.common_axis_index = Some(common_axis_index);

        // We are done.
        Ok(self)
    }

    /// Multiply a `COOTensor` with a `DenseMatrix`.
    ///
    /// This method assumes that the tensor is already sorted. Otherwise, call `prepare` first.
    pub fn execute(self) -> Result<COOTensor<IT, VT>>
    where
        IT: IdxType,
        VT: ValType,
    {
        // Extract values from the task builder.
        // They should be ready.
        let tensor = self.tensor.unwrap();
        let matrix = self.matrix.unwrap();
        let tensor_indices = &tensor.raw_parts().indices;
        let tensor_values = self.tensor_values.unwrap();
        let matrix_values = self.matrix_values.unwrap();
        let common_axis = self.common_axis.as_ref().unwrap();
        let common_axis_index = self.common_axis_index.unwrap();

        let matrix_shape = matrix.shape();
        let result_shape = tensor
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
        let sparse_axes = tensor
            .sparse_axes()
            .iter()
            .filter(|&ax| ax != common_axis)
            .cloned()
            .collect::<SmallVec<_>>();
        let (indices, fiber_offsets) =
            compute_semi_sparse_indices(&tensor.raw_parts().indices, common_axis_index);

        let num_semi_sparse_blocks = indices.nrows();
        let matrix_free_axis_len = matrix_values.ncols();
        let mut values = Array2::<VT>::zeros((num_semi_sparse_blocks, matrix_free_axis_len));

        for i in 0..num_semi_sparse_blocks {
            // # Safety
            // fiber_offests.len() == num_semi_sparse_blocks + 1
            let inz_begin = *unsafe { fiber_offsets.get_unchecked(i) };
            let inz_end = *unsafe { fiber_offsets.get_unchecked(i + 1) };
            for j in inz_begin..inz_end {
                // # Safety
                // j < inz_end <= indices.nrows()
                // 0 <= r < common_axis.size()
                let r = unsafe {
                    (*uncheck_arr(&tensor_indices).get((j, common_axis_index))
                        - common_axis.lower())
                    .to_usize()
                    .unwrap_unchecked()
                };
                for k in 0..matrix_free_axis_len {
                    // # Safety
                    // i < num_semi_sparse_blocks
                    // j < indices.nrows()
                    // r < common_axis.len() == matrix_values.nrows()
                    // k < matrix_free_axis_len
                    unsafe {
                        let value = uncheck_arr_mut(&mut values).get((i, k));
                        *value = value.clone()
                            + uncheck_arr(&tensor_values).get(j).clone()
                                * uncheck_arr(&matrix_values).get((r, k)).clone();
                    }
                }
            }
        }

        let sparse_sort_order = tensor
            .sparse_sort_order()
            .unwrap()
            .iter()
            .filter(|&ax| ax != common_axis)
            .cloned()
            .collect::<SmallVec<_>>();

        let result = COOTensorInner {
            name: None,
            shape: result_shape,
            sparse_axes,
            dense_axes: smallvec![common_axis.clone()],
            indices,
            values: values.into_dyn(),
            sparse_is_sorted: true,
            sparse_sort_order,
        };

        return Ok(
            // # Safety
            // We make sure the tensor is in valid state.
            unsafe { COOTensor::from_raw_parts(result) },
        );

        fn compute_semi_sparse_indices<IT>(
            reference: &Array2<IT>,
            common_axis_index: usize,
        ) -> (Array2<IT>, Vec<usize>)
        where
            IT: IdxType,
        {
            let num_blocks = reference.nrows();
            let num_axes = reference.ncols();
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
                        !index_eq_except_axis(reference, last_index, i, common_axis_index)
                    })
                    .unwrap_or(true)
                {
                    unsafe {
                        copy_index_except_axis(
                            reference,
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

            let semi_sparse_indices = Array2::from_shape_vec(
                (fiber_offsets.len() - 1, num_axes - 1),
                semi_sparse_indices,
            )
            .unwrap();
            (semi_sparse_indices, fiber_offsets)
        }

        /// Compare fiber index at row `row_a` and `row_b`, except for one common axis.
        ///
        /// # Safety
        /// Make sure row_{a,b} < indices.nrows()
        unsafe fn index_eq_except_axis<IT>(
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
                if i != except_axis_index
                    && uncheck_arr(&indices).get((row_a, i))
                        != uncheck_arr(&indices).get((row_b, i))
                {
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
        unsafe fn copy_index_except_axis<IT>(
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
                    .clone_from(uncheck_arr(&indices).get((row, i)));
            }
            for i in except_axis_index + 1..indices.ncols() {
                index_buffer
                    .get_unchecked_mut(i - 1)
                    .clone_from(uncheck_arr(&indices).get((row, i)));
            }
        }
    }
}

impl<'a, IT, VT> Default for COOTensorMulDenseMatrix<'a, IT, VT>
where
    IT: IdxType,
    VT: ValType,
{
    fn default() -> Self {
        Self::new()
    }
}
