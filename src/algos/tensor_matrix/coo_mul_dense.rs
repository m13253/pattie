use crate::structs::axis::{map_axes, Axis};
use crate::structs::tensor::{COOTensor, COOTensorInner};
use crate::structs::vec::{smallvec, SmallVec};
use crate::traits::{IdxType, RawParts, Tensor, ValType};
use anyhow::{anyhow, bail, Result};
use ndarray::{aview1, Array2, ArrayView1, ArrayView2, Ix1, Ix3};

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
        if tensor.dense_axes().len() != 0 {
            bail!("The tensor must be fully sparse.");
        }
        // Check if the matrix has 2 axes.
        if matrix.ndim() != 2 {
            bail!("The matrix must have 2 axes.");
        }
        // Check if the matrix is fully dense.
        if matrix.sparse_axes().len() != 0 {
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

        // Check if the tensor is sorted, and the leading axis is the common axis.
        let sparse_sorting_order = tensor
            .sparse_sort_order()
            .ok_or(anyhow!("The tensor must be sorted"))?;
        if sparse_sorting_order[0] != matrix.dense_axes()[0] {
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
        let num_semi_sparse_axes = matrix_values.ncols();
        let mut values = Array2::<VT>::zeros((num_semi_sparse_blocks, num_semi_sparse_axes));
        for i in 0..num_semi_sparse_blocks {
            let inz_begin = fiber_offsets[i];
            let inz_end = fiber_offsets[i + 1];
            for j in inz_begin..inz_end {
                let r = (tensor_indices[(j, common_axis_index)].clone() - common_axis.lower())
                    .to_usize()
                    .unwrap();
                for k in 0..num_semi_sparse_axes {
                    let value = &mut values[(i, k)];
                    *value =
                        value.clone() + tensor_values[j].clone() * matrix_values[(r, k)].clone();
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
            sparse_axes: sparse_axes,
            dense_axes: smallvec![common_axis.clone()],
            indices: indices,
            values: values.into_dyn(),
            sparse_is_sorted: true,
            sparse_sort_order: sparse_sort_order,
        };

        return Ok(
            // # Safety
            // We make sure the tensor is in valid state.
            unsafe { COOTensor::from_raw_parts(result) },
        );

        fn compute_semi_sparse_indices<'a, IT>(
            reference: &Array2<IT>,
            common_axis_index: usize,
        ) -> (Array2<IT>, Vec<usize>)
        where
            IT: IdxType,
        {
            let num_blocks = reference.nrows();
            let num_axes = reference.ncols();
            assert_ne!(num_axes, 0);

            let mut last_index = num_blocks;
            let mut semi_sparse_indices = Array2::zeros((0, num_axes - 1));
            let mut fiber_offsets = Vec::new();

            for i in 0..num_blocks {
                if last_index == num_blocks
                    || reference
                        .row(last_index)
                        .iter()
                        .zip(reference.row(i).iter())
                        .enumerate()
                        .any(|(ax, (a, b))| {
                            // Compare fiber index at row [last_index] and [i], except for the common axis.
                            ax != common_axis_index && a != b
                        })
                {
                    let index_buffer = reference
                        .row(last_index)
                        .iter()
                        .enumerate()
                        .filter_map(|(ax, index)| {
                            if ax != common_axis_index {
                                Some(index.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<SmallVec<_>>();
                    semi_sparse_indices.push_row(aview1(&index_buffer)).unwrap();
                    last_index = i;
                    fiber_offsets.push(i);
                }
            }
            fiber_offsets.push(num_blocks);

            (semi_sparse_indices, fiber_offsets)
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
