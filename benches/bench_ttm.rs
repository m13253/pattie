#![cfg(test)]
#![feature(test)]

extern crate test;

use pattie::algos::matrix::create_random_dense_matrix;
use pattie::algos::tensor::{create_random_coo, SortCOOTensor};
use pattie::algos::tensor_matrix::COOTensorMulDenseMatrix;
use pattie::structs::axis::AxisBuilder;
use pattie::utils::hint::black_box;
use test::Bencher;

#[bench]
fn bench_ttm(b: &mut Bencher) {
    let tensor_shape = vec![
        AxisBuilder::new().range(1..101).build(),
        AxisBuilder::new().range(1..51).build(),
        AxisBuilder::new().range(1..51).build(),
        AxisBuilder::new().range(1..101).build(),
    ];
    let matrix_shape = (
        tensor_shape[1].clone(),
        AxisBuilder::new().range(1..17).build(),
    );
    let sort_order = vec![
        tensor_shape[0].clone(),
        tensor_shape[2].clone(),
        tensor_shape[3].clone(),
        tensor_shape[1].clone(),
    ];
    let mut tensor = create_random_coo::<u32, f32>(&tensor_shape, 1e-4, 0.0, 1.0).unwrap();
    let matrix = create_random_dense_matrix::<u32, f32>(matrix_shape, 0.0, 1.0).unwrap();

    let sort_task = SortCOOTensor::new().prepare(&mut tensor, &sort_order);
    sort_task.execute();

    b.iter(|| {
        let ttm_task = COOTensorMulDenseMatrix::new()
            .prepare(&tensor, &matrix)
            .unwrap();
        let _output = black_box(ttm_task.execute());
    });
}
