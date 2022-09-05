#![cfg(test)]
#![feature(test)]

extern crate test;

use pattie::algos::matrix::CreateRandomDenseMatrix;
use pattie::algos::tensor::{CreateRandomCOOTensor, SortCOOTensor};
use pattie::algos::tensor_matrix::COOTensorMulDenseMatrix;
use pattie::structs::axis::AxisBuilder;
use pattie::utils::hint::black_box;
use rayon;
use test::Bencher;

#[bench]
fn bench_coo_ttm_multi_thread(b: &mut Bencher) {
    rayon::ThreadPoolBuilder::new().build_global().unwrap_or(());

    let tensor_shape = vec![
        AxisBuilder::new().range(1..201).build(),
        AxisBuilder::new().range(1..201).build(),
        AxisBuilder::new().range(1..201).build(),
        AxisBuilder::new().range(1..201).build(),
    ];
    let matrix_shape = (
        tensor_shape[1].clone(),
        AxisBuilder::new().range(0..128).build(),
    );
    let sort_order = vec![
        tensor_shape[0].clone(),
        tensor_shape[2].clone(),
        tensor_shape[3].clone(),
        tensor_shape[1].clone(),
    ];
    let mut tensor = CreateRandomCOOTensor::<u32, f32>::new(&tensor_shape, 1e-4, 0.0, 1.0)
        .execute()
        .unwrap();
    let matrix = CreateRandomDenseMatrix::<u32, f32>::new(matrix_shape, 0.0, 1.0)
        .execute()
        .unwrap();

    let sort_task = SortCOOTensor::new(&mut tensor, &sort_order);
    sort_task.execute();

    b.iter(|| {
        let mut ttm_task = COOTensorMulDenseMatrix::new(&tensor, &matrix);
        ttm_task.multi_thread = true;
        let _output = black_box(ttm_task.execute().unwrap());
    });
}
