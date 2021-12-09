#![cfg(test)]
#![feature(test)]

extern crate test;

use anyhow::Result;
use pattie::structs::tensor;
use std::fs::File;
use std::path::Path;
use tempfile::tempfile;
use test::Bencher;

fn load_then_store_tensor(filename: &Path) -> Result<()> {
    let mut input_file = File::open(filename)?;
    let tensor = tensor::COOTensor::<u32, f32>::read_from_text(&mut input_file)?;
    drop(input_file);
    let mut output_file = tempfile()?;
    tensor.write_to_text(&mut output_file)?;
    drop(output_file);
    Ok(())
}

macro_rules! bench_tensor_io {
    ($name:ident, $filename:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            b.iter(|| load_then_store_tensor(Path::new($filename)).unwrap());
        }
    };
}

bench_tensor_io!(bench_load_3d_7, "data/tensors/3d_7.tns");
bench_tensor_io!(bench_load_3d_8, "data/tensors/3d_8.tns");
bench_tensor_io!(bench_load_3d_12031, "data/tensors/3D_12031.tns");
bench_tensor_io!(bench_load_3d_dense, "data/tensors/3d_dense.tns");
bench_tensor_io!(bench_load_3d_24, "data/tensors/3d-24.tns");
bench_tensor_io!(bench_load_4d_3_16, "data/tensors/4d_3_16.tns");
