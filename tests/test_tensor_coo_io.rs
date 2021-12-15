#![cfg(test)]

use anyhow::Result;
use pattie::structs::tensor;
use std::fs::File;
use std::path::Path;
use std::str;

fn load_then_store_tensor(filename: &Path) -> Result<()> {
    let mut input_file = File::open(filename)?;
    let tensor = tensor::COOTensor::<u32, f32>::read_from_text(&mut input_file)?;
    drop(input_file);
    let mut output_buffer = Vec::new();
    tensor.write_to_text(&mut output_buffer)?;
    print!("{}", str::from_utf8(&output_buffer)?);
    drop(output_buffer);
    Ok(())
}

macro_rules! test_tensor_io {
    ($name:ident, $filename:expr $(,)?) => {
        #[test]
        fn $name() {
            load_then_store_tensor(Path::new($filename)).unwrap();
        }
    };
}

test_tensor_io!(test_load_3d_7, "data/tensors/3d_7.tns");
test_tensor_io!(test_load_3d_8, "data/tensors/3d_8.tns");
test_tensor_io!(test_load_3d_12031, "data/tensors/3D_12031.tns");
test_tensor_io!(test_load_3d_dense, "data/tensors/3d_dense.tns");
test_tensor_io!(test_load_3d_24, "data/tensors/3d-24.tns");
test_tensor_io!(test_load_4d_3_16, "data/tensors/4d_3_16.tns");
