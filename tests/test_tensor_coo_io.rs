use pattie::structs::tensor;
use std::fs::File;
use std::path::Path;

#[test]
fn test_tensor_coo_io() {
    const TENSORS: [&str; 6] = [
        "3d_7.tns",
        "3d_8.tns",
        "3D_12031.tns",
        "3d_dense.tns",
        "3d-24.tns",
        "4d_3_16.tns",
    ];

    for &tensor_file in TENSORS.iter() {
        let mut f = File::open(Path::new("data/tensors").join(tensor_file)).unwrap();
        let tensor = tensor::COO::<f32, u32>::read_from_text(&mut f, 1).unwrap();
        let mut output_buffer = Vec::new();
        tensor.write_to_text(&mut output_buffer, 1).unwrap();
        print!("{}", String::from_utf8_lossy(&output_buffer));
    }
}
