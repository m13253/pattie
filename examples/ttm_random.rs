use anyhow::Result;
use clap::{App, Arg};
use pattie::algos::matrix::create_random_dense_matrix;
use pattie::algos::tensor::SortCOOTensor;
use pattie::algos::tensor_matrix::COOTensorMulDenseMatrix;
use pattie::structs::axis::axes_to_string;
use pattie::structs::tensor::COOTensor;
use pattie::traits::Tensor;
use pattie::utils::hint::black_box;
use std::fs::File;
use std::iter;
use std::time::Instant;

fn main() -> Result<()> {
    let matches = App::new("COO sparse tensor I/O example")
        .arg(
            Arg::with_name("input")
                .short("i")
                .long("input")
                .takes_value(true)
                .required(true)
                .help("Input tensor file"),
        )
        .arg(
            Arg::with_name("mode")
                .short("m")
                .long("mode")
                .takes_value(true)
                .required(true)
                .help("Common axis of the tensor, indexing from 0"),
        )
        .arg(
            Arg::with_name("rank")
                .short("r")
                .long("rank")
                .takes_value(true)
                .required(true)
                .help("Number of columns in the matrix"),
        )
        .get_matches();

    let input_filename = matches.value_of_os("input").unwrap();
    eprintln!("Reading tensor from {}", input_filename.to_string_lossy());
    let mut input_file = File::open(input_filename)?;
    let mut tensor = COOTensor::<u32, f32>::read_from_text(&mut input_file)?;
    drop(input_file);

    println!("Input tensor shape:  {}", axes_to_string(tensor.shape()));

    let mode = matches.value_of("mode").unwrap().parse::<usize>()?;
    assert!(mode < tensor.shape().len());
    let common_axis = &tensor.shape()[mode];
    let nrows = common_axis.clone();
    let ncols = 0..matches.value_of("rank").unwrap().parse::<u32>()?;
    let matrix = create_random_dense_matrix::<u32, f32, _, _>((nrows, ncols), 0.0, 1.0)?;

    println!("Random matrix shape: {}", axes_to_string(matrix.shape()));

    let sort_order = iter::once(common_axis)
        .chain(tensor.sparse_axes().iter().filter(|&ax| ax != common_axis))
        .cloned()
        .collect::<Vec<_>>();
    SortCOOTensor::new()
        .prepare(&mut tensor, &sort_order)
        .execute();

    let ttm_task = black_box(COOTensorMulDenseMatrix::new().prepare(&tensor, &matrix)?);
    let start_time = Instant::now();
    let output = black_box(ttm_task.execute()?);
    let elapsed_time = start_time.elapsed();

    println!("Output tensor shape: {}", axes_to_string(output.shape()));
    println!(
        "Elapsed time: {}.{:09} seconds",
        elapsed_time.as_secs(),
        elapsed_time.subsec_nanos()
    );

    Ok(())
}
