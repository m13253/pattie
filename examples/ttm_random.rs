use anyhow::Result;
use clap::{App, Arg};
use pattie::algos::matrix::CreateRandomDenseMatrix;
use pattie::algos::tensor::SortCOOTensor;
use pattie::algos::tensor_matrix::COOTensorMulDenseMatrix;
use pattie::structs::axis::{axes_to_string, AxisBuilder};
use pattie::structs::tensor::COOTensor;
use pattie::traits::Tensor;
use pattie::utils::hint::black_box;
use std::fs::File;
use std::iter;
use std::time::{Duration, Instant};

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

    println!(
        "Input tensor shape:  {}\t({} elements)",
        axes_to_string(tensor.shape()),
        tensor.num_non_zeros()
    );

    let mode = matches.value_of("mode").unwrap().parse::<usize>()?;
    assert!(mode < tensor.shape().len());
    let common_axis = &tensor.shape()[mode];
    let nrows = common_axis.clone();
    let ncols = AxisBuilder::new()
        .range(0..matches.value_of("rank").unwrap().parse::<u32>()?)
        .build();
    let matrix = CreateRandomDenseMatrix::<u32, f32>::new((nrows, ncols), 0.0, 1.0).execute()?;

    println!(
        "Random matrix shape: {}\t({} elements)",
        axes_to_string(matrix.shape()),
        matrix.num_non_zeros()
    );

    let sort_order = tensor
        .sparse_axes()
        .iter()
        .filter(|&ax| ax != common_axis)
        .chain(iter::once(common_axis))
        .cloned()
        .collect::<Vec<_>>();
    println!("Sorting tensor by    {}", axes_to_string(&sort_order));
    SortCOOTensor::new(&mut tensor, &sort_order).execute();

    println!("Warming up...");
    let ttm_task = black_box(COOTensorMulDenseMatrix::new(&tensor, &matrix));
    let output = black_box(ttm_task.execute()?);
    println!(
        "Output tensor shape: {}\t({} elements)",
        axes_to_string(output.shape()),
        output.num_non_zeros()
    );
    drop(output);

    const MIN_ELAPSED_TIME: Duration = Duration::from_secs(3);
    const MIN_ROUNDS: u32 = 5;
    let mut elapsed_time = Duration::ZERO;
    let mut rounds = 0;

    println!("Running benchmark...");
    while elapsed_time < MIN_ELAPSED_TIME || rounds < MIN_ROUNDS {
        let ttm_task = black_box(COOTensorMulDenseMatrix::new(&tensor, &matrix));
        let start_time = Instant::now();
        let _output = black_box(ttm_task.execute()?);
        elapsed_time += start_time.elapsed();
        rounds += 1;
    }
    elapsed_time /= rounds;

    println!("Number of iterations: {}", rounds);
    println!(
        "Time per iteration:   {}.{:09} seconds",
        elapsed_time.as_secs(),
        elapsed_time.subsec_nanos()
    );

    Ok(())
}
