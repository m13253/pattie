use anyhow::Result;
use clap::Parser;
use pattie::algos::matrix::CreateRandomDenseMatrix;
use pattie::algos::tensor::SortCOOTensor;
use pattie::algos::tensor_matrix::COOTensorMulDenseMatrix;
use pattie::structs::axis::{axes_to_string, AxisBuilder};
use pattie::structs::tensor::COOTensor;
use pattie::traits::Tensor;
use pattie::utils::hint::black_box;
use std::ffi::OsString;
use std::fs::File;
use std::iter;
use std::time::{Duration, Instant};

/// Tensor-Times-Matrix multiplication example
#[derive(Debug, Parser)]
struct Args {
    /// Input tensor file
    #[clap(short, long)]
    input: OsString,

    /// Common axis of the tensor, indexing from 0
    #[clap(short, long)]
    mode: usize,

    /// Number of columns in the matrix
    #[clap(short, long)]
    rank: u32,

    /// Whether to print time of each step inside TTM
    #[clap(long)]
    debug_step_timing: bool,

    /// Enable multi-threading. Number of threads are determined by RAYON_NUM_THREADS or logical cores
    #[clap(short = 't', long)]
    multi_thread: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input_filename = args.input;
    eprintln!("Reading tensor from {}", input_filename.to_string_lossy());
    let mut input_file = File::open(input_filename)?;
    let mut tensor = COOTensor::<u32, f32>::read_from_text(&mut input_file)?;
    drop(input_file);

    println!(
        "Input tensor shape:  {}\t({} elements)",
        axes_to_string(tensor.shape()),
        tensor.num_non_zeros()
    );

    let mode = args.mode;
    assert!(mode < tensor.shape().len());
    let common_axis = &tensor.shape()[mode];
    let nrows = common_axis.clone();
    let ncols = AxisBuilder::new().range(0..args.rank).build();
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
    let mut ttm_task = black_box(COOTensorMulDenseMatrix::new(&tensor, &matrix));
    ttm_task.multi_thread = args.multi_thread;
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
    while rounds < MIN_ROUNDS || (!args.debug_step_timing && elapsed_time < MIN_ELAPSED_TIME) {
        let mut ttm_task = black_box(COOTensorMulDenseMatrix::new(&tensor, &matrix));
        ttm_task.debug_step_timing = args.debug_step_timing;
        ttm_task.multi_thread = args.multi_thread;
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
