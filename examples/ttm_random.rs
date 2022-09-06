use anyhow::Result;
use clap::Parser;
use log::info;
use pattie::algos::matrix::CreateRandomDenseMatrix;
use pattie::algos::tensor::SortCOOTensor;
use pattie::algos::tensor_matrix::{COOTensorMulDenseMatrix, SemiCOOTensorMulDenseMatrix};
use pattie::structs::axis::{axes_to_string, AxisBuilder};
use pattie::structs::tensor::COOTensor;
use pattie::traits::Tensor;
use pattie::utils::hint::black_box;
use pattie::utils::logger;
use pattie::utils::tracer::Tracer;
use rayon;
use std::ffi::OsString;
use std::fs::File;
use std::iter;
use std::time::{Duration, Instant};

/// Tensor-Times-Matrix multiplication example
#[derive(clap::Parser, Debug)]
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

    /// Algorithm to use
    #[clap(short, long, value_enum)]
    algo: Algo,

    /// Performance tracer output file
    #[clap(long)]
    trace: Option<OsString>,

    /// Enable multi-threading. Number of threads are determined by RAYON_NUM_THREADS or logical cores
    #[clap(short = 't', long)]
    multi_thread: bool,
}

#[derive(clap::ValueEnum, Copy, Clone, Debug)]
enum Algo {
    /// COO tensor times matrix multiplication
    COO,
    /// SemiCOO tensor times matrix multiplication
    SemiCOO,
}

fn main() -> Result<()> {
    logger::init();
    let args = Args::parse();

    let tracer = if let Some(path) = args.trace.as_ref() {
        Tracer::new_to_filename(path).unwrap()
    } else {
        Tracer::new_dummy()
    };

    let input_filename = args.input;
    info!("Reading tensor from {}", input_filename.to_string_lossy());
    let mut input_file = File::open(input_filename)?;
    let mut tensor = COOTensor::<u32, f32>::read_from_text(&mut input_file)?;
    drop(input_file);

    info!(
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

    info!(
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
    info!("Sorting tensor by    {}", axes_to_string(&sort_order));
    SortCOOTensor::new(&mut tensor, &sort_order).execute();

    info!(
        "Warming up... Number of threads: {}",
        if args.multi_thread {
            rayon::current_num_threads()
        } else {
            1
        }
    );
    let mut ttm_task = black_box(COOTensorMulDenseMatrix::new(&tensor, &matrix));
    ttm_task.multi_thread = args.multi_thread;
    let output = black_box(ttm_task.execute()?);
    info!(
        "Output tensor shape: {}\t({} elements)",
        axes_to_string(output.shape()),
        output.num_non_zeros()
    );
    drop(output);

    const MIN_ELAPSED_TIME: Duration = Duration::from_secs(3);
    const MIN_ROUNDS: u32 = 5;
    let mut elapsed_time = Duration::ZERO;
    let mut rounds = 0;

    info!("Running benchmark...");
    while rounds < MIN_ROUNDS || elapsed_time < MIN_ELAPSED_TIME {
        let output = match args.algo {
            Algo::COO => {
                let mut ttm_task =
                    black_box(COOTensorMulDenseMatrix::new(&tensor, &matrix).trace(&tracer));
                ttm_task.multi_thread = args.multi_thread;
                let start_time = Instant::now();
                let output = ttm_task.execute()?;
                elapsed_time += start_time.elapsed();
                output
            }
            Algo::SemiCOO => {
                let mut ttm_task =
                    black_box(SemiCOOTensorMulDenseMatrix::new(&tensor, &matrix).trace(&tracer));
                ttm_task.multi_thread = args.multi_thread;
                let start_time = Instant::now();
                let output = ttm_task.execute()?;
                elapsed_time += start_time.elapsed();
                output
            }
        };
        let _ = black_box(output);
        rounds += 1;
    }
    elapsed_time /= rounds;

    info!("Number of iterations: {}", rounds);
    info!(
        "Time per iteration:   {}.{:09} seconds",
        elapsed_time.as_secs(),
        elapsed_time.subsec_nanos()
    );

    Ok(())
}
