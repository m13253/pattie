use anyhow::Result;
use clap::Parser;
use pattie::structs::axis::axes_to_string;
use pattie::structs::tensor::COOTensor;
use pattie::traits::Tensor;
use std::ffi::OsString;
use std::fs::File;

/// COO sparse tensor I/O example
#[derive(Debug, Parser)]
struct Args {
    /// Input tensor file
    #[clap(short, long)]
    input: OsString,

    /// Output tensor file
    #[clap(short, long)]
    output: Option<OsString>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let input_filename = args.input;
    eprintln!("Reading tensor from {}", input_filename.to_string_lossy());
    let mut input_file = File::open(input_filename)?;
    let tensor = COOTensor::<u32, f32>::read_from_text(&mut input_file)?;
    drop(input_file);

    println!(
        "Tensor shape: {}\t({} elements)",
        axes_to_string(tensor.shape()),
        tensor.num_non_zeros()
    );

    if let Some(output_filename) = args.output {
        eprintln!("Writing tensor to {}", output_filename.to_string_lossy());
        let mut output_file = File::create(output_filename)?;
        tensor.write_to_text(&mut output_file)?;
        drop(output_file);
    }

    Ok(())
}
