use std::fs::File;

use anyhow::Result;
use clap::{App, Arg};
use pattie::structs::tensor;

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
            Arg::with_name("output")
                .short("o")
                .long("output")
                .takes_value(true)
                .required(true)
                .help("Output tensor file"),
        )
        .get_matches();

    let input_filename = matches.value_of_os("input").unwrap();
    eprintln!("Reading tensor from {}", input_filename.to_string_lossy());
    let mut input_file = File::open(input_filename)?;
    let tensor = tensor::COOTensor::<u32, f32>::read_from_text(&mut input_file)?;
    drop(input_file);

    let output_filename = matches.value_of_os("output").unwrap();
    eprintln!("Writing tensor to {}", output_filename.to_string_lossy());
    let mut output_file = File::create(output_filename)?;
    tensor.write_to_text(&mut output_file)?;
    drop(output_file);

    Ok(())
}
