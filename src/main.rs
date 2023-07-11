#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    mem::ManuallyDrop,
    path::Path,
    slice,
};

mod cli;
mod convert;

fn main() {
    if let Err(e) = run() {
        eprintln!("[ERROR] {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = <cli::Cli as clap::Parser>::parse();

    let Some(input_path) = args.input else {
        return Err("No input file specified.".into());
    };

    let Some(output_path) = args.output else {
        return Err("No output path specified, try --output <PATH>".into());
    };

    let json = fs::read_to_string(input_path)?;
    let convert::QuantisedMergedNetwork {
        feature_weights,
        feature_bias,
        output_weights,
        output_bias,
    } = convert::from_json(&json, args.qa, args.qb)?;

    dump(
        &output_path,
        &feature_weights,
        &feature_bias,
        &output_weights,
        &output_bias,
        args.big_out,
    )?;

    Ok(())
}

fn dump<'a>(
    path: &Path,
    feature_weights: &'a [i16],
    feature_bias: &'a [i16],
    output_weights: &'a [i16],
    output_bias: &'a [i16],
    big_out: bool,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;

    let feature_weights = unsafe {
        slice::from_raw_parts::<'a, u8>(feature_weights.as_ptr().cast::<u8>(), feature_weights.len() * 2)
    };
    let feature_bias = unsafe {
        slice::from_raw_parts::<'a, u8>(feature_bias.as_ptr().cast::<u8>(), feature_bias.len() * 2)
    };
    let output_weights = unsafe {
        if big_out {
            slice::from_raw_parts::<'a, u8>(
                output_weights.as_ptr().cast::<u8>(),
                output_weights.len() * 2,
            )
        } else {
            let output_weights = output_weights
                .iter()
                .map(|&i| i8::try_from(i).unwrap())
                .collect::<Vec<_>>();
            // just leak the vec and cast it to a byte slice
            let vec = ManuallyDrop::new(output_weights);
            let ptr = vec.as_ptr().cast::<u8>();
            let len = vec.len();
            slice::from_raw_parts::<'static, u8>(ptr, len)
        }
    };
    let output_bias = unsafe {
        slice::from_raw_parts::<'a, u8>(output_bias.as_ptr().cast::<u8>(), output_bias.len() * 2)
    };
    let byte_slices = [feature_weights, feature_bias, output_weights, output_bias];
    for slice in byte_slices {
        file.write_all(slice)?;
    }
    // determine zero-padding for 64-byte alignment
    let remainder = byte_slices.into_iter().map(<[_]>::len).sum::<usize>() % 64;
    if remainder != 0 {
        let padding = [0u8; 64];
        file.write_all(&padding[..(64 - remainder)])?;
    }

    println!(
        "Wrote {} bytes to {}",
        byte_slices.into_iter().map(<[_]>::len).sum::<usize>(),
        path.display()
    );
    Ok(())
}
