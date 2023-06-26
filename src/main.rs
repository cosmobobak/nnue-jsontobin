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

    if args.unified.is_none() && args.split.is_none() {
        return Err("No output path specified, try --unified <PATH> or --split <PATH> (you probably want --unified)".into());
    }

    let json = fs::read_to_string(input_path)?;
    let convert::QuantisedMergedNetwork {
        feature_weights: ft_weights,
        feature_bias: ft_bias,
        output_weights: out_weights,
        output_bias: out_bias,
    } = convert::from_json(&json, args.qa, args.qb)?;

    if let Some(path) = args.unified {
        dump_unified(
            &path,
            &ft_weights,
            &ft_bias,
            &out_weights,
            &out_bias,
            args.big_out,
        )?;
    }

    if let Some(path) = args.split {
        dump_split(
            &path,
            &ft_weights,
            &ft_bias,
            &out_weights,
            &out_bias,
            args.big_out,
        )?;
    }

    Ok(())
}

fn dump_split(
    path: &Path,
    ft_weights: &[i16],
    ft_bias: &[i16],
    out_weights: &[i16],
    out_bias: &[i16],
    big_out: bool,
) -> Result<(), Box<dyn Error>> {
    const FILE_NAMES: [&str; 4] = [
        "feature_weights.bin",
        "feature_bias.bin",
        "output_weights.bin",
        "output_bias.bin",
    ];
    let ft_weights =
        unsafe { slice::from_raw_parts(ft_weights.as_ptr().cast::<u8>(), ft_weights.len() * 2) };
    let ft_bias =
        unsafe { slice::from_raw_parts(ft_bias.as_ptr().cast::<u8>(), ft_bias.len() * 2) };
    let out_weights = unsafe {
        if big_out {
            slice::from_raw_parts(out_weights.as_ptr().cast::<u8>(), out_weights.len() * 2)
        } else {
            let out_weights = out_weights
                .iter()
                .map(|&i| i8::try_from(i).unwrap())
                .collect::<Vec<_>>();
            // just leak the vec and cast it to a byte slice
            let vec = ManuallyDrop::new(out_weights);
            let ptr = vec.as_ptr().cast::<u8>();
            let len = vec.len();
            slice::from_raw_parts(ptr, len)
        }
    };
    let out_bias =
        unsafe { slice::from_raw_parts(out_bias.as_ptr().cast::<u8>(), out_bias.len() * 2) };
    let byte_slices = [ft_weights, ft_bias, out_weights, out_bias].into_iter();
    for (file_name, slice) in FILE_NAMES.into_iter().zip(byte_slices) {
        let mut file = File::create(path.join(file_name))?;
        file.write_all(slice)?;
        println!(
            "Wrote {} bytes to {}",
            slice.len(),
            path.join(file_name).display()
        );
    }
    Ok(())
}

fn dump_unified(
    path: &Path,
    ft_weights: &[i16],
    ft_bias: &[i16],
    out_weights: &[i16],
    out_bias: &[i16],
    big_out: bool,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;

    let ft_weights =
        unsafe { slice::from_raw_parts(ft_weights.as_ptr().cast::<u8>(), ft_weights.len() * 2) };
    let ft_bias =
        unsafe { slice::from_raw_parts(ft_bias.as_ptr().cast::<u8>(), ft_bias.len() * 2) };
    let out_weights = unsafe {
        if big_out {
            slice::from_raw_parts(out_weights.as_ptr().cast::<u8>(), out_weights.len() * 2)
        } else {
            let out_weights = out_weights
                .iter()
                .map(|&i| i8::try_from(i).unwrap())
                .collect::<Vec<_>>();
            // just leak the vec and cast it to a byte slice
            let vec = ManuallyDrop::new(out_weights);
            let ptr = vec.as_ptr().cast::<u8>();
            let len = vec.len();
            slice::from_raw_parts(ptr, len)
        }
    };
    let out_bias =
        unsafe { slice::from_raw_parts(out_bias.as_ptr().cast::<u8>(), out_bias.len() * 2) };
    let byte_slices = [ft_weights, ft_bias, out_weights, out_bias];
    for slice in byte_slices {
        file.write_all(slice)?;
    }
    println!(
        "Wrote {} bytes to {}",
        byte_slices.into_iter().map(<[_]>::len).sum::<usize>(),
        path.display()
    );
    Ok(())
}
