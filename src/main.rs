#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    path::Path,
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
        dump_unified(&path, &ft_weights, &ft_bias, &out_weights, &out_bias)?;
    }

    if let Some(path) = args.split {
        dump_split(&path, &ft_weights, &ft_bias, &out_weights, &out_bias)?;
    }

    Ok(())
}

fn dump_split(
    path: &Path,
    ft_weights: &[i16],
    ft_bias: &[i16],
    out_weights: &[i16],
    out_bias: &[i16],
) -> Result<(), Box<dyn Error>> {
    const FILE_NAMES: [&str; 4] = [
        "feature_weights.bin",
        "feature_bias.bin",
        "output_weights.bin",
        "output_bias.bin",
    ];
    let byte_slices = [ft_weights, ft_bias, out_weights, out_bias].into_iter().map(|slice| {
        // # Safety
        //
        // The `align_to` method is sound to use here because transmuting i16s to u8s is always sound,
        // and u8 alignment is less strict than i16 alignment.
        unsafe {
            let inner = slice.align_to::<u8>().1;
            if inner.len() != slice.len() * 2 {
                return Err::<_, Box<dyn Error>>(format!("Could not convert i16 weights to bytes for writing, expected {} bytes, got {}", slice.len() * 2, inner.len()).into());
            }
            Ok(inner)
        }
    });
    for (file_name, slice) in FILE_NAMES.into_iter().zip(byte_slices) {
        let slice = slice?;
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
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    let mut contiguous_weights =
        Vec::with_capacity(ft_weights.len() + ft_bias.len() + out_weights.len() + out_bias.len());
    contiguous_weights.extend_from_slice(ft_weights);
    contiguous_weights.extend_from_slice(ft_bias);
    contiguous_weights.extend_from_slice(out_weights);
    contiguous_weights.extend_from_slice(out_bias);
    // # Safety
    //
    // The `align_to` method is sound to use here because transmuting i16s to u8s is always sound,
    // and u8 alignment is less strict than i16 alignment.
    let bytes_to_write = unsafe {
        let inner = contiguous_weights.align_to::<u8>().1;
        if inner.len() != contiguous_weights.len() * 2 {
            return Err(format!(
                "Could not convert i16 weights to bytes for writing, expected {} bytes, got {}",
                contiguous_weights.len() * 2,
                inner.len()
            )
            .into());
        }
        inner
    };
    file.write_all(bytes_to_write)?;
    println!("Wrote {} bytes to {}", bytes_to_write.len(), path.display());
    Ok(())
}
