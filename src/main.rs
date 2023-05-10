#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use std::{
    error::Error,
    io::{BufRead, Write},
};

mod cli;
mod convert;

fn main() -> Result<(), Box<dyn Error>> {
    let args = <cli::Cli as clap::Parser>::parse();
    let json = std::io::stdin().lock().lines().next().ok_or("No input")??;
    let (ft_weights, ft_bias, out_weights, out_bias) =
        convert::from_json(&json, args.qa, args.qb, &args.ft_name, &args.out_name)?;

    if let Some(path) = args.unified {
        let mut file = std::fs::File::create(&path)?;
        let mut contiguous_weights = Vec::with_capacity(
            ft_weights.len() + ft_bias.len() + out_weights.len() + out_bias.len(),
        );
        contiguous_weights.extend_from_slice(&ft_weights);
        contiguous_weights.extend_from_slice(&ft_bias);
        contiguous_weights.extend_from_slice(&out_weights);
        contiguous_weights.extend_from_slice(&out_bias);
        // # Safety
        //
        // The `align_to` method is safe to use here because transmuting i16s to u8s is safe,
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
    }

    if let Some(path) = args.split {
        let byte_slices = [&ft_weights[..], &ft_bias[..], &out_weights[..], &out_bias[..]].into_iter().map(|slice| {
            // # Safety
            //
            // The `align_to` method is safe to use here because transmuting i16s to u8s is safe,
            // and u8 alignment is less strict than i16 alignment.
            unsafe {
                let inner = slice.align_to::<u8>().1;
                if inner.len() != slice.len() * 2 {
                    return Err::<_, Box<dyn Error>>(format!("Could not convert i16 weights to bytes for writing, expected {} bytes, got {}", slice.len() * 2, inner.len()).into());
                }
                Ok(inner)
            }
        });
        for (file_name, slice) in [
            "feature_weights",
            "feature_bias",
            "output_weights",
            "output_bias",
        ]
        .into_iter()
        .zip(byte_slices)
        {
            let slice = slice?;
            let mut file = std::fs::File::create(path.join(file_name))?;
            file.write_all(slice)?;
            println!(
                "Wrote {} bytes to {}",
                slice.len(),
                path.join(file_name).display()
            );
        }
    }

    Ok(())
}
