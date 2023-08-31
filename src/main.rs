#![warn(clippy::all, clippy::pedantic, clippy::nursery)]

use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    mem::ManuallyDrop,
    path::Path,
    slice,
};

use cbnf_rs::CBNFHeader;

const HALF_KA: u8 = 0x00;
const PERSPECTIVE: u8 = 0x01;
const NET_NAME_MAXLEN: usize = 48;

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

    let net_name = match args.name {
        Some(name) => name,
        None => {
            if args.no_header {
                String::new()
            } else {
                return Err("No network name specified for CBNF header, try --name <NAME>".into());
            }
        }
    };

    if net_name.len() > NET_NAME_MAXLEN {
        return Err(format!(
            "Network name too long, must be {NET_NAME_MAXLEN} characters or less."
        )
        .into());
    }

    let json = fs::read_to_string(input_path)?;
    let convert::QuantisedMergedNetwork {
        feature_weights,
        feature_bias,
        output_weights,
        output_bias,
        psqt_weights,
        has_buckets,
        hidden_size,
    } = convert::from_json(&json, args.qa, args.qb)?;

    let arch = if has_buckets { HALF_KA } else { PERSPECTIVE };

    let mut name_buffer = [0u8; NET_NAME_MAXLEN];
    name_buffer[..net_name.len()].copy_from_slice(net_name.as_bytes());
    let cbnf_header = CBNFHeader {
        magic: *b"CBNF",
        version: 1,
        flags: 0,
        padding: 0,
        arch,
        activation: 0,
        hidden_size: hidden_size.try_into().unwrap(),
        input_buckets: if has_buckets { 64 } else { 1 },
        output_buckets: 1,
        name_len: net_name.len().try_into().unwrap(),
        name: name_buffer,
    };

    dump(
        &output_path,
        if args.no_header {
            None
        } else {
            Some(&cbnf_header)
        },
        &feature_weights,
        &feature_bias,
        &output_weights,
        &output_bias,
        psqt_weights.as_deref(),
        args.big_out,
    )?;

    Ok(())
}

fn to_bytes(shorts: &[i16]) -> &[u8] {
    unsafe {
        let (a, b, c) = shorts.align_to();
        assert!(a.is_empty());
        assert!(c.is_empty());
        b
    }
}

#[allow(clippy::too_many_arguments)]
fn dump<'a>(
    path: &Path,
    cbnf_header: Option<&CBNFHeader>,
    feature_weights: &'a [i16],
    feature_bias: &'a [i16],
    output_weights: &'a [i16],
    output_bias: &'a [i16],
    psqt_weights: Option<&'a [i16]>,
    big_out: bool,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    if let Some(cbnf_header) = cbnf_header {
        file.write_all(cbnf_header.as_bytes())?;
    }

    let feature_weights = to_bytes(feature_weights);
    let feature_bias = to_bytes(feature_bias);
    let output_weights = unsafe {
        if big_out {
            to_bytes(output_weights)
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
    let output_bias = to_bytes(output_bias);
    let psqt_weights = psqt_weights.map_or([].as_slice(), |psqt_weights| to_bytes(psqt_weights));
    let byte_slices = [
        feature_weights,
        feature_bias,
        output_weights,
        psqt_weights,
        output_bias,
    ];
    for slice in byte_slices {
        file.write_all(slice)?;
    }
    // determine zero-padding for 64-byte alignment
    // CBNF header is ignored as it is always a multiple of 64 bytes
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
