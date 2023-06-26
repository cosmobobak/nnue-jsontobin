use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about)]
#[allow(clippy::struct_excessive_bools, clippy::option_option)]
pub struct Cli {
    pub input: Option<std::path::PathBuf>,
    /// Output path for a unified converted model (use this if you want a single NNUE binary!)
    #[clap(short, long, value_name = "PATH")]
    pub unified: Option<std::path::PathBuf>,
    /// Output directory for a split converted model.
    #[clap(short, long, value_name = "PATH")]
    pub split: Option<std::path::PathBuf>,
    /// The first quantisation parameter.
    #[clap(long, value_name = "K", default_value = "255")]
    pub qa: i32,
    /// The second quantisation parameter.
    #[clap(long, value_name = "K", default_value = "64")]
    pub qb: i32,
    /// Whether to output the output weights as i16 instead of i8 (legacy support).
    #[clap(long)]
    pub big_out: bool,
}
