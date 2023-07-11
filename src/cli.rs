use clap::Parser;

#[derive(Parser)]
#[clap(author, version, about)]
#[allow(clippy::struct_excessive_bools, clippy::option_option)]
pub struct Cli {
    pub input: Option<std::path::PathBuf>,
    /// Output path for the network binary.
    #[clap(short, long, value_name = "PATH")]
    pub output: Option<std::path::PathBuf>,
    /// The first quantisation parameter.
    #[clap(long, value_name = "K", default_value = "255")]
    pub qa: i32,
    /// The second quantisation parameter.
    #[clap(long, value_name = "K", default_value = "64")]
    pub qb: i32,
    /// Whether to output the output weights as i16 instead of i8 (legacy support).
    #[clap(long)]
    pub big_out: bool,
    /// Whether to omit the binary header.
    #[clap(long)]
    pub no_header: bool,
    /// Name of the network.
    #[clap(long, value_name = "NAME")]
    pub name: Option<String>,
}
