use clap::Parser;

/// Transfer an onnx model file to multiple semantically-equal onnx model files
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(short, long, default_value = "100")]
    /// node limit of equality saturation
    pub node_limit: usize,
    #[clap(short, long, default_value = "5")]
    /// time limit (in second) of equality saturation
    pub time_limit_sec: u64,
    #[clap(short, long, default_value = "200")]
    /// iteration limit of equality saturation
    pub iter_limit: usize,
}
