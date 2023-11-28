use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(author = "canavar", version = "0.1", about = "Embedding Service", long_about = None)]
pub struct Args {
    /// Address to listen
    #[arg(short, long, default_value = "0.0.0.0:50052")]
    pub listen: String,

    /// Vision model input image size, default 224
    #[arg(short, long, default_value_t = 224)]
    pub input_image_size: usize,

    /// Whether to pad and truncate the input text token sequence to 77
    #[arg(short, long, default_value_t = true)]
    pub pad_token_sequence: bool,
    
}
