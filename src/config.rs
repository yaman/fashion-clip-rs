use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Service {
    pub host: String,
    pub port: u16,
    pub url: String,
    pub tcp_keepalive: u16,
    pub http2_keepalive_interval: u16,
    pub http2_keepalive_timeout: u16
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelText {
    pub name: String,
    pub cache_folder: String,
    pub onnx_folder: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelImage {
    pub name: String,
    pub cache_folder: String,
    pub onnx_folder: String,
    pub pretrained_model_folder: Option<String>,
    pub image_width: Option<usize>,
    pub image_height: Option<usize>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Test {
    pub text_embeddings: String,
    pub image_embeddings: String,
    pub text_example: String,
    pub image: String,
    pub image_url: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub service: Option<Service>,
    pub model: Model,
    pub test: Option<Test>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Model {
    pub text: ModelText,
    pub image: ModelImage,
}

impl Config {
    pub fn new(config_file_name: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_file = std::env::var("CONFIG_FILE").unwrap_or(config_file_name.into());
        let config_file = std::path::Path::new(&config_file);
        let config = std::fs::read_to_string(config_file)?;
        let config: Config = toml::from_str(&config)?;
        Ok(config)
    }
}
