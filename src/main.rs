mod args;
mod config;
mod encoder_service;

use autometrics::prometheus_exporter;
use embed_rs::embed::EmbedText;
use tonic::{codec::CompressionEncoding, transport::Server};

use crate::{
    config::Config,
    encoder_service::{encoder::encoder_server::EncoderServer, EncoderService},
};

extern crate num_cpus;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();
    prometheus_exporter::init();
    run_server().await
}

pub async fn run_server() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = Config::new("config.toml").expect("Failed to read config file: config.toml");
    // create embed-rs instance
    let embed_rs = EmbedText::new(&config.model.text.onnx_folder, &config.model.text.name)?;

    // configure gRPC service
    let encoder_service = EncoderService {
        embed_text: embed_rs,
    };
    let server = EncoderServer::new(encoder_service)
        .accept_compressed(CompressionEncoding::Gzip)
        .send_compressed(CompressionEncoding::Gzip);

    let grpc_addr = (config.service.host + ":" + &config.service.port.to_string())
        .parse()
        .unwrap();

    println!("Listening at {:?}", grpc_addr);
    let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_serving::<EncoderServer<EncoderService>>()
        .await; // start grpc service

    Server::builder()
        .add_service(health_service)
        .add_service(server)
        .serve(grpc_addr)
        .await
        .expect("Failed to start gRPC(rustembed) server");

    Ok(())
}
