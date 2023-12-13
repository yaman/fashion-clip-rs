mod config;
mod encoder_service;
mod clip_image_processor;

use std::time::Duration;

use autometrics::prometheus_exporter;
use fashion_clip_rs::embed::{EmbedText, EmbedImage};
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
    let embed_text = EmbedText::new(&config.model.text.onnx_folder, &config.model.text.name)?;
    let embed_image = EmbedImage::new(&config.model.image.onnx_folder)?;
    // configure gRPC service
    let encoder_service = EncoderService {
        embed_text,
        embed_image
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
        .http2_keepalive_interval(Some(Duration::from_secs(config.service.http2_keepalive_interval.into())))
        .http2_keepalive_timeout(Some(Duration::from_secs(config.service.http2_keepalive_timeout.into())))
        .add_service(health_service)
        .add_service(server)
        
        .serve(grpc_addr)
        .await
        .expect("Failed to start gRPC(rustembed) server");

    Ok(())
}
