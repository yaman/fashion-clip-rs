mod args;
mod encoder_service;

use embed_rs::embed::EmbedText;
use encoder_service::encoder::encoder_server::EncoderServer;
use tonic::{codec::CompressionEncoding, transport::Server};

use autometrics::prometheus_exporter;

use clap::Parser;

use args::Args;

use crate::encoder_service::EncoderService;

extern crate num_cpus;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();
    prometheus_exporter::init();
    
    let grpc_addr = "0.0.0.0:50052".parse().unwrap();

    println!("Listening at {:?}", grpc_addr);

    let embed_rs = EmbedText::new("models/text/model.onnx")?;
    let encoder_service = EncoderService{embed_text: embed_rs};
    let server = EncoderServer::new(encoder_service)
        .accept_compressed(CompressionEncoding::Gzip)
        .send_compressed(CompressionEncoding::Gzip);

    Server::builder()
        .add_service(server)
        .serve(grpc_addr)
        .await
        .expect("Failed to start gRPC(rustembed) server");

    Ok(())
}
