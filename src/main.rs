mod args;
mod encoder;
mod encoder_service;

use std::net::SocketAddr;

use encoder_service::encoder::encoder_server::EncoderServer;
use ort::{Environment, ExecutionProvider};
use tonic::{codec::CompressionEncoding, transport::Server};

use autometrics::prometheus_exporter;
use axum::{routing::get, Router};

use clap::Parser;

use args::Args;
use encoder::EncoderService;

extern crate num_cpus;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();
    prometheus_exporter::init();

    let args = Args::parse();

    let grpc_addr = "0.0.0.0:50052".parse().unwrap();

    let environment = Environment::builder()
        .with_name("rustembed")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();

    println!("Listening at {:?}", grpc_addr);

    let encoder_service = EncoderService::new(&environment, args.clone())?;
    let server = EncoderServer::new(encoder_service)
        .accept_compressed(CompressionEncoding::Gzip)
        .send_compressed(CompressionEncoding::Gzip);
    
    tokio::spawn(async move {
        Server::builder()
            .add_service(server)
            .serve(grpc_addr)
            .await
            .expect("Failed to start gRPC(rustembed) server");
    });

    let app = Router::new().route(
        "/metrics",
        get(|| async { prometheus_exporter::encode_http_response() }),
    );
    let web_addr: SocketAddr = "0.0.0.0:3000".parse().unwrap();
    axum::Server::bind(&web_addr)
        .serve(app.into_make_service())
        .await
        .expect("Health Chceck server failed");
    Ok(())
}
