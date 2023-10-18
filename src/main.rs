mod args;
mod encoder;
mod encoder_service;

use encoder_service::encoder::encoder_server::EncoderServer;
use ort::{Environment, ExecutionProvider};
use tonic::transport::Server;

use clap::Parser;

use args::Args;
use encoder::EncoderService;

extern crate num_cpus;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let addr: &String = &args.listen;

    let environment = Environment::builder()
        .with_name("clip")
        .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
        .build()?
        .into_arc();

    println!("Listening at {:?}", addr);

    let server = EncoderService::new(&environment, args.clone())?;

    Server::builder()
        .add_service(EncoderServer::new(server))
        .serve(addr.parse()?)
        .await?;

    Ok(())
}
