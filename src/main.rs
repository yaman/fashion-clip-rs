use embedding_rust_ort::{
    command::commmand::Args,
    encoder_service::{create_encoder_service, encoder::encoder_server::EncoderServer},
};
use tonic::transport::Server;
pub mod encoder {
    tonic::include_proto!("encoder");
}

use ort::{Environment, ExecutionProvider};

extern crate image;

extern crate num_cpus;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    let addr: &String = &args.get_listen();

    let environment = Environment::builder()
        .with_name("clip")
        .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
        .build()?
        .into_arc();

    println!(
        "Listening at {:?} with {} mode.",
        addr,
        if args.get_vision_mode() {
            "vision"
        } else {
            "text"
        }
    );

    // Create a new EncoderService instance and store it in the OnceCell
    let encoder_service = create_encoder_service(&environment, true, false, 224);
    let encoder_server = EncoderServer::new(encoder_service);

    Server::builder()
        .add_service(encoder_server)
        .serve(addr.parse()?)
        .await?;

    Ok(())
}
