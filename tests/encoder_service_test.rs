#[cfg(test)]
mod tests {

    pub mod encoder {
        tonic::include_proto!("encoder");
    }

    use super::*;
    use encoder::encoder_client::EncoderClient;
    use encoder::encoder_server::{Encoder, EncoderServer};
    use encoder::EncodeTextRequest;
    use ort::{Environment, ExecutionProvider, OrtError};
    use tonic::transport::Channel;

    async fn setup() -> OrtError {
        tracing_subscriber::fmt::init();

        let environment = match Environment::builder()
            .with_name("clip")
            .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
            .build()
        {
            Ok(it) => it,
            Err(err) => return err,
        }
        .into_arc();

        let args = Args::parse();

        println!(
            "Listening at {:?} with {} mode.",
            addr,
            if args.vision_mode { "vision" } else { "text" }
        );

        let server = EncoderService::new(&environment, ())?;

        Server::builder()
            .add_service(EncoderServer::new(server))
            .serve(addr.parse()?)
            .await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_encode_text() -> Result<(), Box<dyn std::error::Error>> {
        let channel = Channel::from_static("http://[::1]:50051").connect().await?;

        // Create a client
        let mut client = EncoderClient::new(channel);

        // Create a request
        let request = tonic::Request::new(EncodeTextRequest {
            texts: vec!["Hello, world!".into()],
        });

        // Send the request
        let response = client.encode_text(request).await?;

        // Print the response
        println!("RESPONSE={:?}", response);

        Ok(())
    }
}
