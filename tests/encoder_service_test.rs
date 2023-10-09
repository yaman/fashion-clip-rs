#[cfg(test)]
mod tests {

    pub mod encoder {
        tonic::include_proto!("encoder");
    }

    use encoder::encoder_client::EncoderClient;
    use encoder::EncodeTextRequest;
    use tonic::transport::Channel;

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
        let response = client.encode_text(request).await?.into_inner();

        // Print the response
        println!("RESPONSE={:?}", response);
        assert!(response.embedding.len() > 100);

        Ok(())
    }
}
