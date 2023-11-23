#[cfg(test)]
mod tests {

    pub mod encoder {
        tonic::include_proto!("encoder");
    }

    use encoder::encoder_client::EncoderClient;
    use encoder::EncodeTextRequest;
    use pretty_assertions::assert_eq;
    use tonic::transport::Channel;

    #[tokio::test]
    async fn test_encode_text() -> Result<(), Box<dyn std::error::Error>> {
        // let environment = Environment::builder()
        //     .with_name("clip")
        //     .with_execution_providers([ExecutionProvider::CUDA(Default::default())])
        //     .build()?
        //     .into_arc();

        // // Spawn the server on a separate task
        // tokio::spawn(server);

        // // Wait for the server to start
        // tokio::time::sleep(Duration::from_secs(1)).await;

        let channel = Channel::from_static("http://0.0.0.0:50052")
            .connect()
            .await?;

        // Create a client
        let mut client = EncoderClient::new(channel);

        // Create a request
        let request = tonic::Request::new(EncodeTextRequest {
            texts: vec!["This is an example sentence.".into()],
        });

        // Send the request
        let response = client.encode_text(request).await?.into_inner();

        // Print the response
        println!("RESPONSE={:?}", response);
        assert_eq!(response.embedding.len() == 512, false);
        // assert_eq!(
        //     response.embedding[0..5]
        //         .iter()
        //         .flat_map(|e| e.point.clone())
        //         .collect::<Vec<_>>()
        //         == vec![
        //             -1.26431495e-01,
        //             1.03924334e-01,
        //             -9.06252265e-02,
        //             -1.16209365e-01,
        //             -1.91177040e-01
        //         ],
        //     true
        // );

        Ok(())
    }
}
