#[cfg(test)]
mod tests {

    pub mod encoder {
        tonic::include_proto!("encoder");
    }

    use approx::assert_abs_diff_eq;
    use encoder::encoder_client::EncoderClient;
    use encoder::EncodeTextRequest;
    use std::fs::File;
    use std::io::Read;
    use tonic::transport::Channel;

    #[tokio::test]
    async fn test_encode_text() -> Result<(), Box<dyn std::error::Error>> {
        let channel = Channel::from_static("http://0.0.0.0:50052")
            .connect()
            .await?;

        let mut client = EncoderClient::new(channel);

        let request = tonic::Request::new(EncodeTextRequest {
            texts: vec!["This is an example sentence.".into()],
        });

        let response = client.encode_text(request).await?.into_inner();

        let mut test_file = File::open("tests/data/text_embeddings.json")?;
        let mut test_contents = String::new();
        test_file.read_to_string(&mut test_contents)?;

        println!("RESPONSE={:?}", response);
        let test_data: Vec<f32> = serde_json::from_str(&test_contents).unwrap();
        let response_data: Vec<f32> = response.embedding.to_vec();

        for (test, response) in test_data.iter().zip(response_data.iter()) {
            assert_abs_diff_eq!(test, response, epsilon = 1e-5);
        }

        Ok(())
    }
}
