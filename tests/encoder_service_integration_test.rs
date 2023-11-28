#[cfg(test)]
mod tests {

    pub mod encoder {
        tonic::include_proto!("encoder");
    }

    use approx::assert_abs_diff_eq;
    use embed_rs::config::Config;
    use encoder::encoder_client::EncoderClient;
    use encoder::EncodeTextRequest;
    use std::fs::File;
    use std::io::Read;
    use tonic::transport::Channel;

    fn setup_text_model() -> Config {
       Config::new("config.toml").unwrap()
    }

    #[tokio::test]
    async fn test_encode_text() -> Result<(), Box<dyn std::error::Error>> {
        let config = setup_text_model();
        let url = config.service.url.clone().to_string();
        let leaked_url = Box::leak(url.into_boxed_str());
        let channel = Channel::from_static(leaked_url).connect().await?;

        let mut client = EncoderClient::new(channel);

        let request = tonic::Request::new(EncodeTextRequest {
            texts: vec![config.test.text_example],
        });

        let response = client.encode_text(request).await?.into_inner();

        let mut test_file = File::open(config.test.text_embeddings)?;
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
