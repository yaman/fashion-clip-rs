#[cfg(test)]
mod tests {

    pub mod encoder {
        tonic::include_proto!("encoder");
    }

    use approx::assert_abs_diff_eq;
    use embed_rs::config::Config;
    use encoder::encoder_client::EncoderClient;
    use encoder::EncodeTextRequest;
    use tonic::transport::Channel;

    use encoder::EncodeImageRequest;

    use std::fs::File;
    use std::io::Read;
    use std::path::Path;

    fn image_to_bytes<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(buffer)
    }

    async fn setup_service_client() -> (Config, EncoderClient<Channel>) {
        let config = Config::new("config.toml").unwrap();
        let url = config.service.url.clone().to_string();
        let leaked_url = Box::leak(url.into_boxed_str());
        let channel = Channel::from_static(leaked_url).connect().await.unwrap();
        let client = EncoderClient::new(channel);
        (config, client)
    }

    #[tokio::test]
    async fn test_given_encode_text_request_when_sent_valid_text_then_return_embedding() -> Result<(), Box<dyn std::error::Error>> {
        let (config, mut client) = setup_service_client().await;

        let request = tonic::Request::new(EncodeTextRequest {
            text: config.test.text_example.to_string(),
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

    #[tokio::test]
    async fn test_given_encode_text_request_when_sent_empty_text_then_return_error()
    -> Result<(), Box<dyn std::error::Error>> {
        let (_, mut client) = setup_service_client().await;

        let request = tonic::Request::new(EncodeTextRequest { text: "".to_string() });

        let response = client.encode_text(request).await;
        assert!(matches!(
            response,
            Err(ref e) if e.code() == tonic::Code::InvalidArgument && e.message() == "No text provided"
        ));
        Ok(())
    }

    // #[tokio::test]
    async fn test_given_encode_image_request_when_sent_empty_image_then_return_embedding()
    -> Result<(), Box<dyn std::error::Error>> {
        let (config, mut client) = setup_service_client().await;
        let bytes = image_to_bytes("tests/data/test_image.jpg").unwrap();
        println!("Image bytes: {:?}", bytes.len());

        let request = tonic::Request::new(EncodeImageRequest {
            image: bytes,
        });

        let response = client.encode_image(request).await?.into_inner();

        let mut test_file = File::open(config.test.image_embeddings)?;
        let mut test_contents = String::new();
        test_file.read_to_string(&mut test_contents)?;

        let expected: Vec<f32> = serde_json::from_str(&test_contents).unwrap();
        println!("EXPECTED={:?}", expected);
        let actual: Vec<f32> = response.embedding.to_vec();
        println!("ACTUAL={:?}", actual);

        for (exp, act) in expected.iter().zip(actual.iter()) {
            assert_abs_diff_eq!(exp, act, epsilon = 1e-5);
        }
        Ok(())
    }
}
