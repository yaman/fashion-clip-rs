use approx::assert_abs_diff_eq;
use fashion_clip_rs::config::Config;
use fashion_clip_rs::embed::{EmbedImage, EmbedText};
use serde_json::from_str;

fn setup_text_model() -> (EmbedText, Config) {
    let config = Config::new("config.toml").unwrap();
    (
        EmbedText::new(&config.model.text.onnx_folder, &config.model.text.name).unwrap(),
        config,
    )
}

#[test]
fn test_given_encode_when_sentence_then_return_embedding() {
    let (embed, config) = setup_text_model();
    let actual = match embed.encode(&"This is an example sentence.".to_string()) {
        Ok(result) => result,
        Err(e) => panic!("Failed to encode sentence: {}", e),
    };
    let file_content = fs::read_to_string(config.test.text_embeddings).unwrap();
    let expected: Vec<f32> = from_str(&file_content).unwrap();

    for (test, response) in expected.iter().zip(actual.iter()) {
        assert_abs_diff_eq!(test, response, epsilon = 1e-5);
    }
}

#[test]
fn test_given_encode_when_sentence_is_none_then_return_error() {
    let (embed, _) = setup_text_model();
    let actual = embed.encode(&"".to_string());
    assert!(actual.is_err());
}

fn setup_image_model() -> (EmbedImage, Config) {
    let config = Config::new("config.toml").unwrap();
    (
        EmbedImage::new(
            &config.model.image.onnx_folder
        )
        .unwrap(),
        config,
    )
}

use std::fs::{self, File};
use std::io::Read;
use std::path::Path;

fn image_to_bytes<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    Ok(buffer)
}

#[test]
fn test_given_encode_when_image_then_return_embedding() {
    let (embed, config) = setup_image_model();
    let image_bytes = &image_to_bytes(&config.test.image).unwrap();
    println!("image bytes: {:?}", image_bytes.len());
    let actual = match embed.encode(image_bytes.clone() as Vec<u8>) {
        Ok(result) => result,
        Err(e) => panic!("Failed to encode sentence: {}", e),
    };
    let file_content = fs::read_to_string(config.test.image_embeddings).unwrap();
    let expected: Vec<f32> = from_str(&file_content).unwrap();

    for (test, response) in expected.iter().zip(actual.iter()) {
        assert_abs_diff_eq!(test, response, epsilon = 1e-5);
    }
}
