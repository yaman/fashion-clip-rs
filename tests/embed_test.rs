use std::fs;

use approx::assert_abs_diff_eq;
use embed_rs::config::Config;
use embed_rs::embed::EmbedText;
use serde_json::from_str;

fn setup_text_model() -> (EmbedText, Config) {
    let config = Config::new("config.toml").unwrap();
    (EmbedText::new(&config.model.text.onnx_folder, &config.model.text.name).unwrap(), config)
}

#[test]
fn test_given_encode_when_sentence_then_return_embedding() {
    let (embed, config) = setup_text_model();
    let actual = match embed.encode(&vec!["This is an example sentence.".into()]) {
        Ok(result) => result,
        Err(e) => panic!("Failed to encode sentence: {}", e),
    };
    let file_content = fs::read_to_string(config.test.text_embeddings).unwrap();
    let expected: Vec<f32> = from_str(&file_content).unwrap();

    for (test, response) in expected.iter().zip(actual.iter()) {
        assert_abs_diff_eq!(test, response, epsilon = 1e-5);
    }
}
