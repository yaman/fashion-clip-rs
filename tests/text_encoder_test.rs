// use embedding_rust_ort::encoders::{get_embedding, BertTokenizer};
// use onnxruntime::{environment::Environment, session::Session};
// use std::path::PathBuf;

// #[test]
// fn test_get_embedding() {
//     // Set up environment and session
//     let environment = Environment::builder().build();
//     let model_path = PathBuf::from("bert-base-uncased.onnx");
//     let session = Session::new(&environment, model_path).unwrap();

//     // Set up tokenizer
//     let vocab_path = PathBuf::from("vocab.txt");
//     let tokenizer = BertTokenizer::from_file(vocab_path).unwrap();

//     // Test query
//     let query = "This is a test query.";

//     // Get embedding
//     let embedding = get_embedding(&tokenizer, &session, query);

//     // Check that embedding has expected length
//     assert_eq!(embedding.len(), 768);
// }
