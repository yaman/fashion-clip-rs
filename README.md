
# fashion-clip-rs: Rust-based fashion-clip service

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Rust Version](https://img.shields.io/badge/rust-recent_version-blue)

## 🌟 Introduction
fashion-clip-rs is the onnx ready version of [fashion-clip](https://github.com/patrickjohncyh/fashion-clip) transformers model entirely written in Rust with the help of pykeio/ort. This version leverages the power of Rust for both GRPC services and as a standalone library, providing highly efficient text and image embeddings especially for fashion with multilingual capability.

## 🚀 Features
- **Entirely in Rust:** Re-written for optimal performance.
- **GRPC with Tonic:** Robust and efficient GRPC service.
- **Multilingual Text Embedding:** Utilizing ONNX converted `sentence-transformers/clip-ViT-B-32-multilingual-v1`.
- **Fashion-Focused Image Embedding:** With ONNX converted `patrickjohncyh/fashion-clip`.
- **Cargo for Package Management:** Ensuring reliable dependency management.
- **Built-in Rust Testing:** Leveraging Rust's testing capabilities.
- **GRPC Performance Testing:** With `ghz.sh`.
- **Docker Support:** For containerized deployment.
- **ONNX Runtime with `pykeio/ort` Crate:** For model loading and inference.
- **HF Tokenizers:** For preprocessing in text embedding.
- **Standalone Library Support:** Can be included in other Rust projects.
- **Coverage with Tarpaulin:** For detailed test coverage analysis.

## 🛠 Getting Started

### Prerequisites
Ensure you have the following installed:
- Recent version of Rust
- Docker for containerized deployment
- GHZ for GRPC performance testing
- Tarpaulin for coverage reporting

### 🌐 Installation & Setup
#### Build with Cargo
```bash
cargo build --release
```
#### Build Docker Image
```bash
docker build -t embed-rs .
```
#### Run Locally
```bash
ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo run --release
```
#### Run Docker Container
```bash
docker run -p 50052:50052 embed-rs
```

## 📚 Usage as a library
fashion-clip-rs can also be used as a library in Rust projects.

Add library to your project:
```bash
cargo add fashion_clip_rs
```

given model is exported to onnx with following model structure under models/text:
```
config.json  
model.onnx  
special_tokens_map.json  
tokenizer_config.json  
tokenizer.json  
vocab.txt
```

```rust
use fashion_clip_rs::{config::Config, embed::EmbedText};
let embed_text = EmbedText::new(&"models/text/model.onnx", &"sentence-transformers/clip-ViT-B-32-multilingual-v1").expect("msg");
let query_embedding = embed_text.encode(&"this is a sentence".to_string());

```

## 🧪 Testing

### Performance Testing for Text
```bash
ghz --insecure --enable-compression --proto ./pb/encoder/encoder.proto --call encoder.Encoder.EncodeText -d '{"texts":"{randomString 16 }"}' -c 10 -z 1h --load-schedule=step --load-start=50 --load-end=300 --load-step=10 --load-step-duration=10s 0.0.0.0:50052
```

### Unit Testing
```bash
ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo test
```

### Coverage Reporting
```bash
ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo tarpaulin -o xml --output-dir coverage --skip-clean
```

## 👥 Contributing
Contributions are welcome! Please refer to our [contributing guidelines](LINK_TO_CONTRIBUTING_GUIDELINES) for more information.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE.md](LINK_TO_LICENSE) file for details.

## 📞 Contact
For questions or feedback, please reach out to [Your Contact Information].

# fashion-clip-rs: Advanced Rust gRPC Service for Fashion-Clip Embeddings

fashion-clip-rs is a Rust project that provides a gRPC service for creating embeddings using the Fashion-Clip model. It imports an ONNX file (at the moment, the Fashion-Clip PyTorch library from Hugging Face with an optimum CLI to convert it to ONNX format), creates a gRPC service API to create either text or image embeddings using the Fashion-Clip model, runs inference for the given text or image, and returns the output vectors as a gRPC response.

## Installation

1. Install Rust and Cargo: https://www.rust-lang.org/tools/install
2. Clone the repository: `git clone https://github.com/yaman/fashion-clip-rs.git`
3. Change into the project directory: `cd fashion-clip-rs`
4. Build the project: `cargo build`

## Converting the Fashion-Clip Model to ONNX Format

To use the Fashion-Clip model and clip-ViT-B-32-multilingual-v1 with fashion-clip-rs, you need to convert it to ONNX format using the Hugging Face Optimum tool.

1. install latest optimum cli:
```bash
python -m pip install git+https://github.com/huggingface/optimum.git
```
2. For clip-ViT-B-32-multilingual-v1: 
```bash
optimum-cli export onnx -m sentence-transformers/clip-ViT-B-32-multilingual-v1 --task feature-extraction models/text 
```
3. For fashion-clip:
```bash
optimum-cli export onnx -m patrickjohncyh/fashion-clip --task feature-extraction models/image
```

**Note**: Accurate exporting of **clip-ViT-B-32-multilingual-v1** depends on latest version of optimum. So, do not skip first step even if you have already optimum installed

## gRPC Service

The gRPC service provides two methods:

### EncodeText

Encodes a text input using the Fashion-Clip model.

Request:

```protobuf
message TextRequest {
  string text = 1;
}
```

Response:

```protobuf
message EncoderResponse {
  repeated float embedding = 3;
}
```

### EncodeImage

Encodes an image input using the Fashion-Clip model.

Request:

```protobuf
message ImageRequest {
  bytes image = 2;
}
```

Response:

```protobuf
message EncoderResponse {
  repeated float embedding = 3;
}

```

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Make your changes and commit them: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

This project was created by [Yaman](https://github.com/yaman).

