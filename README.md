
# fashion-clip-rs: fashion-clip service in Rust

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Rust Version](https://img.shields.io/badge/rust-recent_version-blue)

## 🌟 Introduction
fashion-clip-rs is the onnx ready version of [fashion-clip](https://github.com/patrickjohncyh/fashion-clip) transformers model entirely written in Rust with the help of pykeio/ort. It imports an ONNX file (at the moment, the Fashion-Clip PyTorch library from Hugging Face with an optimum CLI to convert it to ONNX format), creates a gRPC service API to create either text or image embeddings using the Fashion-Clip model and clip-ViT-B-32-multilingual-v1, runs inference for the given text or image, and returns the output vectors as a gRPC response.

fashion-clip-rs provides highly efficient text and image embeddings especially for fashion with multilingual capability.

This project can be used as a standalone library to include rust projects.


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
- [Just](https://github.com/casey/just)
- Docker
- [ghz](https://ghz.sh/) for GRPC performance testing
- [Tarpaulin](https://crates.io/crates/cargo-tarpaulin) for coverage reporting
- python >3.11 to export onnx model using hf optimum

## Installation

1. Install Rust and Cargo: https://www.rust-lang.org/tools/install
2. Install [Just](https://github.com/casey/just)
3. Install [Tarpaulin](https://crates.io/crates/cargo-tarpaulin) *Optional: for coverage reports*
4. Install [ghz](https://ghz.sh/) *Optional: for performance testing*
5. Clone the repository: `git clone https://github.com/yaman/fashion-clip-rs.git`
6. Change into the project directory: `cd fashion-clip-rs`
7. Build the project: `just build`

## Model Export

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

**Note 1**: Accurate exporting of **clip-ViT-B-32-multilingual-v1** depends on latest version of optimum. So, do not skip first step even if you have already optimum installed

**Note 2**: At the moment, we are using *clip-ViT-B-32-multilingual-v1* to generate **text** embeddings. *fashion-clip* to generate **image** embeddings.

## Setup(Build & Run)

### Build
```bash
just build
```
### Build Docker Image
```bash
just build-docker
```
### Run Locally
```bash
just run
```
### Run Docker Container
```bash
just run-docker
```

## 🧪 Testing
### Unit Testing
```bash
just unit-test
```

### Integration Testing
```bash
just integration-test
```

### Coverage Reporting
```bash
just coverage
```

### Performance Testing for Text
```bash
just perf-test-for-text
```


## 📚 Usage as a library
fashion-clip-rs can also be used as a library in Rust projects.

**Note**: models must be ready under models/text and models/image directories. Check [Model Export section](#model-export)

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


## 📜 License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 📞 Contact
For questions or feedback, please reach out to [yaman](https://github.com/yaman).

## Author

This project was created by [Yaman](https://github.com/yaman).

