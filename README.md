# RustEmbed

RustEmbed is a Rust project that provides a gRPC service for creating embeddings using the Fashion-Clip model. It imports an ONNX file (at the moment, the Fashion-Clip PyTorch library from Hugging Face with an optimum CLI to convert it to ONNX format), creates a gRPC service API to create either text or image embeddings using the Fashion-Clip model, runs inference for the given text or image, and returns the output vectors as a gRPC response.

## Installation

1. Install Rust and Cargo: https://www.rust-lang.org/tools/install
2. Clone the repository: `git clone https://github.com/yaman/RustEmbed.git`
3. Change into the project directory: `cd RustEmbed`
4. Build the project: `cargo build`

## Converting the Fashion-Clip Model to ONNX Format

To use the Fashion-Clip model with RustEmbed, you need to convert it to ONNX format using the Hugging Face Optimum tool. Here's how to do it:

1. Install the Hugging Face Optimum tool: `pip install optimum`
2. Download and convert the Fashion-Clip model from Hugging Face: `optimum-cli export onnx --model patrickjohncyh/fashion-clip fashion-clip-onnx --device "cuda"`

## Usage

To run the gRPC service, use the following command:

```bash

cargo run

```

The service listens on port 50052 by default. You can change the port by setting the `LISTEN` environment variable:

```bash

LISTEN=0.0.0.0:50053 cargo run

```

## API

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
message TextResponse {
  repeated float32 embedding = 1;
}
```

### EncodeImage

Encodes an image input using the Fashion-Clip model.

Request:

```protobuf
message ImageRequest {
  bytes image = 1;
}
```

Response:

```protobuf
message ImageResponse {
  repeated float32 embedding = 1;
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