DOCKER_IMAGE := "fashion-clip-rs"

build:
    cargo build --release

build-docker:
    docker build -t {{DOCKER_IMAGE}} .

run:
    ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo run --release

run-docker:
    docker run -p 50052:50052 {{DOCKER_IMAGE}}

perf-test-for-text:
    ghz --insecure --enable-compression --proto ./pb/encoder/encoder.proto --call encoder.Encoder.EncodeText -d '{"texts":"{randomString 16 }"}' -c 10 -z 1h --load-schedule=step --load-start=50 --load-end=300 --load-step=10 --load-step-duration=10s 0.0.0.0:50052

check:
    cargo clippy

unit-test:
    ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo test --release --test embed_test --test clip_image_processor_test

integration-test:
    ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo test --release --test encoder_service_integration_test

coverage:
    ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo tarpaulin -o xml --output-dir coverage --skip-clean

watch-test:
    ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo watch -x "test --no-fail-fast" -d 2

watch-run:
    ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo watch -x run

download-models:
    python -m pip install optimum[exporters,onnxruntime]@git+https://github.com/huggingface/optimum.git transformers sentence-transformers yq
    text_model=$(tomlq -r .text_model config.toml)
    image_model=$(tomlq -r .image_model config.toml)
    optimum-cli export onnx -m $text_model --task feature-extraction models/text 
    optimum-cli export onnx -m $image_model --task feature-extraction models/image
