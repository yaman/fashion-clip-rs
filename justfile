DOCKER_IMAGE := "embed-rs"

build:
    cargo build --release

build-docker:
    docker build -t {{DOCKER_IMAGE}} .

run:
    ORT_DYLIB_PATH=./target/release/libonnxruntime.so cargo run --release

run-docker:
    docker run -p 8888:8888 {{DOCKER_IMAGE}}

perf-test:
    ghz --insecure --enable-compression --proto ./pb/encoder/encoder.proto --call encoder.Encoder.EncodeText -d '{"texts":"{randomString 16 }"}' -c 10 -z 1h --load-schedule=step --load-start=50 --load-end=300 --load-step=10 --load-step-duration=10s 0.0.0.0:50052

unit-test:
    cargo test
