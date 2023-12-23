FROM ubuntu:latest

COPY ./config.toml /config.toml
COPY ./target/release/fashion-clip-rs /usr/local/bin/fashion-clip-rs
COPY ./target/release/libonnxruntime.so /lib/libonnxruntime.so

ENV ORT_DYLIB_PATH /lib/libonnxruntime.so

ENTRYPOINT ["fashion-clip-rs"]