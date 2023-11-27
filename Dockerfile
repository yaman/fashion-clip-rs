FROM rust:latest as rust-builder

FROM ubuntu:latest
COPY ./target/embed-rs /usr/local/bin/embed-rs
COPY ./target/release/libonnxruntime.so /lib/libonnxruntime.so
COPY ./models /models
ENV ORT_DYLIB_PATH /lib/libonnxruntime.so
ENTRYPOINT ["embed-rs"]