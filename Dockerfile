FROM rust:slim-buster AS builder

COPY . .
RUN cargo build --release -q

FROM ubuntu:latest

COPY --from=builder ./config.toml /config.toml
COPY --from=builder ./target/embed-rs /usr/local/bin/embed-rs
COPY --from=builder ./target/release/libonnxruntime.so /lib/libonnxruntime.so

ENV ORT_DYLIB_PATH /lib/libonnxruntime.so

ENTRYPOINT ["embed-rs"]