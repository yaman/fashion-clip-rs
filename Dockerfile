FROM ubuntu:latest

RUN ls -althr
COPY ./target/embed-rs /usr/local/bin/embed-rs
COPY ./target/release/libonnxruntime.so /lib/libonnxruntime.so
COPY ./config.toml /config.toml

ENV ORT_DYLIB_PATH /lib/libonnxruntime.so

ENTRYPOINT ["embed-rs"]