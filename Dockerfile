FROM ubuntu:latest

RUN apt update && apt install curl unzip -y

COPY ./target/embed-rs /usr/local/bin/embed-rs
COPY ./target/release/libonnxruntime.so /lib/libonnxruntime.so
COPY ./entrypoint.sh /entrypoint.sh

ENV ORT_DYLIB_PATH /lib/libonnxruntime.so

ENTRYPOINT ["/entrypoint.sh"]