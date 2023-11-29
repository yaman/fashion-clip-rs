FROM ubuntu:latest

RUN apt update && apt install curl unzip -y

COPY ./target/embed-rs /usr/local/bin/embed-rs
COPY ./target/release/libonnxruntime.so /lib/libonnxruntime.so
COPY ./entrypoint.sh /entrypoint.sh
RUN curl -L https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.23/grpc_health_probe-linux-amd64 -o /bin/grpc_health_probe
RUN chmod +x /bin/grpc_health_probe

ENV ORT_DYLIB_PATH /lib/libonnxruntime.so

ENTRYPOINT ["/entrypoint.sh"]