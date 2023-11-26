FROM rust:latest as rust-builder

WORKDIR /embed-rs 
COPY . .
RUN apt update && apt install protobuf-compiler -y
RUN cargo install --path .

FROM python:3.11 as python-builder
WORKDIR /embed-rs
COPY . .
RUN python -m pip install --upgrade-strategy eager optimum[onnxruntime]
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/etc/poetry python3 -
ENV PATH="/etc/poetry/bin:$PATH"
RUN poetry install
RUN poetry run convert_text_model
RUN poetry run convert_image_model

FROM ubuntu:latest
COPY --from=rust-builder /usr/local/cargo/bin/embed-rs /usr/local/bin/embed-rs
COPY --from=rust-builder /embed-rs/target/release/libonnxruntime.so /lib/libonnxruntime.so
COPY --from=python-builder /embed-rs/models /models
ENV ORT_DYLIB_PATH /lib/libonnxruntime.so

ENTRYPOINT ["embed-rs"]
