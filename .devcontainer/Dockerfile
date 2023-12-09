FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.11
ENV TZ=Europe/Istanbul

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/conda/lib"

ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false


RUN apt update && apt install -y bash \
    build-essential \
    git \
    ca-certificates \
    wget \
    curl \
    iputils-ping \
    jq \
    libncurses5-dev \
    libevent-dev \
    net-tools \
    netcat-openbsd \
    rubygems \
    ruby-dev \
    silversearcher-ag \
    socat \
    software-properties-common \
    tmux \
    tzdata \
    zsh \
    protobuf-compiler \
    pkg-config \
    libssl-dev \
    apt-transport-https \
    && rm -rf /var/lib/apt/lists


RUN curl -o packages-microsoft-prod.deb https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb

RUN dpkg -i packages-microsoft-prod.deb

RUN apt update && apt install -y  bash \
    msopenjdk-21 \
    && rm -rf /var/lib/apt/lists \
    && rm packages-microsoft-prod.deb

# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /opt/conda \
    && rm miniconda3.sh \
    && /opt/conda/bin/conda install -y -c anaconda \
    python=$PYTHON_VERSION \
    && /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda config --set ssl_verify False \
    && pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    && ln -s /opt/conda/bin/pip /usr/local/bin/pip3

RUN /opt/conda/bin/conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install requirements
COPY ./.devcontainer/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=nightly

RUN set -eux; \
    dpkgArch="$(dpkg --print-architecture)"; \
    case "${dpkgArch##*-}" in \
    amd64) rustArch='x86_64-unknown-linux-gnu'; rustupSha256='0b2f6c8f85a3d02fde2efc0ced4657869d73fccfce59defb4e8d29233116e6db' ;; \
    armhf) rustArch='armv7-unknown-linux-gnueabihf'; rustupSha256='f21c44b01678c645d8fbba1e55e4180a01ac5af2d38bcbd14aa665e0d96ed69a' ;; \
    arm64) rustArch='aarch64-unknown-linux-gnu'; rustupSha256='673e336c81c65e6b16dcdede33f4cc9ed0f08bde1dbe7a935f113605292dc800' ;; \
    i386) rustArch='i686-unknown-linux-gnu'; rustupSha256='e7b0f47557c1afcd86939b118cbcf7fb95a5d1d917bdd355157b63ca00fc4333' ;; \
    *) echo >&2 "unsupported architecture: ${dpkgArch}"; exit 1 ;; \
    esac; \
    url="https://static.rust-lang.org/rustup/archive/1.26.0/${rustArch}/rustup-init"; \
    wget "$url"; \
    echo "${rustupSha256} *rustup-init" | sha256sum -c -; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${rustArch}; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version;

RUN rustup component add rustfmt rust-src rust-analysis clippy

RUN cargo install cargo-binstall

RUN cargo binstall cargo-watch cargo-edit evcxr_jupyter evcxr evcxr_repl just cargo-tarpaulin -y

RUN evcxr_jupyter --install

RUN apt update && apt install -y  bash \
    libtinfo5 \
    && rm -rf /var/lib/apt/lists

RUN chown -R 1000:1000 $RUSTUP_HOME $CARGO_HOME

RUN groupadd -g 1000 canavar && \
    useradd -u 1000 -g canavar -s /bin/zsh -m canavar && \
    usermod -aG sudo canavar

RUN curl -L https://github.com/bojand/ghz/releases/download/v0.117.0/ghz-linux-x86_64.tar.gz -o /ghz.tar.gz \
    && tar -xzvf /ghz.tar.gz -C /usr/local/bin \
    && rm /ghz.tar.gz

COPY .devcontainer/id_ed25519 /home/canavar/.ssh/id_ed25519

RUN chmod 600 /home/canavar/.ssh/id_ed25519 \
    && chown canavar:canavar /home/canavar/.ssh/id_ed25519

RUN apt update && apt install -y neovim


USER canavar

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/


CMD ["/bin/zsh"]