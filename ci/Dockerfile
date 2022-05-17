FROM ubuntu:22.04

RUN apt update && \
    apt install -y --no-install-recommends \
        autoconf \
        automake \
        binutils \
        clang \
        clang-format \
        g++ \
        git \
        cmake \
        cmake-format \
        curl \
        liblapack-dev \
        libopenblas-dev \
        libtool \
        make \
        ninja-build \
        pkg-config \
        software-properties-common && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && \
    apt install -y --no-install-recommends \
        starpu-tools\
        libstarpu-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
