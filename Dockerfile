# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file Dockerfile
# Instructions for Docker to build an image for testing with NNTile and Torch
#
# @version 1.0.0

ARG BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04

FROM $BASE_IMAGE AS devbase

LABEL org.opencontainers.image.source="https://github.com/nntile/nntile"

ENV DEBIAN_FRONTEND=noninteractive

ADD ci/add-repo-kitware.sh .

RUN --mount=type=cache,target=/var/cache/apt \
    rm -rfv /etc/apt/apt.conf.d/docker* && \
    ./add-repo-kitware.sh && \
    find /var/lib/apt/lists -type f -print -delete && \
    rm -rf ./add-repo-kitware.sh

RUN --mount=type=cache,target=/var/cache/apt \
    rm -rfv /etc/apt/apt.conf.d/docker* && \
    apt update && \
    apt install -y --no-install-recommends \
        autoconf automake binutils build-essential clang cmake \
        cmake-curses-gui fxt-tools gdb git lcov libfxt-dev libhwloc-dev \
        libopenblas-dev libtool-bin ninja-build pkg-config python3 \
        python3-dev python-is-python3 python3-pip vim && \
    find /var/lib/apt/lists -type f -print -delete

FROM devbase AS sandbox

ARG MAKE_JOBS=1

ENV STARPU_SILENT=1

ENV OMP_NUM_THREADS=1

ARG STARPU_VERSION=starpu-1.4.7

RUN set -xe && \
    mkdir -p /usr/src && \
    STARPU_LABEL=$STARPU_VERSION && \
    (curl -SL https://gitlab.inria.fr/starpu/starpu/-/archive/$STARPU_LABEL/starpu-$STARPU_LABEL.tar.gz | \
    tar -xzC /usr/src) && \
    ln -s /usr/src/starpu-$STARPU_LABEL /usr/src/starpu && \
    cd /usr/src/starpu && \
    ./autogen.sh && \
    ./configure \
        --disable-build-doc \
        --disable-build-examples \
        --disable-build-tests \
        --disable-fortran \
        --disable-opencl \
        --disable-socl \
        --disable-starpufft \
        --disable-starpupy \
        --enable-blas-lib=none \
        --enable-maxcudadev=8 \
        --enable-maxbuffers=16 \
        --with-fxt && \
    make -j $MAKE_JOBS install && \
    rm -rf /usr/src/starpu /usr/src/starpu-$STARPU_LABEL && \
    echo '/usr/local/lib' > /etc/ld.so.conf.d/nntile.conf && \
    ldconfig

WORKDIR /workspace/nntile

ADD wrappers/python/pyproject.toml wrappers/python/

RUN --mount=type=cache,target=/root/.cache/pip \
    set -xe && \
    pip install tomli && \
    GIST_PEDS=https://gist.githubusercontent.com/daskol/5513ff9c5b8a2d6b2a0e78f522dd2800 && \
    curl -SL $GIST_PEDS/raw/4e7b80e5f9d49c2e39cf8aa4e6b6b8b951724730/peds.py | \
    python - -i -e test wrappers/python

RUN --mount=type=cache,target=/var/cache/apt \
     apt update && \
    apt install -y --no-install-recommends time && \
    find /var/lib/apt/lists -type f -print -delete

FROM sandbox AS nntile

ADD . /workspace/nntile

ARG CUDA_ARCHS=70;75;80;86;89;90

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} -GNinja

RUN cmake --build build -j $MAKE_JOBS

ENV PYTHONPATH=/workspace/nntile/build/wrappers/python

ENV STARPU_SILENT=1

ENV STARPU_FXT_TRACE=0
