# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file notebooks/Dockerfile
# Instructions for Docker to build a Jupyter server with NNTile support
#
# @version 1.0.0

ARG BASE_IMAGE=nvidia/cuda:12.4.0-devel-ubuntu22.04

FROM $BASE_IMAGE AS devbase

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

ARG MAKE_JOBS=1

ENV STARPU_SILENT=1

ENV OMP_NUM_THREADS=1

ARG STARPU_VERSION=starpu-1.3.11

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

ADD pyproject.toml .

RUN --mount=type=cache,target=/root/.cache/pip \
    set -xe && \
    pip install tomli jupyter && \
    GIST_PEDS=https://gist.githubusercontent.com/daskol/5513ff9c5b8a2d6b2a0e78f522dd2800 && \
    curl -SL $GIST_PEDS/raw/4e7b80e5f9d49c2e39cf8aa4e6b6b8b951724730/peds.py | \
    python - -i -e test .

RUN --mount=type=cache,target=/var/cache/apt \
    apt update && \
    apt install -y --no-install-recommends time && \
    find /var/lib/apt/lists -type f -print -delete

ADD . /workspace/nntile

ARG CUDA_ARCHS=70;75;80;86;89;90

RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} -GNinja

RUN cmake --build build -j $MAKE_JOBS -t nntile_core

ENV PYTHONPATH=/workspace/nntile/build/wrappers/python

ENV STARPU_SILENT=1

ENV STARPU_FXT_TRACE=0

EXPOSE 8888

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", \
    "--allow-root", "--notebook-dir=notebooks"]
