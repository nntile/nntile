# CPU-only development image for NNTile (no CUDA/GPU support)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    rm -rfv /etc/apt/apt.conf.d/docker* && \
    apt update && \
    apt install -y --no-install-recommends \
        autoconf automake binutils build-essential ca-certificates \
        clang cmake cmake-curses-gui curl fxt-tools git g++ lcov \
        libfxt-dev libhwloc-dev libopenblas-dev libopenmpi-dev \
        libopenmpi3 libtool-bin lsb-release ninja-build openmpi-bin \
        pkg-config python3 python3-dev python-is-python3 python3-pip \
        python3-venv && \
    find /var/lib/apt/lists -type f -print -delete

WORKDIR /workspace

RUN chown -R ubuntu:ubuntu /workspace

# Compile and install StarPU (CPU-only, no CUDA)
ARG MAKE_JOBS=4

ARG STARPU_VERSION=1.4.8

RUN set -xe && \
    export STARPU_LABEL=starpu-${STARPU_VERSION} && \
    mkdir -p /usr/src && \
    curl -SL https://gitlab.inria.fr/starpu/starpu/-/archive/${STARPU_LABEL}/starpu-${STARPU_LABEL}.tar.gz \
        | tar -xzC /usr/src && \
    cd /usr/src/starpu-${STARPU_LABEL} && \
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
        --enable-maxcudadev=0 \
        --enable-maxbuffers=16 \
        --with-fxt \
        --prefix=/usr && \
    make -j ${MAKE_JOBS} && \
    make install && \
    rm -rf /usr/src/starpu-${STARPU_LABEL} && \
    ldconfig

ENV STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Python dependencies (CPU-only PyTorch)
RUN pip3 install --break-system-packages \
    numpy scipy fastapi uvicorn pydantic isort mypy pre-commit \
    ruff sentencepiece datasets pytest pytest-cov pytest-dirty \
    tokenizers "transformers<=4.52" "markupsafe<3"

RUN pip3 install --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Switch to ubuntu user
USER ubuntu
