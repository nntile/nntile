# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .cursor/Dockerfile
# CPU-only variant for Cursor development environment (no CUDA/GPU support).
# Instructions for Docker to build an image for testing with NNTile and Torch
# using conda/miniconda for all dependencies.
#
# @version 1.1.0

# Set base OS (CPU-only, no CUDA base image)
ARG BASE_IMAGE=ubuntu:22.04

# Read the base image. Target "sandbox" contains all the NNTile prerequisites
# and is meant for the NNTile development
FROM ${BASE_IMAGE} AS sandbox

# Image labels
LABEL org.opencontainers.image.source="https://github.com/nntile/nntile"
LABEL Maintainer="Aleksandr Mikhalev <al.mikhalev@skoltech.ru>"

# Parallel build
ARG MAKE_JOBS=1

# FXT tool version
ARG FXT_VERSION=0.3.15

# StarPU version
ARG STARPU_VERSION=1.4.12

# Desired Python version
ARG PYTHON_VERSION=3.12

# Desired Pytorch version (CPU-only)
ARG PYTORCH_VERSION=2.9.1

# No interactions with a user during docker build
# ARG is not exposed into a container environment
ARG DEBIAN_FRONTEND=noninteractive

# Override the default shell to ensure bash is used
SHELL ["/bin/bash", "--login", "-c"]

# Install only minimal system packages needed for conda installation
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && \
    apt install -y --no-install-recommends \
        ca-certificates \
        curl \
        wget

# Set Conda-related environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

# Install Miniforge
RUN --mount=type=cache,target=/tmp \
    set -xe && \
    export URL="https://github.com/conda-forge/miniforge/releases/latest" && \
    export URL="${URL}/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    wget ${URL} -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    conda init && \
    source "${CONDA_DIR}/etc/profile.d/conda.sh" && \
    source "${CONDA_DIR}/etc/profile.d/mamba.sh" && \
    conda config --set auto_activate_base false

# Create an environment "nntile" with custom Python and Pytorch versions (CPU-only)
# Also install compilers and libraries, required for the NNTile
RUN --mount=type=cache,target=/opt/conda/pkgs \
    set -xe && \
    conda create -n nntile \
        python=${PYTHON_VERSION} \
        pytorch-cpu=${PYTORCH_VERSION} \
        "transformers==4.52.*" \
        nomkl \
        openblas \
        cmake \
        ninja \
        pkg-config \
        autoconf \
        automake \
        libtool \
        make \
        gcc_linux-64 \
        gxx_linux-64 \
        gfortran_linux-64 \
        libhwloc \
        libxml2-devel \
        git \
        gawk \
        curl \
        vim \
        gdb \
        lcov \
        binutils \
        time \
        torchvision \
        pre-commit \
        isort \
        ruff \
        mypy \
        jupyter \
        tensorflow-cpu \
        tensorboard \
        fastapi \
        numpy \
        pydantic \
        scipy \
        uvicorn \
        tomli \
        sentencepiece \
        datasets \
        pytest \
        pytest-benchmark \
        tokenizers

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "nntile", "/bin/bash", "-c"]

# Make containers use "nntile" environment by default
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "nntile"]

# Launch bash in the "nntile" environment by default
CMD ["/bin/bash"]

# Build and install FXT from source into conda environment
# fxt is required by StarPU but not available in conda-forge
RUN --mount=type=cache,target=/tmp \
    set -xe && \
    cd /tmp && \
    export NAME=fxt-${FXT_VERSION} && \
    export URL=https://download-mirror.savannah.gnu.org/releases/fkt && \
    export URL=${URL}/${NAME}.tar.gz && \
    wget -q ${URL} -O ${NAME}.tar.gz && \
    mkdir -p ${NAME} && \
    tar -xzf ${NAME}.tar.gz -C ${NAME} --strip-components=1 && \
    cd ${NAME} && \
    ./configure \
        --prefix=${CONDA_PREFIX} && \
    make -j ${MAKE_JOBS} && \
    make install && \
    ldconfig

# Build and install StarPU from source into conda environment (CPU-only)
# StarPU is not available in conda-forge
RUN --mount=type=cache,target=/tmp \
    set -xe && \
    cd /tmp && \
    export NAME=starpu-${STARPU_VERSION} && \
    export URL=https://github.com/starpu-runtime/starpu/archive/refs/tags/ && \
    export URL=${URL}/${NAME}.tar.gz && \
    wget -q ${URL} -O ${NAME}.tar.gz && \
    mkdir -p ${NAME} && \
    tar -xzf ${NAME}.tar.gz -C ${NAME} --strip-components=1 && \
    cd ${NAME} && \
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
        --prefix=${CONDA_PREFIX} && \
    make -j ${MAKE_JOBS} && \
    make install && \
    ldconfig

# Set StarPU-specific environment variables
ENV STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1

# Disable OpenMP parallelism of OpenBLAS and MKL
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Set default working directory for the NNTile development
WORKDIR /workspace/nntile

# Define Python path for the development
ENV PYTHONPATH=/workspace/nntile/build/wrappers/python

# Open Jupyter Lab and Tensorboard ports
EXPOSE 8888 6006
