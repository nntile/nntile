# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# NNTile 2.0: CUDA-only, NNHaul backend.
#
# @file Dockerfile
# Instructions for Docker to build an image for testing with NNTile and Torch
# using conda/miniconda for all dependencies
#
# @version 2.0.0

# Set CUDA version
ARG CUDA_VERSION=12.8

# Set base OS
ARG BASE_OS=ubuntu22.04

# Use CUDA base image with CUDA driver
# Everything else is installed via Conda into "nntile" environment
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-base-${BASE_OS}

# Parallel build (inherited in sandbox and nntile stages)
ARG MAKE_JOBS=4

# Target CUDA architectures (inherited in sandbox and nntile stages)
ARG CUDA_ARCHS=70;75;80;86;89;90;100;120

# Read the base image. Target "sandbox" contains all the NNTile prerequisites
# and is meant for the NNTile development
FROM ${BASE_IMAGE} AS sandbox

# Image labels
LABEL org.opencontainers.image.source="https://github.com/nntile/nntile"
LABEL Maintainer="Aleksandr Mikhalev <al.mikhalev@skoltech.ru>"

# Inherit global ARGs
ARG MAKE_JOBS
ARG CUDA_ARCHS

# Desired Python version
ARG PYTHON_VERSION=3.12

# Desired Pytorch version
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
        unzip \
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

# Create an environment "nntile" with custom Python and Pytorch versions
# Also install compilers and libraries, required for the NNTile
RUN --mount=type=cache,target=/opt/conda/pkgs \
    set -xe && \
    CONDA_OVERRIDE_CUDA=${CUDA_VERSION} conda create -n nntile \
        python=${PYTHON_VERSION} \
        pytorch-gpu=${PYTORCH_VERSION} \
        cuda \
        cuda-toolkit \
        "transformers==4.52.*" \
        nomkl \
        cmake \
        ninja \
        make \
        gcc_linux-64 \
        gxx_linux-64 \
        gfortran_linux-64 \
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
        numpy \
        pydantic \
        scipy \
        tomli \
        sentencepiece \
        datasets \
        pytest \
        pytest-benchmark \
        tokenizers \
        patchelf \
        auditwheel

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "nntile", "/bin/bash", "-c"]

# Make containers use "nntile" environment by default
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "nntile"]

# Launch bash in the "nntile" environment by default
CMD ["/bin/bash"]

# Keep the host math libraries single-threaded in the container
ENV OMP_NUM_THREADS=1

# Set default working directory for the NNTile development
WORKDIR /workspace/nntile


# Target "nntile" contains the compiled libraries
FROM sandbox AS nntile

# Inherit global ARGs
ARG MAKE_JOBS
ARG CUDA_ARCHS

# Copy all sources
ADD . /workspace/nntile

# Configure NNTile 2.0 (CUDA-only, NNHaul-only).
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DBUILD_LIBTORCH_NNTILE=OFF \
    -DBUILD_TORCH_NNTILE=OFF \
    -GNinja

# Finally, build the NNTile inplace without installation
RUN cmake --build build -j ${MAKE_JOBS}
