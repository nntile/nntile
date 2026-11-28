# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems with GPU acceleration.
#
# @file Dockerfile
# NNTile 2.0 sandbox: CUDA toolkit + cuDNN + uv-managed Python environment
# with PyTorch. Published as ghcr.io/nntile/nntile-sandbox and reused by CI
# so every job does not reinstall the toolchain.
#
# @version 2.0.0

# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=ubuntu24.04
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-${UBUNTU_VERSION}

FROM ${BASE_IMAGE} AS sandbox

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG USER_UID=1000
ARG USER_GID=1000

LABEL org.opencontainers.image.source="https://github.com/nntile/nntile" \
      org.opencontainers.image.title="NNTile 2.0 sandbox" \
      org.opencontainers.image.description="CUDA-only NNHaul build/test environment" \
      org.opencontainers.image.licenses="MIT"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ccache cmake curl git gdb lcov \
        ninja-build pkg-config python3.12 python3.12-dev python3.12-venv \
        unzip wget vim && \
    rm -rf /var/lib/apt/lists/*

# uv installs and owns the Python environment; no conda layer.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    CCACHE_DIR=/ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_ARCHITECTURES=70;75;80;86;89;90 \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1

RUN uv venv /opt/venv --python ${PYTHON_VERSION} && \
    uv pip install --python /opt/venv/bin/python \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install --python /opt/venv/bin/python \
        "transformers==4.52.*" "numpy<3" pytest setuptools wheel ninja

RUN groupadd -g ${USER_GID} nntile 2>/dev/null || true && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash nntile 2>/dev/null || true && \
    mkdir -p /workspace /ccache && \
    chown -R ${USER_UID}:${USER_GID} /workspace /ccache /opt/venv

WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["bash"]
