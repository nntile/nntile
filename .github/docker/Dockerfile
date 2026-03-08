# Replicates GitHub CI build-and-test environment (ubuntu:24.04, Python 3.12,
# StarPU, NNTile CPU-only). Use to reproduce CI failures locally.
#
# Build (from repo root):  docker build -f .github/docker/Dockerfile -t nntile-ci .
# Shell:                 docker run --rm -it nntile-ci bash
# Run llama test:        docker run --rm nntile-ci

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (matches .github/workflows/on-push.yml)
RUN rm -rfv /etc/apt/apt.conf.d/docker* 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf automake binutils build-essential ca-certificates \
        clang cmake curl fxt-tools gdb git lcov libfxt-dev libhwloc-dev \
        libopenblas-dev libopenmpi-dev libopenmpi3 libtool-bin \
        lsb-release ninja-build openmpi-bin pkg-config python3 python3-pip \
        sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 (Ubuntu 24.04 default)
RUN ln -sf /usr/bin/python3 /usr/bin/python

# StarPU 1.4.8 (matches CI)
RUN export STARPU_VERSION=starpu-1.4.8 \
    && export STARPU_LABEL=$STARPU_VERSION \
    && mkdir -p /usr/src \
    && curl -SL "https://gitlab.inria.fr/starpu/starpu/-/archive/$STARPU_LABEL/starpu-$STARPU_LABEL.tar.gz" \
        | tar -xzC /usr/src \
    && ln -s /usr/src/starpu-$STARPU_LABEL /usr/src/starpu \
    && cd /usr/src/starpu \
    && ./autogen.sh \
    && ./configure \
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
        --with-fxt \
        --prefix=/usr \
    && make -j$(nproc) install \
    && rm -rf /usr/src/starpu /usr/src/starpu-$STARPU_LABEL

WORKDIR /workspace

# Copy project (build context should be repo root)
COPY . .

# Python deps (matches CI: uv sync --extra ci --extra test)
RUN pip install uv --root-user-action=ignore \
    && uv sync --extra ci --extra test \
        --no-install-project --no-build-isolation \
        --default-index https://download.pytorch.org/whl/cpu \
        --index https://download.pytorch.org/whl/cpu \
        --index https://pypi.org/simple \
        --index-strategy unsafe-best-match

# Build NNTile
RUN . .venv/bin/activate \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
        -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)') \
    && cmake --build build -j$(nproc)

# Symlinks for extensions (matches CI)
RUN ln -rs build/wrappers/python/nntile/nntile_core.so \
        wrappers/python/nntile/nntile_core.so \
    && ln -rs build/wrappers/python/nntile/nntile_graph.so \
        wrappers/python/nntile/nntile_graph.so

# Default: run pytest (override with bash for interactive)
ENV PYTHONPATH=/workspace/wrappers/python
CMD [".venv/bin/python", "-m", "pytest", "wrappers/python/tests/graph/model/llama/test_llama_attention.py", "-v"]
