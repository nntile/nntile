FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt \
    rm -rfv /etc/apt/apt.conf.d/docker* && \
    apt update && \
    apt install -y --no-install-recommends \
        autoconf automake binutils build-essential cmake \
        cmake-curses-gui curl git g++ lcov libhwloc-dev \
        libopenblas-dev libtool-bin ninja-build pkg-config python3 \
        python3-dev python-is-python3 python3-pip && \
    find /var/lib/apt/lists -type f -print -delete

WORKDIR /workspace

RUN chown -R ubuntu:ubuntu /workspace

# Compile and install StarPU of a given version (or a hash of commit) using
# parallel build
ARG MAKE_JOBS=1

# Use starpu-1.4.8 tag
ARG STARPU_VERSION=1.4.8

ENV STARPU_LABEL=starpu-${STARPU_VERSION}

ENV STARPU_URL=https://gitlab.inria.fr/starpu/starpu/-/archive

ENV STARPU_URL=${STARPU_URL}/${STARPU_LABEL}/starpu-${STARPU_LABEL}.tar.gz

RUN set -xe && \
    curl -SL ${STARPU_URL} | tar -xzC /workspace && \
    cd /workspace/starpu-${STARPU_LABEL} && \
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
        --enable-maxbuffers=4 && \
    make -j $MAKE_JOBS && \
    make install && \
    rm -rf /workspace/starpu-${STARPU_LABEL} && \
    echo '/usr/local/lib' > /etc/ld.so.conf.d/starpu.conf && \
    ldconfig

ENV STARPU_SILENT=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN pip3 install --break-system-packages \
    numpy scipy fastapi uvicorn pydantic isort mypy \
    ruff sentencepiece datasets pytest pytest-cov pytest-dirty \
    tokenizers "transformers<=4.52" markupsafe

RUN pip3 install --break-system-packages \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Switch to ubuntu user
USER ubuntu
