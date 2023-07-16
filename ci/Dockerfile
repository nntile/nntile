FROM ubuntu:22.04 AS devbase

ADD ci/add-repo-kitware.sh .

RUN set -xe && \
    DEBIAN_FRONTEND=noninteractive && \
    apt update && \
    ./add-repo-kitware.sh && \
    apt update && \
    apt-get install -y kitware-archive-keyring && \
    find /var/lib/apt/lists -type f -print -delete && \
    rm -rf ./add-repo-kitware.sh

RUN apt update && \
    apt install -y --no-install-recommends \
        autoconf automake binutils build-essential clang cmake cmake-curses-gui \
        fxt-tools gdb lcov libfxt-dev libhwloc-dev libopenblas-dev \
        libopenmpi-dev libopenmpi3 libtool-bin ninja-build openmpi-bin \
        pkg-config python3 python3-dev python-is-python3 python3-pip vim && \
    find /var/lib/apt/lists -type f -print -delete

FROM devbase AS sandbox

ARG MAKE_JOBS=1

ENV STARPU_COMMIT=73a184851025319ca5ae7ddaadab92d5d302f10b

RUN set -xe && \
    mkdir -p /usr/src && \
    curl -SL https://gitlab.inria.fr/starpu/starpu/-/archive/$STARPU_COMMIT/starpu-$STARPU_COMMIT.tar.gz | \
    tar -xzC /usr/src && \
    ln -s /usr/src/starpu-$STARPU_COMMIT /usr/src/starpu && \
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
        --with-fxt && \
    make -j $MAKE_JOBS install && \
    rm -rf /usr/src/starpu /usr/src/starpu-$STARPU_COMMIT && \
    echo '/usr/local/lib' > /etc/ld.so.conf.d/nntile.conf && \
    ldconfig

WORKDIR /workspace/nntile

ADD wrappers/python/pyproject.toml wrappers/python/

RUN --mount=type=cache,target=/root/.cache/pip \
    set -xe && \
    pip install tomli && \
    GIST_PEDS=https://gist.githubusercontent.com/daskol/5513ff9c5b8a2d6b2a0e78f522dd2800 && \
    curl -SL $GIST_PEDS/raw/4e7b80e5f9d49c2e39cf8aa4e6b6b8b951724730/peds.py | \
    python - -i -e dev -e test --extra-index-url https://download.pytorch.org/whl/cpu wrappers/python
