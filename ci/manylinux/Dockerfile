FROM quay.io/pypa/manylinux_2_28_x86_64 AS builder

RUN --mount=type=cache,target=/var/cache/dnf \
    echo 'keepcache=True' >> /etc/dnf/dnf.conf && \
    dnf install -y \
        autoconf binutils-devel cmake hwloc-devel make ninja-build \
        openblas openblas-devel

WORKDIR /usr/src/nntile

ADD . .

ARG PYTHON_VERSION=3.12

RUN cmake -S . -B build/cp${PYTHON_VERSION/\./} \
        -DNNTILE_BLAS_VENDOR=OpenBLAS \
        -DPYTHON_EXECUTABLE=python$PYTHON_VERSION && \
    cmake --build build/cp${PYTHON_VERSION/\./} --target nntile-symlink && \
    cd wrappers/python && \
    python$PYTHON_VERSION setup.py \
        build_ext -t ../../build/cp${PYTHON_VERSION/\./} \
        bdist_wheel && \
    auditwheel repair dist/nntile-*.whl

FROM scratch AS artifact

COPY --from=builder /usr/src/nntile/wrappers/python/wheelhouse/* /
