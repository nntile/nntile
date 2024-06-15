#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo "usage: $0 <python-version>"
    exit 1
fi

docker build . \
    -f ci/manylinux/Dockerfile \
    -t ghcr.io/nntile/nntile/manylinux:2_28-x86_64 \
    -o type=local,dest=. \
    --progress=plain \
    --build-arg PYTHON_VERSION=$1
