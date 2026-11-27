#!/usr/bin/env bash
# Fail if libnntile's cublas/cudnn/cudart NEEDED entries resolve under the
# system CUDA toolkit instead of pip site-packages/nvidia (when linked).
set -euo pipefail

lib="${1:?path to libnntile.so required}"
if [ ! -f "${lib}" ]; then
    echo "assert_pip_cuda_libs: missing ${lib}" >&2
    exit 1
fi

if ! command -v readelf >/dev/null 2>&1; then
    echo "assert_pip_cuda_libs: readelf not found; skipping" >&2
    exit 0
fi

needed="$(readelf -d "${lib}" | awk '/NEEDED/ {print $5}' | tr -d '[]')"
echo "NEEDED (nvidia-related) for ${lib}:"
echo "${needed}" | grep -E 'cublas|cudnn|cudart' || true

if command -v ldd >/dev/null 2>&1; then
    resolved="$(ldd "${lib}" 2>/dev/null || true)"
    bad="$(echo "${resolved}" | grep -E 'libcublas|libcudnn|libcudart' \
        | grep '/usr/local/cuda' || true)"
    if [ -n "${bad}" ]; then
        echo "assert_pip_cuda_libs: resolved toolkit copies (want pip nvidia):" >&2
        echo "${bad}" >&2
        exit 1
    fi
    good="$(echo "${resolved}" | grep -E 'libcublas|libcudnn|libcudart' \
        | grep 'site-packages/nvidia' || true)"
    if [ -n "${good}" ]; then
        echo "Resolved from pip nvidia:"
        echo "${good}"
    else
        echo "assert_pip_cuda_libs: warning: no site-packages/nvidia resolution" \
            "(ldd may be incomplete without LD_LIBRARY_PATH); NEEDED check only" >&2
    fi
fi
