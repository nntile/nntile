#!/usr/bin/env bash
set -euo pipefail

wheel="${1:?wheel path is required}"
dest_dir="${2:?destination directory is required}"
package_dir="$(cd "$(dirname "$0")/.." && pwd)"
repo_root="$(cd "${package_dir}/.." && pwd)"
build_dir="${NNTILE_BUILD_DIR:-${repo_root}/build/torch_nntile_wheel}"
starpu_prefix="${STARPU_PREFIX:-/opt/starpu}"

export LD_LIBRARY_PATH="${build_dir}/nntile:${starpu_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p "${dest_dir}"

auditwheel repair \
    --exclude libc10.so \
    --exclude libc10_cuda.so \
    --exclude libtorch.so \
    --exclude libtorch_cpu.so \
    --exclude libtorch_cuda.so \
    --exclude libtorch_global_deps.so \
    --exclude libtorch_python.so \
    -w "${dest_dir}" \
    "${wheel}"
