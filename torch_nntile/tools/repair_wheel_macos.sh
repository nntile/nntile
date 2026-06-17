#!/usr/bin/env bash
set -euo pipefail

wheel="${1:?wheel path is required}"
dest_dir="${2:?destination directory is required}"
require_archs="${3:-arm64}"
package_dir="$(cd "$(dirname "$0")/.." && pwd)"
repo_root="$(cd "${package_dir}/.." && pwd)"
build_dir="${NNTILE_BUILD_DIR:-${repo_root}/build/torch_nntile_wheel}"
starpu_prefix="${STARPU_PREFIX:-/opt/starpu}"

export DYLD_LIBRARY_PATH="${build_dir}/nntile:${starpu_prefix}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

mkdir -p "${dest_dir}"

delocate-wheel \
    --exclude /torch/ \
    --exclude libtorch \
    --exclude libc10 \
    --require-archs "${require_archs}" \
    -w "${dest_dir}" \
    -v \
    "${wheel}"
