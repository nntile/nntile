#!/usr/bin/env bash
set -euo pipefail

wheel="${1:?wheel path is required}"
dest_dir="${2:?destination directory is required}"
require_archs="${3:-arm64}"
package_dir="$(cd "$(dirname "$0")/.." && pwd)"
repo_root="$(cd "${package_dir}/.." && pwd)"
build_dir="${NNTILE_BUILD_DIR:-${repo_root}/build/torch_nntile_wheel}"
starpu_prefix="${STARPU_PREFIX:-/opt/starpu}"

export MACOSX_DEPLOYMENT_TARGET=14.0

# Locate torch/lib so delocate can resolve @rpath links from _C.so.
# Those dylibs are excluded from the wheel (provided by the torch dependency).
resolve_torch_lib() {
    local py
    for py in "${DELOCATE_PYTHON:-}" python python3; do
        if [ -z "${py}" ]; then
            continue
        fi
        if ! command -v "${py}" >/dev/null 2>&1; then
            continue
        fi
        if "${py}" -c 'import torch' >/dev/null 2>&1; then
            "${py}" -c \
                'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")'
            return 0
        fi
    done
    return 1
}

torch_lib=""
if torch_lib="$(resolve_torch_lib)"; then
    :
else
    torch_lib=""
fi

dyld_parts=(
    "${build_dir}/nntile"
    "${build_dir}/torch_nntile"
    "${starpu_prefix}/lib"
)
if [ -n "${torch_lib}" ] && [ -d "${torch_lib}" ]; then
    echo "delocate: using torch lib dir ${torch_lib}"
    dyld_parts+=("${torch_lib}")
else
    echo "delocate: torch lib dir not found on PATH" >&2
fi

export DYLD_LIBRARY_PATH="$(IFS=:; echo "${dyld_parts[*]}")${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_LIBRARY_PATH}${DYLD_FALLBACK_LIBRARY_PATH:+:${DYLD_FALLBACK_LIBRARY_PATH}}"

mkdir -p "${dest_dir}"

# Exclude torch/c10 by path and exact basenames. Do NOT use a bare "libtorch"
# pattern — that also matches libtorch_nntile.dylib which must be bundled.
delocate_opts=(
    --exclude /torch/
    --exclude libtorch.dylib
    --exclude libtorch_cpu.dylib
    --exclude libtorch_python.dylib
    --exclude libtorch_global_deps.dylib
    --exclude libc10.dylib
    --require-archs "${require_archs}"
    -w "${dest_dir}"
    -v
)

# If torch is not resolvable, ignore missing @rpath torch deps only after
# StarPU/libnntile/libtorch_nntile are on DYLD_LIBRARY_PATH for bundling.
if [ -z "${torch_lib}" ] || [ ! -d "${torch_lib}" ]; then
    echo "delocate: falling back to --ignore-missing-dependencies for torch" >&2
    delocate_opts+=(--ignore-missing-dependencies)
fi

delocate-wheel "${delocate_opts[@]}" "${wheel}"
