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
    --exclude libcuda.so \
    --exclude libcuda.so.1 \
    --exclude libnvidia-ml.so \
    --exclude libnvidia-ml.so.1 \
    --exclude "libcublas.so.*" \
    --exclude "libcublasLt.so.*" \
    --exclude "libcudnn.so.*" \
    --exclude "libcudnn_*.so.*" \
    --exclude "libcusparse.so.*" \
    --exclude "libcusolver.so.*" \
    --exclude "libnvJitLink.so.*" \
    --exclude "libcudart.so.*" \
    -w "${dest_dir}" \
    "${wheel}"

repaired_wheel="$(ls -t "${dest_dir}"/*.whl | head -1)"
if [ ! -f "${repaired_wheel}" ]; then
    echo "auditwheel repair did not produce a wheel in ${dest_dir}" >&2
    exit 1
fi

# PyTorch-style RPATH: resolve nvidia-*-cu12 pip libs from site-packages.
nvidia_rpath=(
    '$ORIGIN/../../nvidia/cublas/lib'
    '$ORIGIN/../../nvidia/cudnn/lib'
    '$ORIGIN/../../nvidia/cusparse/lib'
    '$ORIGIN/../../nvidia/cusolver/lib'
    '$ORIGIN/../../nvidia/nvjitlink/lib'
    '$ORIGIN/../../nvidia/cuda_runtime/lib'
)
nvidia_rpath_joined="$(IFS=:; echo "${nvidia_rpath[*]}")"

patch_so_rpath() {
    local sofile="$1"
    local existing=""
    if existing="$(patchelf --print-rpath "${sofile}" 2>/dev/null)"; then
        :
    fi
    local new_rpath="${existing:+${existing}:}${nvidia_rpath_joined}"
    patchelf --set-rpath "${new_rpath}" --force-rpath "${sofile}"
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
unzip -q "${repaired_wheel}" -d "${tmpdir}"

if [ -f "${tmpdir}/torch_nntile/_C.so" ]; then
    patch_so_rpath "${tmpdir}/torch_nntile/_C.so"
fi

shopt -s nullglob
for sofile in \
    "${tmpdir}"/torch_nntile.libs/libnntile*.so \
    "${tmpdir}"/torch_nntile.libs/libstarpu*.so; do
    patch_so_rpath "${sofile}"
done
shopt -u nullglob

wheel_name="$(basename "${repaired_wheel}")"
rm -f "${repaired_wheel}"
(
    cd "${tmpdir}"
    zip -r9 "${dest_dir}/${wheel_name}" .
)

wheel_size_mb="$(du -m "${dest_dir}/${wheel_name}" | awk '{print $1}')"
echo "Repaired wheel size: ${wheel_size_mb} MB (${dest_dir}/${wheel_name})"
