#!/usr/bin/env bash
# Install a headless CUDA toolkit in manylinux_2_28 (RHEL8 / AlmaLinux 8).
# No GPU or NVIDIA driver is required: nvcc, headers, libcudart, and libcuda
# stubs come from dnf packages. Runtime CUDA workers still need a driver.
set -euo pipefail

cuda_version="${CUDA_VERSION:-12.8}"
cudaver="$(echo "${cuda_version}" | cut -d '.' -f-2)"
pkg_ver="${cudaver//./-}"

if ! command -v dnf >/dev/null 2>&1; then
    echo "CUDA toolkit install requires dnf (manylinux_2_28)" >&2
    exit 1
fi

repo_url="https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo"
repo_file="/etc/yum.repos.d/cuda-rhel8.repo"
cuda_packages=(
    "cuda-minimal-build-${pkg_ver}"
    "cuda-driver-devel-${pkg_ver}"
    "cuda-libraries-devel-${pkg_ver}"
    "cuda-nvrtc-devel-${pkg_ver}"
)

configure_cuda_repo() {
    dnf install -y dnf-plugins-core
    rm -f "${repo_file}"
    dnf config-manager --add-repo "${repo_url}"
    dnf clean all
}

install_cuda_packages() {
    dnf install -y "${cuda_packages[@]}"
    dnf clean all
}

max_attempts=4
attempt=1
while [ "${attempt}" -le "${max_attempts}" ]; do
    if configure_cuda_repo && install_cuda_packages; then
        break
    fi
    if [ "${attempt}" -eq "${max_attempts}" ]; then
        echo "CUDA toolkit install failed after ${max_attempts} attempts" >&2
        exit 1
    fi
    echo "CUDA dnf install failed (attempt ${attempt}); retrying..." >&2
    sleep $((attempt * 10))
    attempt=$((attempt + 1))
done

export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "nvcc not found under ${CUDA_HOME}/bin after CUDA toolkit install" >&2
    exit 1
fi
if [ ! -f "${CUDA_HOME}/include/cuda.h" ]; then
    echo "cuda.h not found under ${CUDA_HOME}/include" >&2
    exit 1
fi
