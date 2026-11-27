#!/usr/bin/env bash
# Install a thin, headless CUDA "compiler kit" in manylinux_2_28 (RHEL8).
#
# Provides only what pip nvidia-*-cu12 cannot:
#   - nvcc (+ CRT enough to compile .cu)
#   - lib64/stubs/libcuda.so (StarPU CUDA link)
#
# Math/runtime libraries (cudart, cublas, cudnn, …) must come from pip
# packages installed alongside torch (see setup_torch_cuda_env.sh). Do not
# install cuda-libraries-devel / cuda-nvrtc-devel here — they duplicate
# multi-GB .so trees already present under site-packages/nvidia/.
#
# No GPU or NVIDIA driver is required at build time. Runtime CUDA workers
# still need a host driver.
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
# cuda-minimal-build pulls nvcc + cuda-cudart-devel (headers / small cudart).
# cuda-driver-devel provides lib64/stubs/libcuda.so.
cuda_packages=(
    "cuda-minimal-build-${pkg_ver}"
    "cuda-driver-devel-${pkg_ver}"
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
# Stubs only in LD path from the toolkit; pip nvidia libs are prepended later.
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [ ! -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "nvcc not found under ${CUDA_HOME}/bin after CUDA toolkit install" >&2
    exit 1
fi
if [ ! -f "${CUDA_HOME}/include/cuda.h" ]; then
    echo "cuda.h not found under ${CUDA_HOME}/include" >&2
    exit 1
fi
if [ ! -e "${CUDA_HOME}/lib64/stubs/libcuda.so" ] \
    && [ ! -e "${CUDA_HOME}/lib64/stubs/libcuda.so.1" ]; then
    echo "libcuda stub not found under ${CUDA_HOME}/lib64/stubs" >&2
    exit 1
fi

if [ "${TORCH_NNTILE_DISK_LOG:-0}" = "1" ]; then
    echo "[disk] after thin CUDA toolkit:" >&2
    df -h / /tmp 2>/dev/null || df -h /
    if [ -d "${CUDA_HOME}/lib64" ]; then
        echo "[disk] toolkit lib64 (should lack libcublas/libcudnn):" >&2
        ls "${CUDA_HOME}/lib64"/libcublas* \
            "${CUDA_HOME}/lib64"/libcudnn* 2>/dev/null || true
        du -sh "${CUDA_HOME}" "${CUDA_HOME}/lib64" 2>/dev/null || true
    fi
fi
