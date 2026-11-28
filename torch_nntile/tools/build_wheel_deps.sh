#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems with GPU acceleration.
#
# @file torch_nntile/tools/build_wheel_deps.sh
# NNTile 2.0 file.
#
# @version 2.0.0

# Build libnntile + libtorch_nntile for torch_nntile wheel packaging.
#
# Default (cibuildwheel before-all): compile shared libs only; cibuildwheel
# then builds the Python extension and runs tools/smoke_test_wheel.py.
#
# Optional: TORCH_NNTILE_CMAKE_WHEEL=1 also builds the wheel via CMake target
# torch_nntile_wheel (BUILD_TORCH_NNTILE) into TORCH_NNTILE_WHEELHOUSE.
set -euo pipefail

package_or_repo="${1:-$(pwd)}"
repo_root="$(cd "${package_or_repo}" && pwd)"
if [ ! -f "${repo_root}/CMakeLists.txt" ] || [ ! -d "${repo_root}/nntile" ]; then
    repo_root="$(cd "${repo_root}/.." && pwd)"
fi
if [ ! -f "${repo_root}/CMakeLists.txt" ] || [ ! -d "${repo_root}/nntile" ]; then
    echo "Could not locate NNTile repository root from ${package_or_repo}" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${NNTILE_BUILD_DIR:-${repo_root}/build/torch_nntile_wheel}"
wheelhouse_out="${TORCH_NNTILE_WHEELHOUSE:-${repo_root}/wheelhouse}"
jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
os_name="$(uname -s)"
cmake_wheel="${TORCH_NNTILE_CMAKE_WHEEL:-0}"

install_linux_packages() {
    if command -v dnf >/dev/null 2>&1; then
        dnf install -y \
            bzip2 cmake curl gcc gcc-c++ git make ninja-build \
            pkgconf-pkg-config unzip wget
    elif command -v yum >/dev/null 2>&1; then
        yum install -y \
            bzip2 cmake curl gcc gcc-c++ git make ninja-build \
            pkgconf-pkg-config unzip wget
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y --no-install-recommends \
            build-essential ca-certificates cmake curl git \
            ninja-build pkg-config unzip wget python3 python3-dev python3-pip
    else
        echo "Unsupported Linux image: no yum or apt-get found" >&2
        exit 1
    fi
}

install_python_wheel_tools() {
    local python="$1"
    "${python}" -m pip install --upgrade pip
    "${python}" -m pip install --no-cache-dir "setuptools>=61" wheel ninja numpy
    "${python}" -m pip install --no-cache-dir auditwheel patchelf
}

configure_nntile_cmake() {
    export LD_LIBRARY_PATH="${build_dir}/nntile:${build_dir}/torch_nntile${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    local python="${TORCH_NNTILE_PYTHON:-$("${script_dir}/wheel_python.sh")}"
    if [ -z "${TORCH_PREFIX:-}" ]; then
        export TORCH_PREFIX="$("${python}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    fi
    export CMAKE_PREFIX_PATH="${TORCH_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

    cmake_args=(
        -S "${repo_root}"
        -B "${build_dir}"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_TESTING=OFF
        -DBUILD_LIBTORCH_NNTILE=ON
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
        -DPython3_EXECUTABLE="${python}"
        -GNinja
    )

    if [ "${cmake_wheel}" = "1" ]; then
        mkdir -p "${wheelhouse_out}"
        cmake_args+=(
            -DBUILD_TORCH_NNTILE=ON
            -DTORCH_NNTILE_WHEEL_REPAIR=ON
            -DTORCH_NNTILE_WHEELHOUSE="${build_dir}/wheelhouse"
        )
        if [ -n "${TORCH_NNTILE_WHEEL_VERSION:-}" ]; then
            cmake_args+=(-DTORCH_NNTILE_WHEEL_VERSION="${TORCH_NNTILE_WHEEL_VERSION}")
        fi
    else
        cmake_args+=(-DBUILD_TORCH_NNTILE=OFF)
    fi

    cmake_args+=(
        -DCUDAToolkit_ROOT="${CUDA_HOME}"
    )
    if [ -n "${CMAKE_CUDA_COMPILER:-}" ]; then
        cmake_args+=(-DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER}")
    elif [ -n "${CUDA_HOME:-}" ]; then
        cmake_args+=(-DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc")
    fi
    if [ "${NNTILE_CUDA_FROM_PIP:-0}" = "1" ] \
        || [ -n "${NVIDIA_CUBLAS_LIBRARY_PATH:-}" ]; then
        cmake_args+=(-DNNTILE_CUDA_FROM_PIP=ON)
        if [ -n "${NVIDIA_CUBLAS_LIBRARY_PATH:-}" ]; then
            cmake_args+=(
                -DNVIDIA_CUBLAS_INCLUDE_PATH="${NVIDIA_CUBLAS_INCLUDE_PATH}"
                -DNVIDIA_CUBLAS_LIBRARY_PATH="${NVIDIA_CUBLAS_LIBRARY_PATH}"
            )
        fi
        if [ -n "${NVIDIA_CUDA_RUNTIME_LIBRARY_PATH:-}" ]; then
            cmake_args+=(
                -DNVIDIA_CUDA_RUNTIME_INCLUDE_PATH="${NVIDIA_CUDA_RUNTIME_INCLUDE_PATH}"
                -DNVIDIA_CUDA_RUNTIME_LIBRARY_PATH="${NVIDIA_CUDA_RUNTIME_LIBRARY_PATH}"
            )
        fi
        # Thin toolkit has no libnvrtc; Torch links CUDA_nvrtc_LIBRARY.
        if [ -n "${CUDA_nvrtc_LIBRARY:-}" ]; then
            cmake_args+=(-DCUDA_nvrtc_LIBRARY="${CUDA_nvrtc_LIBRARY}")
        elif [ -n "${NVIDIA_CUDA_NVRTC_LIBRARY_PATH:-}" ]; then
            nvrtc_so="$(
                ls -1 "${NVIDIA_CUDA_NVRTC_LIBRARY_PATH}"/libnvrtc.so* \
                    2>/dev/null | head -1 || true
            )"
            if [ -n "${nvrtc_so}" ]; then
                cmake_args+=(-DCUDA_nvrtc_LIBRARY="${nvrtc_so}")
            fi
        fi
        if [ -n "${CMAKE_LIBRARY_PATH:-}" ]; then
            cmake_args+=(-DCMAKE_LIBRARY_PATH="${CMAKE_LIBRARY_PATH}")
        fi
        # Torch cuda.cmake looks for cublas_v2.h under CUDA_HOME only.
        if [ -n "${NVIDIA_CUBLAS_INCLUDE_PATH:-}" ] \
            && [ -n "${CUDA_HOME:-}" ] \
            && [ -f "${NVIDIA_CUBLAS_INCLUDE_PATH}/cublas_v2.h" ] \
            && [ ! -e "${CUDA_HOME}/include/cublas_v2.h" ]; then
            ln -sfn "${NVIDIA_CUBLAS_INCLUDE_PATH}/cublas_v2.h" \
                "${CUDA_HOME}/include/cublas_v2.h"
            if [ -f "${NVIDIA_CUBLAS_INCLUDE_PATH}/cublas_api.h" ]; then
                ln -sfn "${NVIDIA_CUBLAS_INCLUDE_PATH}/cublas_api.h" \
                    "${CUDA_HOME}/include/cublas_api.h"
            fi
            if [ -f "${NVIDIA_CUBLAS_INCLUDE_PATH}/cublas.h" ]; then
                ln -sfn "${NVIDIA_CUBLAS_INCLUDE_PATH}/cublas.h" \
                    "${CUDA_HOME}/include/cublas.h"
            fi
        fi
    fi
    if [ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
        cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}")
    fi

    if [ "${os_name}" = "Darwin" ]; then
        echo "NNTile 2.0 has no macOS product" >&2
        exit 1
    fi
    cmake_args+=(
        -DCMAKE_C_COMPILER="${CC:-gcc}"
        -DCMAKE_CXX_COMPILER="${CXX:-g++}"
    )

    cmake "${cmake_args[@]}"
}

assert_pip_cuda_link() {
    if [ "${NNTILE_ASSERT_PIP_CUDA:-1}" != "1" ]; then
        return 0
    fi
    if [ "${NNTILE_CUDA_FROM_PIP:-0}" != "1" ] \
        && [ -z "${NVIDIA_CUBLAS_LIBRARY_PATH:-}" ]; then
        return 0
    fi
    local lib=""
    shopt -s nullglob
    for candidate in \
        "${build_dir}/nntile/libnntile.so" \
        "${build_dir}/nntile/libnntile.so."* \
        "${build_dir}/libnntile.so" \
        "${build_dir}/libnntile.so."*; do
        if [ -f "${candidate}" ]; then
            lib="${candidate}"
            break
        fi
    done
    shopt -u nullglob
    if [ -z "${lib}" ]; then
        echo "assert_pip_cuda_link: libnntile.so not found under ${build_dir}" >&2
        return 1
    fi
    # Ensure ldd can see pip nvidia trees when asserting.
    export LD_LIBRARY_PATH="${NVIDIA_CUBLAS_LIBRARY_PATH:-}:${NVIDIA_CUDA_RUNTIME_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
    bash "${script_dir}/assert_pip_cuda_libs.sh" "${lib}"
}


build_nntile_libs() {
    configure_nntile_cmake
    cmake --build "${build_dir}" --target nntile torch_nntile -j "${jobs}"
    assert_pip_cuda_link
}

build_wheel_with_cmake() {
    configure_nntile_cmake
    cmake --build "${build_dir}" --target torch_nntile_wheel -j "${jobs}"
    assert_pip_cuda_link

    shopt -s nullglob
    built_wheels=("${build_dir}/wheelhouse"/*.whl)
    if [ "${#built_wheels[@]}" -eq 0 ]; then
        echo "No wheels produced under ${build_dir}/wheelhouse" >&2
        exit 1
    fi
    cp -f "${built_wheels[@]}" "${wheelhouse_out}/"
    echo "Copied wheels to ${wheelhouse_out}:"
    ls -la "${wheelhouse_out}"/*.whl
    shopt -u nullglob
}

case "${os_name}" in
    Linux)
        install_linux_packages
        # shellcheck disable=SC1091
        source "${script_dir}/install_linux_cuda_toolkit.sh"
        # shellcheck disable=SC1091
        source "${script_dir}/setup_torch_cuda_env.sh"
        export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
        python="$("${script_dir}/wheel_python.sh")"
        install_python_wheel_tools "${python}"
        export TORCH_NNTILE_PYTHON="${python}"
        ;;
    *)
        echo "Unsupported platform for torch_nntile wheel deps: ${os_name}" >&2
        exit 1
        ;;
esac

if [ "${cmake_wheel}" = "1" ]; then
    build_wheel_with_cmake
else
    build_nntile_libs
fi
