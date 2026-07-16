#!/usr/bin/env bash
# Build StarPU + libnntile + libtorch_nntile + torch_nntile wheel via CMake.
# No tests — compile/packaging only.
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
starpu_prefix="${STARPU_PREFIX:-/opt/starpu}"
build_dir="${NNTILE_BUILD_DIR:-${repo_root}/build/torch_nntile_wheel}"
wheelhouse_out="${TORCH_NNTILE_WHEELHOUSE:-${repo_root}/wheelhouse}"
jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
os_name="$(uname -s)"
use_cuda="${TORCH_NNTILE_USE_CUDA:-0}"
starpu_github_repo="${STARPU_GITHUB_REPO:-nntile/starpu}"
starpu_git_branch="${STARPU_GIT_BRANCH:-master}"

install_linux_packages() {
    if command -v dnf >/dev/null 2>&1; then
        dnf install -y \
            autoconf automake bzip2 cmake curl gcc gcc-c++ git hwloc-devel \
            libtool make ninja-build openblas-devel pkgconf-pkg-config unzip wget
    elif command -v yum >/dev/null 2>&1; then
        yum install -y \
            autoconf automake bzip2 cmake curl gcc gcc-c++ git hwloc-devel \
            libtool make ninja-build openblas-devel pkgconf-pkg-config unzip wget
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y --no-install-recommends \
            autoconf automake build-essential ca-certificates cmake curl git \
            libhwloc-dev libopenblas-dev libtool-bin ninja-build pkg-config \
            unzip wget python3 python3-dev python3-pip
    else
        echo "Unsupported Linux image: no yum or apt-get found" >&2
        exit 1
    fi
}

install_macos_packages() {
    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew is required for macOS wheel dependency setup" >&2
        exit 1
    fi
    for package in autoconf automake cmake hwloc libtool ninja pkg-config wget; do
        if ! brew list --versions "${package}" >/dev/null 2>&1; then
            brew install "${package}"
        fi
    done
    libtool_prefix="$(brew --prefix libtool)"
    export PATH="${libtool_prefix}/libexec/gnubin:${PATH}"
}

prepare_prefix() {
    if [ "${starpu_prefix}" = "/opt/starpu" ] && [ "${os_name}" = "Darwin" ]; then
        sudo mkdir -p "${starpu_prefix}"
        sudo chown -R "$(id -u):$(id -g)" "${starpu_prefix}"
    else
        mkdir -p "${starpu_prefix}"
    fi
}

install_python_wheel_tools() {
    local python="$1"
    "${python}" -m pip install --upgrade pip
    "${python}" -m pip install "setuptools>=61" wheel ninja numpy
    if [ "${os_name}" = "Darwin" ]; then
        "${python}" -m pip install delocate
    else
        "${python}" -m pip install auditwheel patchelf
    fi
}

install_torch_cpu() {
    local python="$("${script_dir}/wheel_python.sh")"
    install_python_wheel_tools "${python}"
    "${python}" -m pip install \
        "torch==${TORCH_VERSION:-2.9.1}" \
        "torchvision==0.24.1"
    export TORCH_PREFIX="$("${python}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    export CMAKE_PREFIX_PATH="${TORCH_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
    export TORCH_NNTILE_PYTHON="${python}"
}

build_starpu() {
    if PKG_CONFIG_PATH="${starpu_prefix}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}" \
        pkg-config --exists starpu-1.4; then
        return
    fi

    prepare_prefix
    tmp_parent="$(mktemp -d)"
    trap 'rm -rf "${tmp_parent}"' EXIT

    archive="${tmp_parent}/starpu.zip"
    curl -SL \
        "https://github.com/${starpu_github_repo}/archive/refs/heads/${starpu_git_branch}.zip" \
        -o "${archive}"
    unzip -q "${archive}" -d "${tmp_parent}"
    starpu_src="$(find "${tmp_parent}" -maxdepth 1 -type d -name 'starpu-*' -print -quit)"
    if [ -z "${starpu_src}" ]; then
        echo "StarPU source directory was not found after extraction" >&2
        exit 1
    fi

    configure_args=(
        --disable-build-doc
        --disable-build-examples
        --disable-build-tests
        --disable-fortran
        --disable-mpi
        --disable-opencl
        --disable-socl
        --disable-starpufft
        --disable-starpupy
        --enable-blas-lib=none
        --enable-maxbuffers=16
        --prefix="${starpu_prefix}"
        --libdir="${starpu_prefix}/lib"
    )

    if [ "${use_cuda}" = "1" ]; then
        configure_args+=(
            --enable-maxcudadev=8
            --without-fxt
            --with-cuda-dir="${CUDA_HOME}"
        )
    else
        configure_args+=(
            --disable-cuda
            --without-fxt
        )
    fi

    (
        cd "${starpu_src}"
        ./autogen.sh
        ./configure "${configure_args[@]}"
        make -j "${jobs}" install
    )

    rm -rf "${tmp_parent}"
    trap - EXIT
}

build_wheel_with_cmake() {
    export PKG_CONFIG_PATH="${starpu_prefix}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
    export LD_LIBRARY_PATH="${build_dir}/nntile:${build_dir}/torch_nntile:${starpu_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export DYLD_LIBRARY_PATH="${build_dir}/nntile:${build_dir}/torch_nntile:${starpu_prefix}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

    local python="${TORCH_NNTILE_PYTHON:-$("${script_dir}/wheel_python.sh")}"
    if [ -z "${TORCH_PREFIX:-}" ]; then
        export TORCH_PREFIX="$("${python}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    fi
    export CMAKE_PREFIX_PATH="${TORCH_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
    mkdir -p "${wheelhouse_out}"

    cmake_args=(
        -S "${repo_root}"
        -B "${build_dir}"
        -DCMAKE_BUILD_TYPE=Release
        -DBUILD_TESTS=OFF
        -DBUILD_TORCH_NNTILE=ON
        -DBUILD_TORCH_NNTILE_WHEEL=ON
        -DTORCH_NNTILE_WHEEL_REPAIR=ON
        -DTORCH_NNTILE_WHEELHOUSE="${build_dir}/wheelhouse"
        -DPython3_EXECUTABLE="${python}"
        -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
        -GNinja
    )

    if [ -n "${TORCH_NNTILE_WHEEL_VERSION:-}" ]; then
        cmake_args+=(-DTORCH_NNTILE_WHEEL_VERSION="${TORCH_NNTILE_WHEEL_VERSION}")
    fi

    if [ "${use_cuda}" = "1" ]; then
        export TORCH_NNTILE_USE_CUDA=1
        cmake_args+=(
            -DUSE_CUDA=ON
            -DCUDAToolkit_ROOT="${CUDA_HOME}"
        )
        if [ -n "${CMAKE_CUDA_COMPILER:-}" ]; then
            cmake_args+=(-DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER}")
        elif [ -n "${CUDA_HOME:-}" ]; then
            cmake_args+=(-DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc")
        fi
        if [ -n "${CUDNN_PATH:-}" ]; then
            cmake_args+=(
                -DCUDNN_PATH="${CUDNN_PATH}"
                -DCUDNN_INCLUDE_PATH="${CUDNN_INCLUDE_PATH}"
                -DCUDNN_LIBRARY_PATH="${CUDNN_LIBRARY_PATH}"
            )
        fi
        if [ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]; then
            cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}")
        fi
    else
        export TORCH_NNTILE_USE_CUDA=0
        cmake_args+=(-DUSE_CUDA=OFF)
    fi

    if [ "${os_name}" = "Darwin" ]; then
        export MACOSX_DEPLOYMENT_TARGET=14.0
        cmake_args+=(
            -DCMAKE_OSX_ARCHITECTURES=arm64
            -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
        )
    else
        cmake_args+=(
            -DCMAKE_C_COMPILER="${CC:-gcc}"
            -DCMAKE_CXX_COMPILER="${CXX:-g++}"
        )
    fi

    cmake "${cmake_args[@]}"
    cmake --build "${build_dir}" --target torch_nntile_wheel -j "${jobs}"

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
        if [ "${use_cuda}" = "1" ]; then
            # shellcheck disable=SC1091
            source "${script_dir}/install_linux_cuda_toolkit.sh"
            # shellcheck disable=SC1091
            source "${script_dir}/setup_torch_cuda_env.sh"
            export CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc"
            python="$("${script_dir}/wheel_python.sh")"
            install_python_wheel_tools "${python}"
            export TORCH_NNTILE_PYTHON="${python}"
        else
            install_torch_cpu
        fi
        ;;
    Darwin)
        install_macos_packages
        install_torch_cpu
        ;;
    *)
        echo "Unsupported platform for torch_nntile wheel deps: ${os_name}" >&2
        exit 1
        ;;
esac

build_starpu
build_wheel_with_cmake
