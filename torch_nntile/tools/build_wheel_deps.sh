#!/usr/bin/env bash
# Build StarPU + libnntile + libtorch_nntile for torch_nntile wheel packaging.
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
starpu_prefix="${STARPU_PREFIX:-/opt/starpu}"
build_dir="${NNTILE_BUILD_DIR:-${repo_root}/build/torch_nntile_wheel}"
wheelhouse_out="${TORCH_NNTILE_WHEELHOUSE:-${repo_root}/wheelhouse}"
jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
os_name="$(uname -s)"
use_cuda="${TORCH_NNTILE_USE_CUDA:-0}"
cmake_wheel="${TORCH_NNTILE_CMAKE_WHEEL:-0}"
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
    "${python}" -m pip install --no-cache-dir "setuptools>=61" wheel ninja numpy
    if [ "${os_name}" = "Darwin" ]; then
        "${python}" -m pip install --no-cache-dir delocate
    else
        "${python}" -m pip install --no-cache-dir auditwheel patchelf
    fi
}

install_torch_cpu() {
    local python="$("${script_dir}/wheel_python.sh")"
    install_python_wheel_tools "${python}"
    "${python}" -m pip install --no-cache-dir \
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
        # Thin toolkit has nvcc + cudart + libcuda stub only. CUBLAS headers
        # / libs come from pip nvidia-cublas (setup_torch_cuda_env.sh).
        configure_args+=(
            --enable-maxcudadev=8
            --without-fxt
            --with-cuda-dir="${CUDA_HOME}"
        )
        if [ -n "${NVIDIA_CUBLAS_INCLUDE_PATH:-}" ] \
            && [ -n "${NVIDIA_CUBLAS_LIBRARY_PATH:-}" ]; then
            # Pip ships only libcublas.so.12; autoconf -lcublas needs an
            # unversioned .so name. Symlink into a stable dir under prefix.
            cublas_link_dir="${starpu_prefix}/lib/nntile-cublas-link"
            mkdir -p "${cublas_link_dir}"
            for base in cublas cublasLt; do
                so="$(
                    ls -1 "${NVIDIA_CUBLAS_LIBRARY_PATH}/lib${base}.so."* \
                        2>/dev/null | head -1 || true
                )"
                if [ -z "${so}" ]; then
                    echo "missing lib${base}.so.* under " \
                        "${NVIDIA_CUBLAS_LIBRARY_PATH}" >&2
                    exit 1
                fi
                ln -sfn "${so}" "${cublas_link_dir}/lib${base}.so"
            done
            configure_args+=(
                --with-cublas-include-dir="${NVIDIA_CUBLAS_INCLUDE_PATH}"
                --with-cublas-lib-dir="${cublas_link_dir}"
            )
            export LD_LIBRARY_PATH="${NVIDIA_CUBLAS_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
            # starpu_cusolver.h includes cusolverDn.h whenever CUDA is on,
            # even if StarPU does not link cusolver. Headers come from pip.
            if [ -z "${NVIDIA_CUSOLVER_INCLUDE_PATH:-}" ] \
                || [ ! -f "${NVIDIA_CUSOLVER_INCLUDE_PATH}/cusolverDn.h" ]; then
                echo "StarPU CUDA build needs cusolverDn.h from pip " \
                    "nvidia-cusolver (NVIDIA_CUSOLVER_INCLUDE_PATH)" >&2
                exit 1
            fi
            starpu_cuda_includes=()
            for inc in \
                "${NVIDIA_CUSOLVER_INCLUDE_PATH:-}" \
                "${NVIDIA_CUSPARSE_INCLUDE_PATH:-}" \
                "${NVIDIA_CUBLAS_INCLUDE_PATH:-}" \
                "${NVIDIA_CUDA_RUNTIME_INCLUDE_PATH:-}"; do
                if [ -n "${inc}" ] && [ -d "${inc}" ]; then
                    starpu_cuda_includes+=("-I${inc}")
                fi
            done
            if [ "${#starpu_cuda_includes[@]}" -gt 0 ]; then
                starpu_cppflags="${starpu_cuda_includes[*]}"
                export CPPFLAGS="${starpu_cppflags}${CPPFLAGS:+ ${CPPFLAGS}}"
                export CFLAGS="${starpu_cppflags}${CFLAGS:+ ${CFLAGS}}"
                export CXXFLAGS="${starpu_cppflags}${CXXFLAGS:+ ${CXXFLAGS}}"
                # Pass on the configure cmdline so autoconf does not drop
                # env-only CPPFLAGS across its SAVE/restore checks.
                configure_args+=(
                    "CPPFLAGS=${CPPFLAGS}"
                    "CFLAGS=${CFLAGS}"
                    "CXXFLAGS=${CXXFLAGS}"
                )
            fi
        elif [ ! -f "${CUDA_HOME}/include/cublas.h" ]; then
            echo "StarPU CUDA build needs CUBLAS: set NVIDIA_CUBLAS_* " \
                "from setup_torch_cuda_env.sh, or install cublas into " \
                "${CUDA_HOME}" >&2
            exit 1
        fi
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

configure_nntile_cmake() {
    export PKG_CONFIG_PATH="${starpu_prefix}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
    export LD_LIBRARY_PATH="${build_dir}/nntile:${build_dir}/torch_nntile:${starpu_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export DYLD_LIBRARY_PATH="${build_dir}/nntile:${build_dir}/torch_nntile:${starpu_prefix}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

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
        -DNNTILE_TORCH_NATIVE_OPS=ON
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
}

assert_pip_cuda_link() {
    if [ "${use_cuda}" != "1" ]; then
        return 0
    fi
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
    export LD_LIBRARY_PATH="${NVIDIA_CUBLAS_LIBRARY_PATH:-}:${CUDNN_LIBRARY_PATH:-}:${NVIDIA_CUDA_RUNTIME_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
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
if [ "${cmake_wheel}" = "1" ]; then
    build_wheel_with_cmake
else
    build_nntile_libs
fi
