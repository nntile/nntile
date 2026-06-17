#!/usr/bin/env bash
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

starpu_version="${STARPU_VERSION:-starpu-1.4.8}"
starpu_prefix="${STARPU_PREFIX:-/opt/starpu}"
build_dir="${NNTILE_BUILD_DIR:-${repo_root}/build/torch_nntile_wheel}"
jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
os_name="$(uname -s)"

install_linux_packages() {
    if command -v dnf >/dev/null 2>&1; then
        dnf install -y \
            autoconf automake bzip2 cmake curl gcc gcc-c++ git hwloc-devel \
            libtool make ninja-build openblas-devel pkgconf-pkg-config
    elif command -v yum >/dev/null 2>&1; then
        yum install -y \
            autoconf automake bzip2 cmake curl gcc gcc-c++ git hwloc-devel \
            libtool make ninja-build openblas-devel pkgconf-pkg-config
    elif command -v apt-get >/dev/null 2>&1; then
        apt-get update
        apt-get install -y --no-install-recommends \
            autoconf automake build-essential ca-certificates cmake curl git \
            libhwloc-dev libopenblas-dev libtool-bin ninja-build pkg-config
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
    for package in autoconf automake cmake hwloc libtool ninja pkg-config; do
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

build_starpu() {
    if PKG_CONFIG_PATH="${starpu_prefix}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}" \
        pkg-config --exists starpu-1.4; then
        return
    fi

    prepare_prefix
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "${tmp_dir}"' EXIT
    archive="${tmp_dir}/starpu.tar.gz"
    curl -SL \
        "https://gitlab.inria.fr/starpu/starpu/-/archive/${starpu_version}/starpu-${starpu_version}.tar.gz" \
        -o "${archive}"
    tar -xzf "${archive}" -C "${tmp_dir}"
    starpu_src="$(find "${tmp_dir}" -maxdepth 1 -type d -name 'starpu-*' -print -quit)"
    if [ -z "${starpu_src}" ]; then
        echo "StarPU source directory was not found after extraction" >&2
        exit 1
    fi

    (
        cd "${starpu_src}"
        ./autogen.sh
        ./configure \
            --disable-build-doc \
            --disable-build-examples \
            --disable-build-tests \
            --disable-cuda \
            --disable-fortran \
            --disable-mpi \
            --disable-opencl \
            --disable-socl \
            --disable-starpufft \
            --disable-starpupy \
            --enable-blas-lib=none \
            --enable-maxbuffers=16 \
            --without-fxt \
            --prefix="${starpu_prefix}" \
            --libdir="${starpu_prefix}/lib"
        make -j "${jobs}" install
    )

    rm -rf "${tmp_dir}"
    trap - EXIT
}

build_nntile() {
    export PKG_CONFIG_PATH="${starpu_prefix}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
    export LD_LIBRARY_PATH="${build_dir}/nntile:${starpu_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export DYLD_LIBRARY_PATH="${build_dir}/nntile:${starpu_prefix}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

    cmake_args=(
        -S "${repo_root}"
        -B "${build_dir}"
        -DCMAKE_BUILD_TYPE=Release
        -DUSE_CUDA=OFF
        -DBUILD_PYTHON_WRAPPERS=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_EXAMPLES=OFF
        -GNinja
    )
    if [ "${os_name}" = "Darwin" ]; then
        cmake_args+=(-DCMAKE_OSX_ARCHITECTURES=arm64)
    else
        cmake_args+=(
            -DCMAKE_C_COMPILER="${CC:-gcc}"
            -DCMAKE_CXX_COMPILER="${CXX:-g++}"
        )
    fi

    cmake "${cmake_args[@]}"
    cmake --build "${build_dir}" --target nntile -j "${jobs}"
}

case "${os_name}" in
    Linux) install_linux_packages ;;
    Darwin) install_macos_packages ;;
    *)
        echo "Unsupported platform for torch_nntile wheel deps: ${os_name}" >&2
        exit 1
        ;;
esac

build_starpu
build_nntile
