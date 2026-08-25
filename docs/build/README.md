# Building and testing NNTile

**Default CMake** builds **libnntile** (`BUILD_LIBNNTILE`),
**libtorch_nntile** (`BUILD_LIBTORCH_NNTILE`), and the installable
**torch_nntile** Python product (`BUILD_TORCH_NNTILE`: `_C` + wheel). The
last two default **ON** and need PyTorch on `CMAKE_PREFIX_PATH`. Install the
wheel from `build/wheelhouse/` or see
[torch_nntile/README.md](../../torch_nntile/README.md) for prebuilt artifacts.

Layered CI and Torch-less configures opt out explicitly:

```bash
cmake -S . -B build -GNinja \
  -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF
```

Release builds also use cibuildwheel or
[`torch_nntile/tools/build_wheel_deps.sh`](../../torch_nntile/tools/build_wheel_deps.sh).

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| CMake | ≥ 3.24 |
| Generator | **Ninja** or Unix Makefile (single-configuration only) |
| StarPU | 1.4 via `pkg-config starpu-1.4` (use the [nntile/starpu](https://github.com/nntile/starpu) fork for SGOC) |
| CUDA (default) | Toolkit ≥ 11.0, cuBLAS, cuDNN (cuDNN frontend is built from `external/cudnn_frontend`) |
| CPU BLAS | OpenBLAS or compatible when `USE_CBLAS=ON` |
| Python | 3.x; **PyTorch 2.9.1** only when building libtorch_nntile / torch_nntile |
| GPU | Compute capability ≥ 8.0 |

If StarPU was built in **SimGrid** mode, the build emulates CUDA/CBLAS at compile
time and disables the normal test suite.

## Docker (recommended)

[`Dockerfile`](../../Dockerfile) provides two stages:

| Stage | Build | Contents |
|-------|-------|----------|
| `sandbox` | `docker build . -t nntile_sandbox:latest --target sandbox` | Conda env `nntile`, FXT, StarPU + SGOC DSO, PyTorch, Jupyter — **no NNTile compile** |
| `nntile` (default) | `docker build . -t nntile:latest` | Above + NNTile built in `/workspace/nntile/build` |

### Docker build arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `CUDA_VERSION` | 12.9.1 | CUDA base image version |
| `BASE_OS` | ubuntu22.04 | Base OS tag |
| `BASE_IMAGE` | `nvidia/cuda:…` | Override full base image |
| `MAKE_JOBS` | 4 | Parallel jobs (FXT, StarPU, NNTile) |
| `CUDA_ARCHS` | `70;75;80;86;89;90;100;120` | Passed to `-DCMAKE_CUDA_ARCHITECTURES` |
| `PYTHON_VERSION` | 3.12 | Conda Python |
| `PYTORCH_VERSION` | 2.9.1 | Conda PyTorch |
| `STARPU_GITHUB_REPO` | `nntile/starpu` | StarPU source repository |
| `STARPU_GIT_BRANCH` | `master` | StarPU branch |
| `FXT_VERSION` | 0.3.15 | FXT tracing library for StarPU |

Example:

```shell
docker build . -t nntile:latest \
  --build-arg MAKE_JOBS=8 \
  --build-arg CUDA_ARCHS="80;86;89;90"
```

### CMake inside the `nntile` image

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_DISABLE_FIND_PACKAGE_pybind11=ON \
  -GNinja
cmake --build build -j ${MAKE_JOBS}
```

`CMAKE_DISABLE_FIND_PACKAGE_pybind11=ON` forces the in-tree pybind11 2.11.0
needed for stable Python bindings.

### Running the container

```shell
docker run -it --gpus all nntile:latest
```

- Working directory: `/workspace/nntile`
- Jupyter / training examples: see [torch_nntile](../torch_nntile.md)

SGOC library path and env vars: [sgoc/README.md](../sgoc/README.md).

## Native build

After StarPU (and optional SGOC DSO) are installed:

```bash
conda activate nntile   # env with starpu-1.4 on PKG_CONFIG_PATH

cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
  -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"

cmake --build build -j$(nproc)
```

CPU-only development (product defaults; needs PyTorch):

```bash
TORCH_PREFIX="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
  -DCMAKE_PREFIX_PATH="${TORCH_PREFIX}" -GNinja
cmake --build build -j$(nproc)
```

Libnntile-only (no LibTorch):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
  -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF -GNinja
cmake --build build -j$(nproc)
```

## CMake options

Defined in [`CMakeLists.txt`](../../CMakeLists.txt):

| Option | Default | Effect |
|--------|---------|--------|
| `BUILD_SHARED_LIBS` | ON | Shared vs static **libnntile** |
| `BUILD_LIBNNTILE` | ON | Build the core C++ **libnntile** library |
| `USE_CUDA` | ON | CUDA, cuBLAS, cuDNN; OFF for CPU-only |
| `USE_CUDA_FP16` | ON | FP16 kernels (`NNTILE_USE_CUDA_FP16`) |
| `USE_CUDA_TF32` | ON | TF32 fast paths |
| `USE_CUDA_BF16` | ON | BF16 support |
| `USE_CUDA_FP8` | ON | FP8 if CUDA ≥ 11.8 |
| `USE_CBLAS` | ON | CPU BLAS kernels |
| `BUILD_TESTING` | ON | Standard CMake CTest switch; tests are built only for enabled components (`BUILD_LIBNNTILE` / `BUILD_LIBTORCH_NNTILE` / `BUILD_TORCH_NNTILE`). With all three OFF, layered CI links tests against an install prefix |
| `BUILD_DOCS` | OFF | Doxygen documentation |
| `BUILD_LIBTORCH_NNTILE` | ON | Build C++ **libtorch_nntile** (LibTorch PrivateUse1 bridge; requires LibTorch) |
| `BUILD_LIBTORCH_NNTILE_EXAMPLES` | OFF | C++ examples under `torch_nntile/examples` |
| `BUILD_TORCH_NNTILE` | ON | Build the installable **torch_nntile** Python product (`_C` extension + wheel; use `-DNNTILE_PREFIX` to skip rebuilding libs) |
| `BUILD_COVERAGE` | OFF | LCOV coverage; forces `BUILD_TESTING=ON`; `make coverage` |

### Common cache variables

| Variable | Use |
|----------|-----|
| `CMAKE_BUILD_TYPE` | `Release`, `RelWithDebInfo`, `Debug` |
| `CMAKE_CUDA_ARCHITECTURES` | Semicolon-separated SM versions for your GPUs |
| `CMAKE_PREFIX_PATH` | Conda prefix (StarPU, CUDA, PyTorch) |
| `CMAKE_DISABLE_FIND_PACKAGE_pybind11` | Pin in-tree pybind11 (Docker default) |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | ON by default → `compile_commands.json` |

### LibTorch / torch_nntile

**libnntile** C++ tests do **not** need PyTorch. Default CMake enables
`BUILD_LIBTORCH_NNTILE` and `BUILD_TORCH_NNTILE`, so put LibTorch on
`CMAKE_PREFIX_PATH` (from `torch.utils.cmake_prefix_path`). Pass both options
**OFF** for a Torch-less / libnntile-only configure (as layered CI does).

Both **libnntile** and **libtorch_nntile** install CMake packages
(`find_package(nntile)` / `find_package(torch_nntile)`).

Default in-tree product build (libs + wheel):

```bash
TORCH_PREFIX="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX};${TORCH_PREFIX}"

cmake --build build -j$(nproc)              # includes torch_nntile_wheel
# Wheels land in build/wheelhouse/*.whl
cmake --install build --prefix "$PWD/install"
# .whl also installed under share/torch_nntile/wheels/
```

Separate build against an installed libnntile:

```bash
cmake -S . -B build-torch -GNinja \
  -DBUILD_LIBNNTILE=OFF -DBUILD_LIBTORCH_NNTILE=ON \
  -DBUILD_TORCH_NNTILE=OFF \
  -DCMAKE_PREFIX_PATH="$PWD/install;${TORCH_PREFIX}"
cmake --build build-torch -j$(nproc)
cmake --install build-torch --prefix "$PWD/install"

# Product wheel only (reuse install; do not rebuild libs)
cmake -S . -B build-wheel -GNinja \
  -DBUILD_LIBNNTILE=OFF -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TESTING=OFF \
  -DBUILD_TORCH_NNTILE=ON -DTORCH_NNTILE_WHEEL_REPAIR=OFF \
  -DNNTILE_PREFIX="$PWD/install" -DTORCH_NNTILE_PREFIX="$PWD/install" \
  -DCMAKE_PREFIX_PATH="$PWD/install;${TORCH_PREFIX}"
cmake --build build-wheel --target torch_nntile_wheel
```

Consumer:

```cmake
find_package(nntile REQUIRED CONFIG)
find_package(torch_nntile REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE torch_nntile::torch_nntile)
```

C++ TensorGraph tests (against an installed libnntile, no library rebuild):

```bash
cmake -S . -B build-tests -GNinja \
  -DBUILD_LIBNNTILE=OFF -DBUILD_LIBTORCH_NNTILE=OFF \
  -DBUILD_TORCH_NNTILE=OFF -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build-tests -j$(nproc)
export LD_LIBRARY_PATH="$PWD/install/lib:${LD_LIBRARY_PATH:-}"
ctest --test-dir build-tests -R 'tests_(tile|tensor|core|kernel|starpu)_' \
  --output-on-failure
```

See [torch_nntile.md](../torch_nntile.md) and [graph.md](../graph.md).

### CUDA runtime (source / conda)

CUDA-enabled **libnntile** / **libtorch_nntile** link against NVIDIA math
libraries at build time and need them on ``LD_LIBRARY_PATH`` at import/run.
You do **not** need a separate ``pip install nvidia-*-cu12`` stack when PyTorch
and CUDA already come from the same conda env (or when ``torch`` pulled those
pip packages as its own dependencies).

**CMake** — point at LibTorch (same as above):

```bash
TORCH_PREFIX="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
```

**Python runtime** — export before ``import torch_nntile``, pytest, or training
examples:

```bash
export TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export NNTILE_BUILD_DIR=$PWD/build
export TORCH_NNTILE_BUILD_DIR=$PWD/build
export NNTILE_SOURCE_DIR=$PWD
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:$PWD/build/nntile:$PWD/build/torch_nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
```

| Variable | Role |
|----------|------|
| ``TORCH_PREFIX`` | CMake ``CMAKE_PREFIX_PATH`` entry for LibTorch |
| ``TORCH_LIB_DIR`` | ``torch/lib`` (`libtorch_cuda.so`, ``libc10_cuda.so``, …) |
| ``${CONDA_PREFIX}/lib`` | Conda CUDA math libs (`libcublas`, ``libcudnn``, ``libcudart``, …) when using a conda-forge CUDA stack |
| ``NNTILE_BUILD_DIR`` / ``TORCH_NNTILE_BUILD_DIR`` | In-tree ``torch_nntile._C`` + ``libnntile`` / ``libtorch_nntile`` when not using ``pip install`` |

``torch_nntile`` import checks that CUDA 12 sonames are visible under
``TORCH_LIB_DIR``, ``CONDA_PREFIX/lib``, ``CUDA_HOME``, ``LD_LIBRARY_PATH``,
or the pip ``site-packages/nvidia/*/lib`` layout (wheels / ``torch`` cu128
deps). See ``torch_nntile/torch_nntile/_cuda_deps.py``.

**pip / manylinux wheel CI** still uses
[`torch_nntile/tools/setup_torch_cuda_env.sh`](../../torch_nntile/tools/setup_torch_cuda_env.sh)
to install ``torch==2.9.1+cu128`` and wire pip ``nvidia-*-cu12`` paths — that
is the wheel-build path, not required for a conda dev env that already has
CUDA + torch.

**Editable install** (after the C++ build):

```bash
CXX=g++ pip install -e ./torch_nntile --no-build-isolation
# or PYTHONPATH=$PWD/torch_nntile when developing without reinstalling _C
```


- Core library and C++ test binaries under `build/`, including
  `build/nntile/libnntile.*`
- LibTorch bridge under `build/torch_nntile/libtorch_nntile.*` when
  `BUILD_LIBTORCH_NNTILE=ON`
- Installable Python wheels under `build/wheelhouse/torch_nntile-*.whl` when
  `BUILD_TORCH_NNTILE=ON`

## Build and test CI (layered)

[`.github/workflows/build-test.yml`](../../.github/workflows/build-test.yml)
runs on pushes/PRs to `main` and `graph_api`. On
`graph_api_torch_kernels` / `NNTILE_TORCH_NATIVE_OPS=ON`, libnntile links
LibTorch (StarPU aten codelets) and CTest labels show progress of the
torch-native stack:

| Job | Depends on | Role |
|-----|------------|------|
| `build-libnntile` | — | Build + install libnntile with **LibTorch** + `NNTILE_TORCH_NATIVE_OPS=ON` (`BUILD_LIBTORCH_NNTILE=OFF` + `BUILD_TORCH_NNTILE=OFF`) |
| `test-libnntile` | prefix | Build **tests only** vs install; `ctest -L torch_native --no-tests=error` (starpu/core/tile/tensor) |
| `build-libtorch-nntile` | libnntile prefix | Build + install libtorch_nntile against prefix (slim PrivateUse1) |
| `test-libtorch-nntile` | torch prefix | `ctest -L libtorch_nntile --no-tests=error` (aten parity progress) |
| `build-torch-nntile` | torch prefix | CMake `-DBUILD_TORCH_NNTILE=ON` + `-DNNTILE_PREFIX` (extension + wheel; no lib rebuild) |
| `test-torch-nntile` | CI wheel | Install wheel + assert `TORCH_NATIVE_OPS` + Python smoke allowlist |

`test-libtorch-nntile` does **not** depend on the Python wheel: it links
`torch_nntile::torch_nntile` from the install prefix (`BUILD_TESTING=ON` with
product libs OFF). Aten op fwd+bwd parity is C++ ctest here, not pytest.
`test-torch-nntile` covers device/context smoke under the slim wheel.

Install packaging is exercised by the stepwise build → test consumers
(`find_package` / wheel / ctest), not by separate install-layout jobs.

Python tests always consume the wheel from `build-torch-nntile`, not
an editable install. This is separate from the release/manylinux wheel
workflow below.

## torch_nntile wheels (CI)

Prebuilt `torch_nntile` wheels are built by the **`torch_nntile wheels`**
workflow ([`.github/workflows/torch-nntile-wheels.yml`](../../.github/workflows/torch-nntile-wheels.yml))
via **cibuildwheel**. `before-all` builds StarPU + libnntile + libtorch_nntile;
cibuildwheel builds the Python extension, repairs the wheel, and runs
[`tools/smoke_test_wheel.py`](../../torch_nntile/tools/smoke_test_wheel.py).
Full pytest suites stay in **build-and-test**.

| | |
|-|-|
| **Trigger** | Pull requests to `graph_api`, or `workflow_dispatch` |
| **Skipped when** | PR closed without merge |
| **Tooling** | cibuildwheel + [`tools/build_wheel_deps.sh`](../../torch_nntile/tools/build_wheel_deps.sh) |
| **Smoke** | `tools/smoke_test_wheel.py` (cibuildwheel `test-command`) |
| **Version** | `0.0.6` (`TORCH_NNTILE_WHEEL_VERSION`) |

### Triggering

| Goal | Action |
|------|--------|
| Build from a PR branch | Push to the PR (automatic) |
| Build after landing changes | Merge PR into `graph_api` |
| Rebuild without a merge | `gh workflow run torch-nntile-wheels.yml --ref graph_api` |
| Download artifacts | `gh run download RUN_ID -D wheelhouse` |

### Matrix

| Job | Runner | Wheel tag | CUDA |
|-----|--------|-----------|------|
| `cp312-manylinux_x86_64` | `ubuntu-24.04` + manylinux_2_28 container | manylinux x86_64 | Yes (`torch==2.9.1` cu128) |
| `cp312-macosx_arm64` | `macos-14` | macOS 14+ arm64 | No (CPU StarPU) |

Artifacts: `torch-nntile-wheel-cp312-manylinux_x86_64`,
`torch-nntile-wheel-cp312-macosx_arm64`.

### Build pipeline

| Script | Role |
|--------|------|
| `build_wheel_deps.sh` | cibuildwheel `before-all`: StarPU + libnntile + libtorch_nntile |
| `smoke_test_wheel.py` | cibuildwheel `test-command` (import + tiny add on `nntile`) |
| `install_linux_cuda_toolkit.sh` | manylinux: dnf CUDA 12.8 toolkit |
| `setup_torch_cuda_env.sh` | Linux CUDA: torch cu128 + pip cuDNN |
| `repair_wheel_linux.sh` / `repair_wheel_macos.sh` | auditwheel / delocate |

Optional CMake-only wheel (no cibuildwheel):

```bash
export TORCH_NNTILE_CMAKE_WHEEL=1
export TORCH_NNTILE_USE_CUDA=0   # or 1 on Linux with CUDA toolkit
bash torch_nntile/tools/build_wheel_deps.sh "$PWD"
# → wheelhouse/*.whl
```

Or plain CMake:

```bash
cmake -S . -B build -GNinja -DUSE_CUDA=OFF -DBUILD_TESTING=OFF \
  -DBUILD_TORCH_NNTILE=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build --target torch_nntile_wheel
```

PyTorch is **not** bundled; wheels declare `torch==2.9.1` as a runtime
dependency. CUDA-built Linux wheels also declare `nvidia-*-cu12`; CPU wheels
do not. Import checks NVIDIA libs only when `built_with_cuda()` is true.

### Download and publish

```bash
gh run list --workflow=torch-nntile-wheels.yml --limit 5
gh run download RUN_ID -D wheelhouse
gh workflow run torch-nntile-wheels.yml --ref graph_api   # manual rebuild
```

End-user install instructions:
[`torch_nntile/README.md`](../../torch_nntile/README.md#prebuilt-wheels-001).

PyPI upload is manual (`twine upload` from downloaded artifacts).

**CUDA without a GPU (Linux wheel CI):** GitHub-hosted runners have no NVIDIA
device. StarPU and libnntile can still be built with CUDA enabled: the
manylinux `before-all` hook installs a **thin** CUDA 12.8 toolkit via dnf
(`cuda-minimal-build` for `nvcc`/`cuda.h`, `cuda-driver-devel` for
`lib64/stubs/libcuda.so`). Math/runtime libs (`cudart`, `cublas`, `cudnn`)
come from the same pip `nvidia-*-cu12` packages that `torch` (cu128) installs
— see `cmake/NNTilePreferPipCuda.cmake` and
[`docs/dev/cuda_wheel_single_nvidia_stack_plan.md`](../dev/cuda_wheel_single_nvidia_stack_plan.md).
A driver is only required at runtime when using CUDA StarPU workers
(`ncuda > 0`).

## Running tests

Requires `BUILD_TESTING=ON` (default) and a finished build. Tests are skipped when
StarPU uses SimGrid.

### Running tests with CTest

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# C++ tests (same filter as CI: skip NotImplemented)
ctest --test-dir build -LE NotImplemented --output-on-failure

# Everything registered
ctest --test-dir build --output-on-failure

# By prefix
ctest --test-dir build -R tests_kernel_add --output-on-failure
ctest --test-dir build -R tests_tensor_ --output-on-failure

# Parallel
ctest --test-dir build -j 8 --output-on-failure
```

Useful flags: `-R` / `-E` (regex), `-L` / `-LE` (labels), `--output-on-failure`.

Coverage (`-DBUILD_COVERAGE=ON`, Debug recommended):

```bash
cmake --build build --target coverage
```

### How C++ tests are implemented

[`tests/CMakeLists.txt`](../../tests/CMakeLists.txt) defines `add_test_set()`:
one executable per operation, registered with CTest. Multiple argument sets
become `target_name_1`, `target_name_2`, …

| Directory | CTest prefix | Executable | Level |
|-----------|--------------|------------|-------|
| `tests/kernel/` | `tests_kernel_<op>` | `test_<op>` | `nntile::kernel` |
| `tests/starpu/` | `tests_starpu_<op>` | `test_<op>` | StarPU codelets |
| `tests/core/` | `tests_core_<op>` | `test_<op>` | `Tile<T>` |
| `tests/tensor/` | `tests_tensor_<op>` | `test_<op>` | `Tensor<T>` |
| `tests/tile/`, `tests/tensor/`, … | `tests_tile_*`, `tests_tensor_*`, … | various | Graph / autograd (needs LibTorch; see above) |

**Catch2** tests (most kernel ops): `Catch2::Catch2WithMain`, `TEMPLATE_TEST_CASE`,
reference checks, tags like `[add]`. Helpers: [`tests/testing.hh`](../../tests/testing.hh).

**Legacy** executables: some ops use standalone mains without Catch2.

Tests labeled `NotImplemented` are excluded in CI (`-LE NotImplemented`).

**Python tests** live under [`torch_nntile/tests/`](../../torch_nntile/tests/)
and are run directly with pytest. They require `libnntile`, `libtorch_nntile`,
and StarPU on the runtime library path, plus `NNTILE_BUILD_DIR` (and
`TORCH_NNTILE_BUILD_DIR` when using the in-tree extension build).

### Example: C++ kernel benchmark

Benchmarks use Catch2 tag `[!benchmark]` (skipped in default test runs).
See [`tests/kernel/add.cc`](../../tests/kernel/add.cc).

```bash
# Correctness (default ctest run for this target)
ctest --test-dir build -R tests_kernel_add --output-on-failure

# Run benchmark cases directly
./build/tests/kernel/test_add '[add][!benchmark]' -d yes

# Narrower filter
./build/tests/kernel/test_add '[add][!benchmark]' -d yes 'nelems=1048576' cuda
```

`ctest` invokes the binary without Catch2 filters, so it runs verification tests,
not `[!benchmark]` sections.

### Example: torch_nntile Python tests

Layout and fixtures: [`torch_nntile/tests/conftest.py`](../../torch_nntile/tests/conftest.py).

**Correctness:**

```bash
# Requires libnntile + libtorch_nntile + StarPU (+ CUDA libs on GPU builds).
export TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export NNTILE_BUILD_DIR=$PWD/build
export TORCH_NNTILE_BUILD_DIR=$PWD/build
export NNTILE_SOURCE_DIR=$PWD
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:$PWD/build/nntile:$PWD/build/torch_nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

pytest -vv torch_nntile/tests/
pytest -vv torch_nntile/tests/test_add_inplace_parity.py
pytest -vv torch_nntile/tests/test_add_inplace_parity.py::test_add_inplace_matches_cpu
```

**Installed wheel:**

```bash
pip install build/wheelhouse/torch_nntile-*.whl
export TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
pytest -vv torch_nntile/tests/
```

On pip-only torch (cu128) without conda CUDA, the wheel's ``nvidia-*-cu12``
dependencies supply math libs instead of ``${CONDA_PREFIX}/lib``.

### Python coverage (CI pattern)

From repository root with a venv and built or installed `torch_nntile`:

```bash
pytest -vv --cov=torch_nntile torch_nntile/tests/
```

## See also

- [cpp/README.md](../cpp/README.md) — C++ layers under test
- [torch_nntile.md](../torch_nntile.md) — PyTorch `device="nntile"` bridge
- [sgoc/README.md](../sgoc/README.md) — SGOC built in Docker sandbox
- [STYLE_GUIDE.md](../../STYLE_GUIDE.md) — C++ coding style
