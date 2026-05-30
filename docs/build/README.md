# Building and testing NNTile

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| CMake | ≥ 3.24 |
| Generator | **Ninja** or Unix Makefile (single-configuration only) |
| StarPU | 1.4 via `pkg-config starpu-1.4` (use the [nntile/starpu](https://github.com/nntile/starpu) fork for SGOC) |
| CUDA (default) | Toolkit ≥ 11.0, cuBLAS, cuDNN (cuDNN frontend is built from `external/cudnn_frontend`) |
| CPU BLAS | OpenBLAS or compatible when `USE_CBLAS=ON` |
| Python | 3.x + PyTorch if building LibTorch graph tests (`BUILD_TESTS_PYTORCH`) |
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
- `PYTHONPATH=/workspace/nntile/build/wrappers/python`
- Jupyter Lab: see [python/training.md](../python/training.md) (notebooks section)

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
export PYTHONPATH="$(pwd)/build/wrappers/python:${PYTHONPATH}"
```

CPU-only development:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF -GNinja
cmake --build build -j$(nproc)
```

## CMake options

Defined in [`CMakeLists.txt`](../../CMakeLists.txt):

| Option | Default | Effect |
|--------|---------|--------|
| `BUILD_SHARED_LIBS` | ON | Shared vs static `nntile` library |
| `USE_CUDA` | ON | CUDA, cuBLAS, cuDNN; OFF for CPU-only |
| `USE_CUDA_FP16` | ON | FP16 kernels (`NNTILE_USE_CUDA_FP16`) |
| `USE_CUDA_TF32` | ON | TF32 fast paths |
| `USE_CUDA_BF16` | ON | BF16 support |
| `USE_CUDA_FP8` | ON | FP8 if CUDA ≥ 11.8 |
| `USE_CBLAS` | ON | CPU BLAS kernels |
| `BUILD_TESTS` | ON | CTest suite (off in SimGrid mode) |
| `BUILD_TESTS_PYTORCH` | ON | Build WIP graph/NNGraph tests that compare against LibTorch (needs Torch; see below) |
| `BUILD_DOCS` | OFF | Doxygen documentation |
| `BUILD_EXAMPLES` | ON | C++ examples in `examples/` |
| `BUILD_COVERAGE` | OFF | LCOV coverage; enables tests; `make coverage` |
| `BUILD_PYTHON_WRAPPERS` | ON | pybind11 modules + Python package under `build/wrappers/python` |

### Common cache variables

| Variable | Use |
|----------|-----|
| `CMAKE_BUILD_TYPE` | `Release`, `RelWithDebInfo`, `Debug` |
| `CMAKE_CUDA_ARCHITECTURES` | Semicolon-separated SM versions for your GPUs |
| `CMAKE_PREFIX_PATH` | Conda prefix (StarPU, CUDA, PyTorch) |
| `CMAKE_DISABLE_FIND_PACKAGE_pybind11` | Pin in-tree pybind11 (Docker default) |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | ON by default → `compile_commands.json` |

### LibTorch and graph API tests (work in progress)

The [**NNTile Graph API**](../graph-wip.md) (`include/nntile/graph/`, `tests/graph/`,
`nntile_graph` Python bindings) is still under development. To **build and run the
graph test suite** that checks NNGraph results against PyTorch’s C++ frontend,
all of the following are required:

| Requirement | Notes |
|-------------|--------|
| `BUILD_TESTS=ON` | Default; graph tests live under `tests/graph/` |
| `BUILD_TESTS_PYTORCH=ON` | Default; enables LibTorch lookup at configure time |
| **LibTorch** on `CMAKE_PREFIX_PATH` | Usually from a PyTorch install in the same environment |
| StarPU **not** in SimGrid mode | Graph tests are disabled when `HAVE_STARPU_SIMGRID` is set |
| CUDA build (typical) | Graph tests are not built in CPU-only `USE_CUDA=OFF` CI configs |

If LibTorch is missing, CMake still configures but emits a warning and skips
Torch-linked graph targets (`NNTILE_HAVE_TORCH=OFF`). The rest of the C++ and
Python tests can still run.

Configure with PyTorch’s CMake prefix (merge with your Conda/StarPU prefix if needed):

```bash
TORCH_PREFIX="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="${CONDA_PREFIX};${TORCH_PREFIX}" \
  -DBUILD_TESTS=ON \
  -DBUILD_TESTS_PYTORCH=ON

cmake --build build -j$(nproc)
ctest --test-dir build -R tests_graph_ --output-on-failure
```

Graph API usage and architecture are **not** documented here; see
[graph-wip.md](../graph-wip.md) and [graph.md](../../graph.md).

## Build outputs

- `nntile` library and test binaries under `build/`
- Python package: `build/wrappers/python/nntile/` (`nntile_core`, `nntile_graph` extensions)
- Examples copied to `build/wrappers/python/examples/`
- No `make install` required for development — set `PYTHONPATH` to the build tree

## Running tests

Requires `BUILD_TESTS=ON` (default) and a finished build. Tests are skipped when
StarPU uses SimGrid.

### Running tests with CTest

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# C++ tests (same filter as CI: no wrappers, skip NotImplemented)
ctest --test-dir build -E wrappers -LE NotImplemented --output-on-failure

# Everything registered (C++ + pytest per file)
ctest --test-dir build --output-on-failure

# By prefix
ctest --test-dir build -R tests_kernel_add --output-on-failure
ctest --test-dir build -R tests_tensor_ --output-on-failure
ctest --test-dir build -R wrappers_python_tests_ --output-on-failure

# Parallel
ctest --test-dir build -j 8 -E wrappers --output-on-failure
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
| `tests/graph/` | `tests_graph_*` | various | WIP graph / autograd (needs LibTorch; see above) |

**Catch2** tests (most kernel ops): `Catch2::Catch2WithMain`, `TEMPLATE_TEST_CASE`,
reference checks, tags like `[add]`. Helpers: [`tests/testing.hh`](../../tests/testing.hh).

**Legacy** executables: some ops use standalone mains without Catch2.

Tests labeled `NotImplemented` are excluded in CI (`-LE NotImplemented`).

**Python tests** are registered in [`wrappers/python/CMakeLists.txt`](../../wrappers/python/CMakeLists.txt):
each `test_*.py` under `wrappers/python/tests/` is copied to
`build/wrappers/python/tests/` and run as
`python -m pytest -rx` with name `wrappers_python_tests_<dir>_<test>`.

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

### Example: Python tests and benchmarks

Layout and conventions: [`wrappers/python/tests/README.md`](../../wrappers/python/tests/README.md).

Fixtures in [`conftest.py`](../../wrappers/python/tests/conftest.py): `context`,
`context_cuda`, `benchmark_operation`, `benchmark_model`.

**Correctness:**

```bash
export PYTHONPATH="$(pwd)/build/wrappers/python:${PYTHONPATH}"
cd build/wrappers/python/tests

pytest -vv
pytest nntile_core/test_tensor_add_inplace.py
pytest nntile_core/test_tensor_add_inplace.py::test_add_inplace
pytest -k add_inplace --dtype=bf16
```

**One test via CTest:**

```bash
ctest --test-dir build -R wrappers_python_tests_nntile_core_tensor_add_inplace -V
```

**Benchmarks** (require `-m benchmark`; skipped by default):

```bash
cd build/wrappers/python/tests
pytest -m benchmark -vv
pytest -m benchmark -k test_rms_norm --dtype=fp32
```

Example benchmark test: `layer/test_rms_norm.py` (`@pytest.mark.benchmark`,
`context_cuda`, `benchmark_operation`).

### Python coverage (CI pattern)

From repository root with a venv and built extensions:

```bash
pytest -vv --cov=wrappers/python/nntile wrappers/python/tests
```

## See also

- [cpp/README.md](../cpp/README.md) — C++ layers under test
- [sgoc/README.md](../sgoc/README.md) — SGOC built in Docker sandbox
- [STYLE_GUIDE.md](../../STYLE_GUIDE.md) — C++ coding style
