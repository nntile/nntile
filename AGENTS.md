# NNTile Development Guide

## Cursor Cloud specific instructions

### Environment overview

NNTile is a C++17/Python framework for distributed neural network training built on the StarPU runtime.
The Cloud Agent VM builds **CPU-only** (`-DUSE_CUDA=OFF`) since no GPU is available.

### Pre-installed dependencies

- **StarPU 1.4.8** is installed at `/opt/starpu` (built from source).
  `PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig` is needed at CMake time.
  `LD_LIBRARY_PATH=/opt/starpu/lib` is needed at runtime.
- **System packages**: autoconf, automake, libtool, ninja-build, cmake, ccache,
  libhwloc-dev, libopenblas-dev, libfxt-dev, pkg-config, python3.12-dev, g++.
- **Python packages**: torch (CPU-only via `https://download.pytorch.org/whl/cpu`),
  torchvision (CPU), numpy, scipy, transformers, pytest, ruff, isort, mypy, pre-commit, etc.
- The default `c++` is clang which lacks C++ stdlib headers in this image.
  Always pass `-DCMAKE_CXX_COMPILER=g++` to CMake.
- A symlink `/usr/lib/x86_64-linux-gnu/libstdc++.so` must point to
  `/usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so` (created during setup).

### Building NNTile (CPU-only)

```bash
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -GNinja
cmake --build build -j$(nproc)
```

Build takes ~12 minutes on 4 cores. ccache speeds up subsequent rebuilds significantly.

### Running tests

Standard commands from `docs/build/README.md`:

```bash
export LD_LIBRARY_PATH=/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1

# C++ tests (excluding MPI and NotImplemented)
ctest --test-dir build -E wrappers -LE "(MPI|NotImplemented)" --output-on-failure

# Python tests
export PYTHONPATH=$PWD/build/wrappers/python:$PWD/wrappers/python:$PYTHONPATH
pytest -vv
```

### Known issues in CPU-only builds

- `tests_graph_model_llama_*_data_setup` tests fail due to a torch/torchvision
  CPU registration incompatibility. These are not caused by NNTile code and are
  skipped in practice.
- Many Python tests are skipped (`no cuda` marker) because CUDA is not available.
- The `nntile_init()` function referenced in older examples has been replaced by
  `nntile.Context(ncpu=..., ncuda=..., ooc=..., logger=..., verbose=...)`.

### Lint

```bash
pre-commit run --all-files
```

Uses ruff, isort, and standard pre-commit hooks. Configuration is in
`.pre-commit-config.yaml` and `pyproject.toml`.
