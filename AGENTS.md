# NNTile Development Guide

## Cursor Cloud specific instructions

### Environment overview

NNTile is a C++17/Python framework for distributed neural network training built on the StarPU runtime.
The Cloud Agent VM builds **CPU-only** (`-DUSE_CUDA=OFF`) since no GPU is available.

Libraries:

- **libnntile** — TensorGraph stack (kernel → StarPU → core → tile → tensor → Runtime)
- **libtorch_nntile** — LibTorch PrivateUse1 (`device=nntile`) + models (**required**)

### Pre-installed dependencies

- **StarPU 1.4.8** is installed at `/opt/starpu` (built from source).
  `PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig` is needed at CMake time.
  `LD_LIBRARY_PATH=/opt/starpu/lib` is needed at runtime.
- **System packages**: autoconf, automake, libtool, ninja-build, cmake, ccache,
  libhwloc-dev, libopenblas-dev, libfxt-dev, pkg-config, python3.12-dev, g++.
- **Python packages**: `torch==2.9.1` and `torchvision==0.24.1` (matching
  torch_nntile ABI; do **not** use torch 2.12 — incompatible ABI),
  numpy, scipy, transformers, pytest, ruff, isort, mypy, pre-commit, etc.
- The default `c++` is clang which lacks C++ stdlib headers in this image.
  Always pass `-DCMAKE_CXX_COMPILER=g++` to CMake.
- A symlink `/usr/lib/x86_64-linux-gnu/libstdc++.so` must point to
  `/usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.so` (created during setup).

### Building NNTile (CPU-only)

```bash
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
  -DBUILD_TESTS=OFF \
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

# torch_nntile Python tests (requires libnntile built + extension install)
export NNTILE_BUILD_DIR=$PWD/build NNTILE_SOURCE_DIR=$PWD
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib:$LD_LIBRARY_PATH
pytest -vv torch_nntile/tests/
```

### Known issues in CPU-only builds

- Many torch_nntile tests are skipped (`no cuda` marker) because CUDA is not available.

### Lint

```bash
pre-commit run --all-files
```

Uses ruff, isort, and standard pre-commit hooks. Configuration is in
`.pre-commit-config.yaml` and `pyproject.toml`.

### Graph API work (`graph_api` branch)

- **O(N) compiler design:** [docs/dev/graph_compiler_on_design.md](docs/dev/graph_compiler_on_design.md)
- **Agent checklist (actionable):** [docs/dev/graph_static_execution_agentic_plan.md](docs/dev/graph_static_execution_agentic_plan.md)
- **Roadmap:** [docs/dev/graph_static_execution_plan.md](docs/dev/graph_static_execution_plan.md)
- **Migration:** [docs/dev/libtorch_nntile_migration.md](docs/dev/libtorch_nntile_migration.md)
- **`torch_nntile` wheels:** built on PRs to `graph_api` and `workflow_dispatch`; see [torch_nntile/README.md](torch_nntile/README.md#prebuilt-wheels-001)
