# NNTile subsystem builds

Unified library target **`nntile`** with per-subsystem switches `NNTILE_BUILD_*`.

## Stack (bottom to top)

`kernel` → `starpu` → `core` → `tile` → `tensor` → `nn` → `runtime` / `module` / `model` / …

- **core** — eager tile execution (former `tile` layer)
- **tile**, **tensor**, **nn** — graph layers (`TileGraph`, `TensorGraph`, `NNGraph`)

The deprecated eager **tensor** layer has been removed.

## Presets

| Preset | Use |
|--------|-----|
| `full` | Default development / PR integration |
| `core` | Kernel → StarPU → core only |
| `graph_min` | Through NN graph + runtime |

```bash
cmake -S . -B build -DNNTILE_PRESET=full
cmake --build build -j"$(nproc)"
```

## CI

`.github/workflows/build-test.yml` runs four jobs:

| Job | Purpose |
|-----|---------|
| `build-lib` | Configure and build `libnntile` |
| `build-tests` | Reconfigure with tests, build test binaries (`-j`) |
| `run-tests` | `ctest -j` (excludes MPI, NotImplemented, FixtureData) |
| `build-examples` | Link `nntile_all_examples` against the library from `build-lib` |

`build-tests` and `build-examples` both depend on `build-lib` and may run in parallel.

## Running tests locally

Per-subsystem test trees live under `nntile/tests/<subsystem>/`.
Enable with `BUILD_TESTS_<SUBSYSTEM>` or `NNTILE_TEST_SUBSYSTEM` (see `nntile/cmake/NNTileTests.cmake`).

```bash
cmake -S . -B build -DNNTILE_PRESET=full -DBUILD_TESTS=ON \
  -DNNTILE_TEST_SUBSYSTEM=nn
cmake --build build -j"$(nproc)"
ctest -R tests_graph_nn_
```

Plain `-DBUILD_TESTS=ON` without `NNTILE_TEST_SUBSYSTEM` enables all subsystem test trees.
