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
cmake --build build
```

## Compile-check (CI / local)

Compiles **only** one subsystem's `.cc` files into `nntile_objs_<name>` (no link):

```bash
cmake -S . -B build -DNNTILE_COMPILE_CHECK_SUBSYSTEM=tensor \
  -DBUILD_TESTS=OFF
cmake --build build --target nntile_compile_check_tensor
```

## Test compile-check (no libnntile)

CI job `compile-check-tests-<subsystem>`: compiles test `.cc` files into an
OBJECT library only (no link, no `ctest`).

```bash
cmake -S . -B build -DNNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM=nn \
  -DBUILD_TESTS=OFF
cmake --build build --target nntile_compile_check_tests_nn
```

## Test build and run (CI)

PR workflow uses three separate test stages per subsystem:

| Job | Purpose |
|-----|---------|
| `build-test-prerequisites` | Build Catch2 once; cache `build/_deps` for test jobs |
| `compile-check-tests-*` | Compile test sources only (reuse cached Catch2) |
| `test-build-*` | `-DNNTILE_TEST_SUBSYSTEM=<name>`, link only that subsystem's test binaries |
| `test-run-*` | Restore build tree from cache, `ctest -R` only (no compile) |

`test-build` does **not** run `cmake --build` on the default `all` target (that would
build every enabled test). It builds `nntile` plus only the CTest targets for the
matrix subsystem.

## Running tests locally (requires full libnntile)

Per-subsystem test trees live under `nntile/tests/<subsystem>/`.
Enable with `BUILD_TESTS_<SUBSYSTEM>` (see `nntile/cmake/NNTileTests.cmake`).

```bash
cmake -S . -B build -DNNTILE_PRESET=full -DBUILD_TESTS=ON \
  -DNNTILE_TEST_SUBSYSTEM=nn
cmake --build build --target nntile $(.github/scripts/cmake-test-build-targets.sh nn build)
ctest -R tests_graph_nn_
```

Plain `-DBUILD_TESTS=ON` without `NNTILE_TEST_SUBSYSTEM` enables all subsystem
test trees (local full test build).

See `.github/scripts/cmake-test-subsystem.sh` (`NNTILE_BUILD_*`) and
`cmake-build-tests-subsystem.sh` / `NNTILE_TEST_SUBSYSTEM`.
