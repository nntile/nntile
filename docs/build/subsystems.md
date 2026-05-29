# NNTile subsystem builds (Plan B)

Unified library target **`nntile`** with per-subsystem switches `NNTILE_BUILD_*`.

## Presets

| Preset | Use |
|--------|-----|
| `full` | Default development / PR integration |
| `core` | Kernel → StarPU → Tile only |
| `graph_min` | Through NNGraph + Runtime |

```bash
cmake -S . -B build -DNNTILE_PRESET=full
cmake --build build
```

## Compile-check (CI / local)

Compiles **only** one subsystem's `.cc` files into `nntile_objs_<name>` (no link):

```bash
cmake -S . -B build -DNNTILE_COMPILE_CHECK_SUBSYSTEM=tensor_graph \
  -DBUILD_TESTS=OFF
cmake --build build --target nntile_compile_check_tensor_graph
```

## Test compile-check (no libnntile)

Compiles test `.cc` files for one subsystem into `nntile_test_objs_<name>` (no link, no CTest):

```bash
cmake -S . -B build -DNNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM=nn_graph \
  -DBUILD_TESTS=OFF
cmake --build build --target nntile_compile_check_tests_nn_graph
```

## Running tests (requires full libnntile)

Per-subsystem test trees live directly under `nntile/tests/<subsystem>/` (e.g. `kernel/`,
`tile_graph/`). Enable with `BUILD_TESTS_<SUBSYSTEM>` (see `nntile/cmake/NNTileTests.cmake`).

CI order: source `compile-check-*` → `build-nntile` (full preset) →
`compile-check-tests-*` → `test-run-*` (link tests against `nntile`, then `ctest`).

```bash
cmake -S . -B build -DNNTILE_PRESET=full -DBUILD_TESTS=ON -DBUILD_TESTS_NN_GRAPH=ON \
  -DBUILD_TESTS_KERNEL=OFF ...
cmake --build build
ctest -R tests_graph_nn_
```

See `.github/scripts/cmake-test-subsystem.sh` (`NNTILE_BUILD_*`) and
`cmake-build-tests-subsystem.sh` (`BUILD_TESTS_*`).
