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

## Tests by subsystem

```bash
ctest -L SUBSYSTEM_NN_GRAPH
```

See `.github/scripts/cmake-test-subsystem.sh` for cumulative `-DNNTILE_BUILD_*` flags used in CI test jobs.
