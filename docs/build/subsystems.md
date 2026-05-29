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

Compiles **only** one subsystem's `.cc` files into `nntile_objs_<name>` (no link).
CI packs them into `build/nntile_objs_cache/libnntile_objs_<name>.a` and caches that
archive for the link-only library job.

```bash
cmake -S . -B build -DNNTILE_COMPILE_CHECK_SUBSYSTEM=tensor \
  -DBUILD_TESTS=OFF
cmake --build build --target nntile_compile_check_tensor
```

Subsystems: see `.github/scripts/nntile-lib-obj-subsystems.txt` (`graph_base` is
`dtype.cc` / `kv_cache.cc`, separate from `tile` / `tensor` / `nn`).

## Bundled library archives (CI)

Job `bundle-nntile-lib-objs` runs once after all `compile-check-*` jobs finish,
restores every per-subsystem `libnntile_objs_*.a`, and saves a single cache key
`nntile-objs-all-a-<sha>` for the whole `build/nntile_objs_cache/` directory.
Downstream link jobs restore that bundle once instead of thirteen separate cache
lookups.

## build-nntile (CI, link only)

Job `build-nntile` configures `-DNNTILE_PRESET=full -DNNTILE_LINK_CACHED_OBJECTS=ON`,
restores the bundled archives, then links `libnntile`. CMake does **not** register
subsystem sources for compilation in that mode (only `context.cc`, logger sources,
and the shared-library link). It saves cache key `nntile-lib-linked-<sha>` with
`build/nntile/libnntile.so` and `build/include/nntile/defs.h` for `build-tests-*`.

## Test compile-check (no libnntile, no Catch2 build)

CI job `compile-check-tests-<subsystem>`: compiles test `.cc` files into an
OBJECT library only (no link, no `ctest`, no `libnntile`). It downloads
`build/_deps` from `build-test-prerequisites` (Catch2 sources plus
`catch2-build/generated-includes` for `catch_user_config.hpp`) and compiles with
header paths only — no `FetchContent_MakeAvailable(Catch2)` and no Catch2 library
build in that job. Catch2 libraries are built once in `build-test-prerequisites`.

```bash
cmake -S . -B build -DNNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM=nn \
  -DBUILD_TESTS=OFF
cmake --build build --target nntile_compile_check_tests_nn
```

## CI container image

GitHub Actions jobs run inside
`ghcr.io/nntile/nntile_sandbox:starpu1.4-9253daf_sgoc_pytorch2.9.1_cuda12.9.1`
(StarPU 1.4, build toolchain, PyTorch 2.9). Workflows set
`NNTILE_CI_CONTAINER_IMAGE` and do not run apt or compile StarPU on the runner.

## Test build and run (CI)

| Job | Purpose |
|-----|---------|
| `compile-check-*` | Compile one subsystem's lib sources; cache `nntile_objs_*` |
| `bundle-nntile-lib-objs` | Restore all lib archives once; save bundled cache |
| `build-nntile` | Link `libnntile` from bundled archives; cache `libnntile.so` |
| `build-test-prerequisites` | Build Catch2 once; cache `build/_deps` for test jobs |
| `subsystem-test-<name>` | One job per test subsystem (see below) |

`subsystem-test-<name>` runs `.github/workflows/subsystem-test-pipeline.yml` in a
**single container job**: restore upstream test-object caches (if any), compile
**only** that subsystem's test `.cc` once, link test executables once against
prebuilt `libnntile`, then `ctest`. There is no second compile/link pass and no
full `build/` tree cache between steps.

Ordering: `subsystem-test-starpu` needs `subsystem-test-kernel`;
`subsystem-test-core` needs `subsystem-test-starpu`. Each job saves
`nntile-test-obj-a-<subsystem>-<sha>` (one tarball) for downstream deps.

`ctest` uses `-LE NotImplemented`; for `model`, also `-LE FixtureData` (fixture
tests run in `test-full`).

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
