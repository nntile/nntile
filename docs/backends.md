<!-- SPDX-License-Identifier: MIT -->
# Backend: NNHaul (NNTile 2.0)

NNTile 2.0 has one execution backend: **NNHaul**. There is no second runtime
and no CPU backend.

| Property | Value |
|---|---|
| Runtime | NNHaul, in-tree `nnhaul/` or `NNHAUL_ROOT` |
| Device | one CUDA GPU per process (`ncuda=1`) |
| Tile memory | every tile is chunk-backed |
| Kernels | CUDA: chunk-addressable entry points; CPU: contiguous reference kernels |
| torch_nntile | classic `torch_nntile.nn` kernels and models |
| Stock `torch.nn` / HF on `device="nntile"` | not supported |
| CPU / macOS | not supported |
| Flash SDPA | not built |

## Build

```bash
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake --preset nnhaul -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build --preset nnhaul -j

# engine tests at the default class; nntile tests with the documented caps
sh tools/run_chunked_suite.sh build-nnhaul
```

A plain configure needs `USE_CUDA=ON`; a CUDA-free configure is a hard error.
There is no backend-selection option.

In this product `libnntile.so` links a shared `libnnhaul.so` (installed beside
it, found through `$ORIGIN`), so a consumer that links both sees one engine
instance. A standalone NNHaul build still produces a static archive.

`NNHAUL_ROOT=<prefix-or-build-dir>` links an external NNHaul instead of the
vendored `nnhaul/` subproject. That external tree must be built with
`-DNNHAUL_LIBRARY_TYPE=SHARED`; a static archive is refused. The resulting
nntile install still `find_dependency(nnhaul)`: keep that external prefix on
`CMAKE_PREFIX_PATH` for consumers, or install nntile with the in-tree nnhaul
for a self-contained prefix.

When configuring the test tree against an installed prefix, put it on
`CMAKE_PREFIX_PATH` (not `NNTILE_PREFIX`, which only feeds the wheel
bookkeeping). The tests-only tree adopts the installed library's CUDA and
disabled-kernel scope.

## Planner

The lowering planner is replaceable through the public NNHaul planner API
(`nnhaul/docs/planner-api.md`, `NNHAUL_PLANNER`). The in-tree LRU planner is
the default implementation.
