# C++ implementation overview

NNTile is split into two CMake packages:

| Package | Libraries | Umbrella headers |
|---------|-----------|------------------|
| **core** | `nntile` | [`include/nntile/core.hh`](../../include/nntile/core.hh) |
| **graph** | `nntile` (links core) | [`include/nntile/graph.hh`](../../include/nntile/graph.hh) |
| **full** | both | [`include/nntile.hh`](../../include/nntile.hh) |

Core code lives under `nntile/src/` and `include/nntile/` with namespace
`nntile::{kernel,starpu,tile,tensor,...}`. Graph code lives under
`nntile/src/` and `include/nntile/` with namespace `nntile::graph`.

```mermaid
flowchart TB
  subgraph graph_pkg [graph]
    TG[TensorGraph]
    NN[NNGraph]
    RT[Runtime]
  end
  subgraph core_pkg [core]
    Tensor[nntile::tensor]
    Tile[nntile::tile]
    StarPU[nntile::starpu]
    Kernel[nntile::kernel]
  end
  graph_pkg --> core_pkg
```

## Core layer headers

- [`include/nntile/kernel.hh`](../../include/nntile/kernel.hh)
- [`include/nntile/starpu.hh`](../../include/nntile/starpu.hh)
- [`include/nntile/tile.hh`](../../include/nntile/tile.hh)
- [`include/nntile/tensor.hh`](../../include/nntile/tensor.hh)

Sources mirror tests: `nntile/src/<level>/<op>.cc` ↔ `nntile/tests/<level>/<op>.cc`.

## kernel

**Namespace:** `nntile::kernel::<op>`

Raw numerical kernels on contiguous buffers (CPU and CUDA translation units under
`nntile/src/kernel/<op>/`).

## starpu

**Namespace:** `nntile::starpu`

StarPU codelets wrapping kernel calls.

## tile

**Namespace:** `nntile::tile`

Single-tile operations (`Tile<T>`).

## tensor

**Namespace:** `nntile::tensor`

Distributed tensors (`Tensor<T>`).

## graph

**Namespace:** `nntile::graph`

Symbolic graphs, lowering, runtime, modules, and optimizers. See
[`include/nntile/graph.hh`](../../include/nntile/graph.hh).
