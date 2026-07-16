# NNTile documentation

**NNTile** is a framework for training large neural networks on distributed-memory
heterogeneous systems. It uses [StarPU](https://starpu.gitlabpages.inria.fr) for
task-based scheduling and asynchronous data movement across CPU and GPU resources.

The primary path on the `graph_api` branch is the **Graph API**: deferred
TensorGraph → TileGraph → Runtime, exposed to users as PyTorch
`device="nntile"` via **torch_nntile** / **libtorch_nntile**. The Graph API is
still evolving (WIP), but it is the main product surface — not a side experiment.

## Quick start

| Goal | Start here |
|------|------------|
| **Graph API (main, WIP)** | [graph.md](graph.md) |
| PyTorch `device="nntile"` (torch_nntile) | [torch_nntile.md](torch_nntile.md) |
| Install prebuilt `torch_nntile` wheel (CI) | [torch_nntile.md#prebuilt-wheels](torch_nntile.md#prebuilt-wheels) |
| Build from source or Docker | [build/README.md](build/README.md) |
| C++ stack (kernel → TensorGraph → Runtime) | [cpp/README.md](cpp/README.md) |
| Design notes (`docs/dev`) | [dev/README.md](dev/README.md) |
| Inference, gateway, Telegram bot | [inference/README.md](inference/README.md) |
| SGOC scheduler (limited VRAM, single GPU) | [sgoc/README.md](sgoc/README.md) |

## Documentation map

```
docs/
  README.md                 ← you are here
  graph.md                  Graph API overview (main entry)
  torch_nntile.md           PyTorch device="nntile", tiling, wheels
  build/README.md           Build, CMake, Docker, testing, wheel CI
  cpp/README.md             C++ kernel / starpu / tile / tensor / runtime
  dev/README.md             Design notes index
  sgoc/README.md            SGOC StarPU scheduler
  inference/README.md       Inference, nntile_gateway, nntile_tgbot
```

Package-level docs also live next to the code:

- [`torch_nntile/README.md`](../torch_nntile/README.md) — Python package README

## Hardware note

NNTile targets CUDA devices with compute capability **8.0 or higher**.
