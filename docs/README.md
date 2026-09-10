# NNTile documentation

**NNTile** is a framework for training large neural networks on distributed-memory
heterogeneous systems. It uses [StarPU](https://starpu.gitlabpages.inria.fr) for
task-based scheduling and asynchronous data movement across CPU and GPU resources.

The product entry point on the `torch_nntile` branch is **torch_nntile**:
PyTorch `device="nntile"` backed by **PyTorch autograd**. Ops record into a
deferred TensorGraph, lower to tiles, and run through StarPU. There is no
separate NNTile autograd.

Two Python surfaces:

- Stock `torch.nn` / `torch.nn.functional` on `device=nntile` — torch-provided
  kernels (untiled ATen StarPU codelets).
- `torch_nntile.nn` — NNTile kernels (tiling allowed):
  `functional` (autograd functions), `module` (`nn.Module` subclasses),
  `model` (full models built from those modules).

## Quick start

| Goal | Start here |
|------|------------|
| **torch_nntile (main)** | [torch_nntile.md](torch_nntile.md) |
| Install prebuilt `torch_nntile` wheel (CI) | [torch_nntile.md#prebuilt-wheels](torch_nntile.md#prebuilt-wheels) |
| Build from source or Docker | [build/README.md](build/README.md) |
| TensorGraph execution backend | [graph.md](graph.md) |
| C++ stack (kernel → TensorGraph → Runtime) | [cpp/README.md](cpp/README.md) |
| Design notes (`docs/dev`) | [dev/README.md](dev/README.md) |
| Inference, gateway, Telegram bot | [inference/README.md](inference/README.md) |
| SGOC scheduler (limited VRAM, single GPU) | [sgoc/README.md](sgoc/README.md) |

## Documentation map

```
docs/
  README.md                 ← you are here
  torch_nntile.md           PyTorch device="nntile" (main entry)
  graph.md                  TensorGraph → TileGraph → Runtime backend
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
