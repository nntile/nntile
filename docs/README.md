# NNTile documentation

**NNTile** is a framework for training large neural networks on distributed-memory
heterogeneous systems. It uses [StarPU](https://starpu.gitlabpages.inria.fr) for
task-based scheduling and asynchronous data movement across CPU and GPU resources.

This documentation covers how to build NNTile, the C++ implementation stack, the
Python API, inference services (gateway and Telegram bot), the SGOC scheduler,
and end-to-end training examples.

## Quick start

| Goal | Start here |
|------|------------|
| Build from source or Docker | [build/README.md](build/README.md) |
| C++ stack overview (kernel → tensor) | [cpp/README.md](cpp/README.md) |
| Python package and runtime | [python/README.md](python/README.md) |
| Tensor ops reference | [python/functions.md](python/functions.md) |
| Layers and models | [python/layers.md](python/layers.md), [python/models.md](python/models.md) |
| Training scripts and notebooks | [python/training.md](python/training.md) |
| Data preparation | [python/data-preparation.md](python/data-preparation.md) |
| Inference, gateway, Telegram bot | [inference/README.md](inference/README.md) |
| SGOC scheduler (limited VRAM, single GPU) | [sgoc/README.md](sgoc/README.md) |
| NNTile Graph API (work in progress) | [graph-wip.md](graph-wip.md) |

## Documentation map

```
docs/
  README.md                 ← you are here
  build/README.md           Build, CMake, Docker, testing
  cpp/README.md             C++ kernel / starpu / tile / tensor
  graph-wip.md              NNTile Graph API status
  sgoc/README.md            SGOC StarPU scheduler
  inference/README.md       Inference, nntile_gateway, nntile_tgbot
  python/
    README.md               Package overview
    tensors.md              Creation and I/O
    functions.md            Tensor operation wrappers
    layers.md               Layer API
    models.md               Model catalog
    training.md             Pipeline, loss, optimizers, examples
    data-preparation.md     Dataset scripts
```

Internal graph design notes under `docs/dev/` (including
[dev/graph_static_execution_plan.md](dev/graph_static_execution_plan.md)) are
not part of this guide.

## Hardware note

NNTile targets CUDA devices with compute capability **8.0 or higher**.
