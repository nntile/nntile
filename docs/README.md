<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file docs/README.md
 @version 2.0.0
-->

# NNTile documentation

**NNTile** is a framework for training large neural networks on a single
CUDA GPU. NNTile 2.0 runs on **NNHaul** with chunk-addressable kernels.

The product entry point is **torch_nntile**: PyTorch `device="nntile"`
backed by **PyTorch autograd**. Ops record into a deferred TensorGraph,
lower to tiles, and run through NNHaul. There is no separate NNTile autograd.

One Python surface:

- `torch_nntile.nn` — NNTile's classic kernels (tiling allowed):
  `functional` (autograd functions), `module` (`nn.Module` subclasses),
  `model` (full models built from those modules).

Stock `torch.nn` / `torch.nn.functional` on `device="nntile"` and CPU/macOS
products are not part of NNTile 2.0.

## Quick start

| Goal | Start here |
|------|------------|
| **torch_nntile (main)** | [torch_nntile.md](torch_nntile.md) |
| Build the `torch_nntile` wheel | [torch_nntile.md#build](torch_nntile.md#build) |
| Build from source or Docker | [build/README.md](build/README.md) |
| TensorGraph execution backend | [graph.md](graph.md) |
| C++ stack (kernel → TensorGraph → Runtime) | [cpp/README.md](cpp/README.md) |
| Kernel scope (transformers only) | [kernel-scope.md](kernel-scope.md) |
| Unused / reference-only code registry | [unused-code.md](unused-code.md) |

## Documentation map

```
docs/
  README.md                 ← you are here
  torch_nntile.md           PyTorch device="nntile" (main entry)
  graph.md                  TensorGraph → TileGraph → Runtime backend
  build/README.md           Build, CMake, Docker, testing, wheel CI
  cpp/README.md             C++ kernel / NNHaul / tile / tensor / runtime
  backends.md               Backend, device and kernel-scope summary
  chunked-tiles.md          Chunk model, kernels and driver modes
  kernel-scope.md           Disabled kernel families and revival
  unused-code.md            Unused/reference-only code registry
```

Package-level docs also live next to the code:

- [`torch_nntile/README.md`](../torch_nntile/README.md) — Python package README

## Minimal requirements

### Running NNTile alone without comparison against Pytorch requires

- Operating System: Ubuntu 22.04 or above.
- CUDA GPUs: compute capability 7.0 or higher.
- CUDA toolkit: 12.x (the wheel links CUDA 12 sonames).
- CPU: x86-64 or amd64 compatible.
- Python: 3.10 or above.
- C++ compiler with C++17 standard supported.
- CPU RAM: 1 GB.
- CUDA GPU VRAM: 1 GB.
- Hard Drive: 64 GB.

### Testing and comparing against Pytorch requires additionally

- PyTorch 2.9.x.
- Transformers Python package 4.52.*.
- Pytest 8.2 or above.
