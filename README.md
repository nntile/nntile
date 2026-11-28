<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file README.md
 @version 2.0.0
-->

NNTile
======

![NNHaul backend](https://github.com/nntile/nntile/actions/workflows/nnhaul.yml/badge.svg)

## General purpose

**NNTile** is a framework for training large neural networks. It relies on a
task-based parallel programming paradigm that distributes computations across
available hardware resources dynamically and moves data asynchronously.
NNTile 2.0 runs on NNHaul, single-GPU CUDA, with chunk-addressable kernels.

## Install (Python product)

Plain CMake defaults to building **libnntile**, **libtorch_nntile**, and the
installable **torch_nntile** pip wheel (needs PyTorch on `CMAKE_PREFIX_PATH`).
Build the wheel: [torch_nntile/README.md](torch_nntile/README.md).

```bash
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -GNinja -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build build -j$(nproc)
# → build/wheelhouse/torch_nntile-*.whl  (linux_x86_64; no bundled libs)
pip install --force-reinstall build/wheelhouse/torch_nntile-*.whl
# The unrepaired local wheel links the build tree: run Python with
export LD_LIBRARY_PATH="$PWD/build/torch_nntile:$PWD/build/nntile:$PWD/build/nnhaul:${LD_LIBRARY_PATH:-}"
```

Local CMake leaves `TORCH_NNTILE_WHEEL_REPAIR=OFF`, so the wheel keeps a plain
`linux_x86_64` tag and does not bundle `libtorch_nntile`, `libnntile` or
`libnnhaul`; it is loadable only with the build-tree libraries on the loader
path (or with the libraries installed system-wide). Pass
`-DTORCH_NNTILE_WHEEL_REPAIR=ON` to bundle and patch them (needs
`auditwheel`; the tag then depends on the host glibc). The wheel workflow
repairs inside the manylinux_2_28 image (`manylinux_2_28_x86_64`).

NNTile 2.0 is CUDA-only and NNHaul-only (see
[docs/backends.md](docs/backends.md)); the CUDA wheel is the NNHaul
product (`torch_nntile.backend() == "nnhaul"`). There is no CPU or macOS
wheel.

Libnntile-only (no LibTorch), as in layered CI:

```bash
cmake -S . -B build -GNinja \
  -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF
```

The wheel workflow uses cibuildwheel /
[`torch_nntile/tools/build_wheel_deps.sh`](torch_nntile/tools/build_wheel_deps.sh).

## Example usage

1. Training and fine-tuning transformer language models on a single CUDA
   device, with chunked tile storage that keeps large activations within a
   bounded GPU arena.
2. Running PyTorch autograd models through `torch_nntile.nn` on
   `device="nntile"`, with NNTile's classic kernels behind the PrivateUse1
   dispatcher.
3. Deferred graph capture: `compile_graph()` / `run()` / `wait()` let a
   training phase be recorded once and replayed with pipelined GPU work.

## Documentation

Full documentation lives under **[docs/](docs/README.md)**. Start at
[docs/README.md](docs/README.md) for the complete map.

| Topic | Guide |
|-------|--------|
| torch_nntile (main: `device=nntile`, PyTorch autograd) | [docs/torch_nntile.md](docs/torch_nntile.md) |
| TensorGraph execution backend | [docs/graph.md](docs/graph.md) |
| Build, Docker, CMake, testing | [docs/build/README.md](docs/build/README.md) |
| C++ stack (kernel → TensorGraph → Runtime) | [docs/cpp/README.md](docs/cpp/README.md) |

CUDA 12.x and compute capability **7.0+** are required ([docs/README.md](docs/README.md)).

## Authors

**NNTile** is developed by specialists from

- Skolkovo Institute of Science and Technology ([Skoltech](https://skoltech.ru))
- Artificial Intelligence Research Institute ([AIRI](https://airi.net))

Main contributors are:

- Aleksandr Mikhalev
- Aleksandr Katrutsa
- Konstantin Sozykin
- Gleb Karpov
- Daniel Bershatsky
- Danil Sivtsov
- Ekaterina Lisovskaya

## Acknowledgement

Authors of **NNTile** would like to thank Ivan Oseledets for bringing the idea
of this project to life.

The work was generously supported by the Center in the field of Artificial
Intelligence in the direction of optimizing management decisions to reduce the
carbon footprint on the basis of the Skolkovo Institute of Science and
Technology under Contract No. 70-2021-00145/10841 dated 02.11.2021 (items
2.3.1, 2.3.3, 3.3.2 and 3.3.4) and Contract No. 10825/3978620 dated 26.08.2021.

This work was supported by FASIE (fasie.ru).
