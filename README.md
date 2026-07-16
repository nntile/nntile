NNTile
======

![Build and test.](https://github.com/nntile/nntile/actions/workflows/build-test.yml/badge.svg)
![Nightly building.](https://github.com/nntile/nntile/actions/workflows/on-schedule.yml/badge.svg)

## General purpose

**NNTile** is a framework for training large neural networks. It relies on a
task-based parallel programming paradigm that distributes computations across
available hardware resources dynamically and moves data asynchronously, using
the [StarPU](https://starpu.gitlabpages.inria.fr) runtime.

## Install (Python product)

Plain CMake defaults to building **libnntile**, **libtorch_nntile**, and the
installable **torch_nntile** pip wheel (needs PyTorch on `CMAKE_PREFIX_PATH`).
Prebuilt wheels: [torch_nntile/README.md](torch_nntile/README.md).

```bash
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -GNinja -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build build -j$(nproc)
# → build/wheelhouse/torch_nntile-*.whl  (linux_x86_64; no auditwheel by default)
pip install build/wheelhouse/torch_nntile-*.whl
```

Local CMake leaves `TORCH_NNTILE_WHEEL_REPAIR=OFF`, so the wheel keeps a plain
`linux_x86_64` tag. Host `auditwheel` on Ubuntu 24.04 would otherwise emit
`manylinux_2_39_x86_64`, which many pip installs reject. Release CI still
repairs inside the manylinux_2_28 image (`manylinux_2_28_x86_64`).

Libnntile-only (no LibTorch), as in layered CI:

```bash
cmake -S . -B build -GNinja \
  -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF
```

Release CI also uses cibuildwheel /
[`torch_nntile/tools/build_wheel_deps.sh`](torch_nntile/tools/build_wheel_deps.sh).

## Documentation

Full documentation lives under **[docs/](docs/README.md)**. Start at
[docs/README.md](docs/README.md) for the complete map.

| Topic | Guide |
|-------|--------|
| Graph API (main, WIP) | [docs/graph.md](docs/graph.md) |
| torch_nntile (Python / LibTorch) | [torch_nntile/README.md](torch_nntile/README.md) |
| Build, Docker, CMake, testing | [docs/build/README.md](docs/build/README.md) |
| C++ stack (kernel → TensorGraph → Runtime) | [docs/cpp/README.md](docs/cpp/README.md) |
| Inference, HTTP gateway, Telegram bot | [docs/inference/README.md](docs/inference/README.md) |
| Gateway + bot deployment | [infra/README.md](infra/README.md) |
| SGOC scheduler (limited VRAM) | [docs/sgoc/README.md](docs/sgoc/README.md) |

CUDA compute capability **8.0+** is required ([docs/README.md](docs/README.md)).

## Preliminary experimental results

Experiments with custom 4-layer and 8-layer GPT models of sizes up to 50B
showed both good performance and a possibility to train 4 times larger models
compared to PyTorch FSDP on the same hardware (a single server with 8 x
Nvidia A100 80GB SXM).

![Custom 4-layer model on 4 GPUs](images/gpt_short_perf_4gpu.png)
![Custom 4-layer model on 8 GPUs](images/gpt_short8_perf.png)

The same figures in better quality:

- [Custom 4-layer model on 4 GPUs](images/gpt_short_perf_4gpu.pdf)
- [Custom 4-layer model on 8 GPUs](images/gpt_short_perf_8gpu.pdf)
- [Custom 8-layer model on 8 GPUs](images/gpt_short8_perf.pdf)

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
