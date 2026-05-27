NNTile
======

![Build and test.](https://github.com/nntile/nntile/actions/workflows/build-test.yml/badge.svg)
![Nightly building.](https://github.com/nntile/nntile/actions/workflows/on-schedule.yml/badge.svg)

## General purpose

**NNTile** is a framework for training large neural networks. It relies on a
task-based parallel programming paradigm that distributes computations across
available hardware resources dynamically and moves data asynchronously, using
the [StarPU](https://starpu.gitlabpages.inria.fr) runtime.

## Documentation

Full documentation lives under **[docs/](docs/README.md)**. Start at
[docs/README.md](docs/README.md) for the complete map.

| Topic | Guide |
|-------|--------|
| Build, Docker, CMake, testing | [docs/build/README.md](docs/build/README.md) |
| C++ stack (kernel → tensor) | [docs/cpp/README.md](docs/cpp/README.md) |
| Python API | [docs/python/README.md](docs/python/README.md) |
| Training scripts and notebooks | [docs/python/training.md](docs/python/training.md) |
| Data preparation | [docs/python/data-preparation.md](docs/python/data-preparation.md) |
| Inference, HTTP gateway, Telegram bot | [docs/inference/README.md](docs/inference/README.md) |
| Gateway + bot deployment | [infra/README.md](infra/README.md) |
| SGOC scheduler (limited VRAM) | [docs/sgoc/README.md](docs/sgoc/README.md) |
| Graph API (work in progress) | [docs/graph-wip.md](docs/graph-wip.md) |

**Quick path:** build the image (`docker build . -t nntile:latest`), then follow
[docs/build/README.md](docs/build/README.md) and the GPT-2 walkthrough in
[docs/python/training.md](docs/python/training.md) (`gpt2_custom_training.py`).

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
