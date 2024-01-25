NNTile
======

## General purpose

**NNTile** is a framework for training large neural networks. It relies
on a task-based parallel programming paradigm, which distributes computations
across all avialable hardware resources dynamically and transmits data
asynchronously. For this purpose **NNTile** utilizes
[StarPU](https://starpu.gitlabpages.inria.fr) library.

## Preliminary experimental results

Experiments with custom 4-layer and 8-layer GPT models of sizes up to 50B
showed both good performance and a possibility to train 4 times larger models
compared to PyTorch FSDP on the same hardware (a single server with 8 x
Nvidia A100 80GB SXM).

![Custom 4-layer model on 4 GPUs](images/gpt_short_perf_4gpu.png)
![Custom 4-layer model on 4 GPUs](images/gpt_short8_perf.png)

The same figures in better quality:
 * [Custom 4-layer model on 4 GPUs](images/gpt_short_perf_4gpu.pdf)
 * [Custom 4-layer model on 8 GPUs](images/gpt_short_perf_8gpu.pdf)
 * [Custom 8-layer model on 8 GPUs](images/gpt_short8_perf.pdf)

## Authors

**NNTile** is developed by specialists from
 * Skolkovo Institute of Science and Technology ([Skoltech](https://skoltech.ru))
 * Artifical Intelligence Research Institute ([AIRI](https://airi.net))

Main contributors are:
 * Aleksandr Mikhalev
 * Aleksandr Katrutsa
 * Konstantin Sozykin
 * Gleb Karpov
 * Daniel Bershatsky

## Acknowledgement

Authors of the **NNTile** would like to thank Ivan Oseledets for bringing idea
of this project to life.
The work was generously supported by the Center in the field of Artificial
Intelligence in the direction of optimizing management decisions to reduce the
carbon footprint on the basis of the Skolkovo Institute of Science and
Technology under Contract No. 70-2021-00145/10841 dated 02.11.2021 (items
2.3.1, 2.3.3, 3.3.2 and 3.3.4) and Contract No. 10825/3978620 dated 26.08.2021.

## Assembly

**NNTile** comes with a `ci/Dockerfile` to construct docker image with NNTile
and all prerequisites. Ready image can be acquired from the GitHub container
registry:
```shell
docker pull ghcr.io/skolai/nntile:1.0.0-starpu1.3.11-cuda12.2.0-ubuntu22.04
```

Alternatively, the docker image can be built on your own system with the following
command:
```shell
docker build . \
    -f ci/Dockerfile \
    -t nntile:latest \
    --build-arg MAKE_JOBS=4 \
    --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04 \
    --build-arg CUDA_ARCHS="80;86;90"
```

During image building `StarPU` is compiled with `make`. This process can be
adjusted with degree of parallelism with `MAKE_JOBS` option (default no
parallelism). Due to Nvidia pruning their old docker images, it could be
possible that a default `nvidia/cuda:12.2.0-devel-ubuntu-22.04` is not
available. In such a case, input name of an appropriate available image.
Argument `CUDA_ARCHS` defines target CUDA architectures to be supported by
**NNTile**.

## Minimal working GPT example

To make **NNTile** train your custom GPT model there is a minimal working example
[gpt2_custom_training.py](./wrappers/python/examples/gpt2_custom_training.py).
It works either with a
WikiText-103 datasets or with a dataset stored in a train.bin format that contains
a stream of uint16 values just like [NanoGPT](https://github.com/karpathy/nanogpt)
does it with a help of its special script
[prepare.py](https://github.com/karpathy/nanogpt/data/openwebtext/prepare.py)
for the OpenWebText.
