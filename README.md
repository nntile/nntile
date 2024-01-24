NNTile
======

## General purpose

**NNTile** is a framework for training large neural networks. It relies
on a task-based parallel programming paradigm, which distributes computations
across all avialable hardware resources dynamically and transmits data
asynchronously. For this purpose **NNTile** utilizes
[StarPU](https://starpu.gitlabpages.inria.fr) library.

## Authors

**NNTile** is developed by specialists from Skolkovo Institute of Science and
TEchnology [Skoltech](https://skoltech.ru) and Artifical Intelligence Research
Institute [AIRI](https://airi.net)

Main contributors are:
Aleksandr Mikhalev
Aleksandr Katrutsa
Konstantin Sozykin
Gleb Karpov
Daniel Bershatsky

## Acknowledgement

Authors of the **NNTile** would like to thank Ivan Oseledets for bringing idea
of this project to life.
The work was generously supported by the Center in the field of Artificial
Intelligence in the direction of optimizing management decisions to reduce the
carbon footprint on the basis of the Skolkovo Institute of Science and
Technology under Contract No. 70-2021-00145/10841 dated 02.11.2021 (items
2.3.1, 2.3.3, 3.3.2 and 3.3.4) and Contract No. 10825/3978620 dated 26.08.2021.

## Assembly

In order to prepare a development environment one is expected to have a Docker
or Podman container manager. With the command below one can build an image with
all dependencies required for development and testing.

```shell
docker build . \
    -f ci/Dockerfile \
    -t nntile:latest \
    --build-arg MAKE_JOBS=4 \
    --build-arg BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu22.04 \
    --build-arg CUDA_ARCHS=80;86
```

During image building `StarPU` is compiled with `make`. This process can be
adjusted with degree of parallelism with `MAKE_JOBS` option (default no
parallelism). Due to Nvidia pruning their old docker images, it could be
possible that a default `nvidia/cuda:12.2.0-devel-ubuntu-22.04` is not
available. In such a case, input name of an appropriate available image.
Argument `CUDA_ARCHS` defines target CUDA architectures to be supported by
**NNTile**.

