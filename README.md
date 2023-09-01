NNTile
======

**NNTile** is a software for training big neural networks over heterogeneous distributed-memory systems. The approach is based on divide-and-conquer principle: all data are stored in small chunks, so-called tiles, and each layer is applied directly to the tiles. This way each layer, as well as entire neural network, is represented by a directed acyclic graph (DAG for short) of tasks operating on separate tiles. Execution of tasks is scheduled dynamically with help of StarPU library. The StarPU library is meant for distributed heterogeneous computing and allows using different processing units on the same node.

## Assembly

In order to prepare a development environment one is expected to have a Docker
or Podman container manager. With the command below one can build an image with
all dependencies required for development and testing.

```shell
docker build . \
    -f ci/Dockerfile \
    -t doge.skoltech.ru/green-ai/nntile/sandbox:cuda \
    --build-arg MAKE_JOBS=4 \
    --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04 \
    --target sandbox
```

During image building `starpu` is compiled with `make`. This process can be
adjusted with degree of parallelism with `MAKE_JOBS` option (default no
parallelism). As soon as a development image is built, one can start a
container and build `nntile` as follows.

```shell
cmake -B build -S . -G 'Ninja Multi-Config' \
    -DCMAKE_DEFAULT_BUILD_TYPE=RelWithDebInfo \
    -DUSE_CUDA=ON
cmake --build build
```

Finally, tests can be run with the following command.

```shell
ctest --test-dir build -C RelWithDebInfo
```
