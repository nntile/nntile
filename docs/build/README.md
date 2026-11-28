<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file docs/build/README.md
 @version 2.0.0
-->

# Building NNTile 2.0

NNTile 2.0 is CUDA-only, NNHaul-only, and chunked-only. There is no CPU
build and no macOS product.

## Requirements

- CMake >= 3.24, Ninja, a C++17 compiler
- CUDA toolkit with `nvcc` and cuBLAS
- LibTorch (PyTorch) when building `torch_nntile`
- Python 3.12 for the wheel

## Product build

```bash
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake --preset nnhaul -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build --preset nnhaul -j
```

This produces `libnnhaul.so`, `libnntile.so`, `libtorch_nntile.so`, and a
local `torch_nntile-2.0.0` wheel under `build-nnhaul/wheelhouse`.

Configure always enables CUDA; a missing toolkit is a hard error.

## Libnntile only

```bash
cmake -S . -B build -GNinja \
  -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF
cmake --build build -j
```

## Tests

```bash
cmake -S . -B build -GNinja -DBUILD_TESTING=ON \
  -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF
cmake --build build -j
NNHAUL_CHUNK_SIZE_MB=1 sh tools/run_chunked_suite.sh build
NNHAUL_CHUNK_SIZE_MB=2 sh tools/run_chunked_suite.sh build
```

Engine tests run at the default chunk class; the nntile group runs at the
requested class with the documented memory caps.

## Tests-only build against an installed prefix

```bash
cmake -S . -B build-tests -GNinja -DBUILD_TESTING=ON \
  -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF \
  -DCMAKE_PREFIX_PATH=/path/to/nntile/install
cmake --build build-tests -j
ctest --test-dir build-tests -LE NotImplemented
```

The test tree adopts the installed CUDA and disabled-kernel scope.

## Wheel

`cmake --build build-nnhaul` builds a local wheel. By default it is not
self-contained: keep the build-tree libraries on `LD_LIBRARY_PATH`, or build
with `-DTORCH_NNTILE_WHEEL_REPAIR=ON` to bundle them (requires `auditwheel`).

## Docker

```bash
docker build -t nntile:2.0 .
docker run --gpus all -it --rm nntile:2.0
```

The image configures a CUDA-only, NNHaul-only build (`BUILD_LIBTORCH_NNTILE=OFF`,
`BUILD_TORCH_NNTILE=OFF`) and leaves the libraries under
`/workspace/nntile/build`.
