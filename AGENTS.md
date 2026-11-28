# NNTile Development Guide

NNTile 2.0 is a CUDA-only framework with one runtime: **NNHaul**.
Every tile is chunk-backed; there is no CPU backend, and no
torch-native/ATen-on-nntile path.

## Environment

- CUDA toolkit with `nvcc`, cuBLAS, cuDNN.
- PyTorch matching `torch_nntile` (2.9.x), CMake >= 3.24, Ninja, g++.

## Build

```bash
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake --preset nnhaul -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build --preset nnhaul -j
```

Libnntile only:

```bash
cmake -S . -B build -GNinja \
  -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF
cmake --build build -j
```

There is no CUDA switch: the build always enables CUDA.

## Tests

```bash
cmake -S . -B build -GNinja -DBUILD_TESTING=ON \
  -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF
cmake --build build -j
NNHAUL_CHUNK_SIZE_MB=1 sh tools/run_chunked_suite.sh build
NNHAUL_CHUNK_SIZE_MB=2 sh tools/run_chunked_suite.sh build
```

The engine tests run at the default class; the nntile group runs at the
requested class with the documented memory caps.

## Layout

- `nntile/src/kernel` — raw CPU/CUDA kernels, including chunked entry points
- `nntile/src/nnhaul` — NNHaul codelets and chunk-pointer transport
- `nntile/src/{core,tile,tensor}` and `nntile/src/runtime.cc` — graph stack on NNHaul
- `nnhaul/` — vendored runtime/engine + planner API
- `torch_nntile/` — PrivateUse1 backend, classic `torch_nntile.nn` only

## Rules

- No alternate runtime option and no torch-native code path.
- CUDA and chunked-only; one GPU per process (`ncuda=1`).
- Stock `torch.nn` / HF on `device="nntile"` is out of scope for 2.0;
  use `torch_nntile.nn`.
