<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file CONTRIBUTING.md
 @version 2.0.0
-->

# Contributing to NNTile

NNTile 2.0 is CUDA-only and NNHaul-only. Contributions must keep that
contract: no alternate runtime, no CPU backend, no torch-native/ATen-on-nntile
family, and chunked kernels only.

## Build and test

```bash
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake --preset nnhaul -DCMAKE_PREFIX_PATH="$TORCH_PREFIX"
cmake --build --preset nnhaul -j

NNHAUL_CHUNK_SIZE_MB=1 sh tools/run_chunked_suite.sh build-nnhaul
NNHAUL_CHUNK_SIZE_MB=2 sh tools/run_chunked_suite.sh build-nnhaul
```

There is no CUDA switch; configure always enables CUDA.

## Python product

`torch_nntile` exposes `torch_nntile.nn` only. Stock `torch.nn` / HuggingFace
models on `device="nntile"` are out of scope for 2.0.

```bash
export LD_LIBRARY_PATH="$PWD/build-nnhaul/nntile:$PWD/build-nnhaul/nnhaul:${LD_LIBRARY_PATH:-}"
python3 -m pip install build-nnhaul/wheelhouse/torch_nntile-*.whl
python3 torch_nntile/tools/smoke_test_wheel.py
```

## Style

Follow `STYLE_GUIDE.md` and `.clang-format`.

## Compatibility

NNTile 1.x (CPU, macOS, stock-torch-on-nntile) remains at the `1.1.0`
tag. Changes that require those products belong there.
