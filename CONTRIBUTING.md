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

`USE_CUDA=OFF` is a configure error by design.

## Python product

`torch_nntile` exposes `torch_nntile.nn` only. Stock `torch.nn` / HuggingFace
models on `device="nntile"` are out of scope for 2.0.

```bash
python3 -m pip install build-nnhaul/wheelhouse/torch_nntile-*.whl
python3 torch_nntile/tools/smoke_test_wheel.py
```

## Style

Follow `STYLE_GUIDE.md` and `.clang-format`.

## Compatibility

NNTile 1.x (CPU, macOS, stock-torch-on-nntile) remains on the `1.x`
branch. Changes that require those products belong there.
