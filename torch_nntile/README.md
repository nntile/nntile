# torch_nntile

PyTorch **PrivateUse1** device registered as `device="nntile"`.

## Phase 1 (stub)

Tensor storage is backed by a host `std::vector<uint8_t>` buffer. Supports
allocation, `tensor.to("nntile")` / `.cpu()`, and a global CPU fallback for
unsupported ATen ops. Does **not** require `libnntile`.

## Phase 2 (TensorGraph add)

When built with `NNTILE_BUILD_DIR` pointing at a CMake build tree, `a + b` on
`device="nntile"` runs `nntile::tensor::add` through `TensorGraph` →
`TileGraph` → `Runtime`. PyTorch shapes use C-order labels; the bridge converts
to TensorGraph storage layout internally. Gradients use **PyTorch autograd**
(not `NNGraph` autograd).

## Install (stub only)

```bash
CXX=g++ pip install -e ./torch_nntile --no-build-isolation
```

## Install (with libnntile / phase 2)

Build NNTile first (CPU-only example):

```bash
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -GNinja
cmake --build build -j$(nproc)
```

Then install the extension against that build:

```bash
export NNTILE_BUILD_DIR=$PWD/build
export NNTILE_SOURCE_DIR=$PWD
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CXX=g++ pip install -e ./torch_nntile --no-build-isolation --force-reinstall
```

## Usage

Run Python from **outside** the repo root (or from inside `torch_nntile/`) so
`import torch_nntile` resolves the installed package, not the project folder.

```python
import torch
import torch_nntile  # registers the nntile backend once

x = torch.tensor([1.0, 2.0, 3.0], device="nntile")
y = x.cpu()

a = torch.tensor([1.0, 2.0], device="nntile")
b = torch.tensor([3.0, 4.0], device="nntile")
z = a + b  # TensorGraph add when libnntile is linked
```

## macOS / PyTorch 2.12

PyTorch 2.12 exports `at::native::cpu_fallback` with four arguments
(`OperatorHandle`, `Stack*`, `bool error_on_views`, `c10::DispatchKey`).
Calling it with fewer arguments can leave an unresolved reference to a
three-argument symbol at load time on macOS. The extension calls the
four-argument overload explicitly.

After upgrading PyTorch, rebuild:

```bash
CXX=clang++ pip install -e ./torch_nntile --no-build-isolation --force-reinstall
```

## Tests

```bash
# Stub tests (no libnntile)
pytest -vv torch_nntile/tests/test_device_stub.py

# Full suite (requires libnntile build + LD_LIBRARY_PATH)
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
pytest -vv torch_nntile/tests
```
