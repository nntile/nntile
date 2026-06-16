# torch_nntile

PyTorch **PrivateUse1** device stub registered as `device="nntile"`.

Tensor storage is backed by a host `std::vector<uint8_t>` buffer. Phase 1
supports allocation, `tensor.to("nntile")` / `.cpu()`, and a global CPU
fallback for unsupported ATen ops.

This package does **not** link to `libnntile`.

## Install

```bash
CXX=g++ pip install -e ./torch_nntile --no-build-isolation
```

Requires PyTorch >= 2.1 with C++ extension build tools (g++). Use `CXX=g++` because
the default `c++` on some images may lack libstdc++ headers. Use
`--no-build-isolation` so the extension compiles against the installed `torch`.

## Usage

```python
import torch
import torch_nntile  # registers the nntile backend once

x = torch.tensor([1.0, 2.0, 3.0], device="nntile")
y = x.cpu()
assert torch.allclose(y, torch.tensor([1.0, 2.0, 3.0]))
```

## Tests

```bash
pytest -vv torch_nntile/tests
```
