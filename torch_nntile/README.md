# torch_nntile

PyTorch **PrivateUse1** device registered as `device="nntile"`.

## Phase 1 (stub)

Tensor storage is backed by a host `std::vector<uint8_t>` buffer. Supports
allocation, `tensor.to("nntile")` / `.cpu()`, and a global CPU fallback for
unsupported ATen ops. Does **not** require `libnntile`.

## Phase 2 (TensorGraph ops)

When built with `NNTILE_BUILD_DIR` pointing at a CMake build tree, selected ops
run through libnntile `TensorGraph` → `TileGraph` → `Runtime`:

| PyTorch op | libnntile |
|------------|-----------|
| `a + b` | `tensor::add` |
| `F.linear` / `nn.Linear` (no bias) | `tensor::gemm` |
| `F.relu` / `nn.ReLU` | `tensor::relu` |
| ReLU backward | `tensor::relu_backward` (+ `tensor::clear` on output) |
| `linear` backward / `mm` | `tensor::gemm` |
| `torch_nntile.training.cross_entropy` | `maxsumexp`, `logsumexp`, `total_sum_accum`, `softmax`, `subtract_indexed_outputs` |

PyTorch C-order shapes are converted to TensorGraph storage layout internally.
Gradients use **PyTorch autograd** (not `NNGraph` autograd).

### CPU fallback control

```python
torch_nntile.init_context(ncpu=1, ncuda=0, cpu_fallback=False)
```

When `cpu_fallback=False`, unsupported ATen ops raise instead of running on CPU.
Use this to verify that a model forward uses only nntile kernels.

## Phase 3 (DeepReLU example)

Bias-free MLP matching `nntile/examples/deep_relu_forward.cc`:

```python
import torch
import torch_nntile
from torch_nntile.models import DeepReLU

torch_nntile.init_context(ncpu=1, ncuda=0, cpu_fallback=False)

model = DeepReLU.tiny().to("nntile")
x = torch.randn(32, 128).to("nntile")
y = model(x)
y.backward(torch.ones(y.shape, device="cpu").to("nntile"))
```

Parity test (forward + backward, nntile vs CPU, no fallback):

```bash
pytest -vv torch_nntile/tests/test_deep_relu_parity.py
```

## Phase 4 (MNIST full-batch training)

Train `DeepReLU.mnist()` on all **60 000** MNIST training images in one batch,
comparing CPU PyTorch vs `device="nntile"` with the same weight initialization.

Cross-entropy runs entirely on nntile via `torch_nntile.training.cross_entropy`
(same tensor-op chain as `NNCrossEntropyOp` in libnntile). The scalar loss is
returned on CPU so PyTorch autograd can call `loss.backward()` without extra
ATen kernels on PrivateUse1. Optimizer steps use manual SGD (no `torch.optim`
on nntile yet).

```bash
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
python torch_nntile/examples/train_deep_relu_mnist.py --epochs 5
```

Integration test (downloads MNIST, 3 epochs, compares losses and weights):

```bash
pytest -vv -m slow torch_nntile/tests/test_deep_relu_mnist_train.py
```

Cross-entropy parity (forward, backward, `ignore_index`):

```bash
pytest -vv torch_nntile/tests/test_cross_entropy_parity.py
```

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

### StarPU worker placement (libnntile)

Pin codelets to CPU or CUDA workers, matching `nntile.Context` in the main
package:

```python
import torch_nntile

torch_nntile.init_context(ncpu=1, ncuda=1, verbose=0)
torch_nntile.restrict_cuda()   # CUDA-only kernels
# ... run nntile-backed ops ...
torch_nntile.restore_where()   # default placement again
```

`init_context()` must be called before the first libnntile-backed operation
(e.g. `a + b` on `device="nntile"`). `restrict_cpu()` / `restrict_cuda()` /
`restore_where()` auto-create the context with defaults if needed.

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
