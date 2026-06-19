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
| `torch_nntile.training.cross_entropy` | `maxsumexp`, `logsumexp`, `total_sum_accum`, `softmax`, `subtract_indexed_outputs`; backward: chained `scale_slice`, `multiply_slice` |
| `torch_nntile.training.SGD` | `tensor::sgd_step` (fused SGD with momentum) |

PyTorch C-order shapes are converted to TensorGraph storage layout internally.
Gradients use **PyTorch autograd** (not `NNGraph` autograd).

### CPU fallback control

```python
torch_nntile.init_context(ncpu=1, ncuda=0, cpu_fallback=False)
```

When `cpu_fallback=False`, unsupported ATen ops raise instead of running on CPU.
Use this to verify that a model forward uses only nntile kernels.

### Runtime mode: eager vs graph

```python
# Default: each op records a TensorGraph slice and runs it immediately.
torch_nntile.init_context(ncpu=1, ncuda=0, cpu_fallback=False)

# Deferred: ops append to one shared TensorGraph until you flush.
torch_nntile.init_context(
    ncpu=1, ncuda=0, cpu_fallback=False, runtime_mode="graph"
)
y = model(x)              # recorded, not executed yet
loss.backward()           # backward ops recorded too
torch_nntile.compile_graph()
torch_nntile.run()
z = y.to("cpu")           # host readout after run
```

In graph mode, forward and backward can stay in one pending graph (StarPU
resolves dependencies). Call ``torch_nntile.compile_graph()`` then
``torch_nntile.run()`` each step. Host reads from **nntile** tensors use
``.to("cpu")`` or ``.cpu()`` after ``run()`` (data is synced from tile memory).
Copies **to** ``device="nntile"`` move host storage into tiles via ``.to()``;
there is no ``bind_data`` in torch_nntile. Training helpers such as
``train_full_batch_step`` call ``compile_graph()`` + ``run()`` in graph mode
and return ``loss.to("cpu").item()``.

Tests: `pytest -vv torch_nntile/tests/test_graph_execution.py`

### Axis-group naming and tiling (graph mode)

Full reference: [docs/torch_nntile.md](../docs/torch_nntile.md).

Tiling is configured on named **axis groups** in the recorded `TensorGraph`
(mirroring the C++ `AxisDescriptor` workflow). Name dimensions from a tensor,
then set tile sizes by group name before ``compile_graph()``.

| API | Purpose |
|-----|---------|
| `set_axis_group_name(tensor, {dim: name})` | Name axis groups (partial dims OK) |
| `set_axis_group_tiling(name, tile_sizes)` | Uniform `int` or heterogeneous `list` |
| `format_axis_groups()` | String summary of pending axis groups |
| `print_axis_groups()` | Print summary (includes `pending_tile=` before compile) |

```python
torch_nntile.init_context(
    ncpu=4, ncuda=0, cpu_fallback=False, runtime_mode="graph"
)
x = torch.randn(4, 128).to("nntile")
torch_nntile.set_axis_group_name(x, {0: "batch", 1: "features"})
logits = model(x)
torch_nntile.set_axis_group_tiling("batch", [1, 1, 2])
torch_nntile.print_axis_groups()
torch_nntile.compile_graph()
torch_nntile.run()
```

Models do **not** assign axis names internally. The MNIST example defines
``name_mnist_axis_groups`` and passes it to ``train_full_batch_step``:

```python
def name_mnist_axis_groups(x, logits):
    torch_nntile.set_axis_group_name(x, {0: "batch", 1: "features"})
    torch_nntile.set_axis_group_name(logits, {1: "classes"})
```

CLI: ``--axis-tiling NAME=SIZES`` (repeatable), ``--print-axis-groups``.

Tests: `pytest -vv torch_nntile/tests/test_axis_group_tiling.py`

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

Cross-entropy is evaluated on nntile via `torch_nntile.training.cross_entropy`
(same tensor-op chain as `NNCrossEntropyOp` in libnntile). Logits use **class
dim last** (`[..., C]`); labels match logits without the class axis (`...`).
The scalar loss lives on ``device="nntile"``; use ``loss.to("cpu")`` after
``compile_graph()`` and ``run()`` in graph mode. Backward keeps ``grad_output`` as a
graph tensor (no host scalar read during recording) and broadcasts it to the
label shape with one ``scale_slice`` per label dimension, then applies
``multiply_slice`` along the class axis. Optimizer steps use fused
``tensor::sgd_step`` via ``torch_nntile.training.SGD`` (no per-parameter CPU
round-trip).

Axis naming (`batch`, `features`, `classes`) is in the example script — see
[docs/torch_nntile.md](../docs/torch_nntile.md).

```bash
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
python torch_nntile/examples/train_deep_relu_mnist.py --epochs 5
python torch_nntile/examples/train_deep_relu_mnist.py --epochs 5 --runtime-mode graph
python torch_nntile/examples/train_deep_relu_mnist.py --epochs 5 \
  --print-axis-groups \
  --axis-tiling batch=15000,15000,15000,15000,15000
```

Integration test (downloads MNIST, 3 epochs, compares losses and weights):

```bash
pytest -vv -m slow torch_nntile/tests/test_deep_relu_mnist_train.py
```

Cross-entropy parity (forward, backward, multi-D labels, `ignore_index`):

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
