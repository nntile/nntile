# torch_nntile

PyTorch **PrivateUse1** device registered as `device="nntile"`.

## Prebuilt wheels (0.0.1)

Wheels are built in CI, not published to PyPI. Install from a downloaded
`.whl` file after installing the matching `torch` build.

### CI workflow

| | |
|-|-|
| **Workflow** (Actions sidebar / run title) | `torch_nntile wheels` |
| **Workflow file** | `.github/workflows/torch-nntile-wheels.yml` |
| **Trigger** | Pull requests to `graph_api`, or manual **Run workflow** |
| **Python** | 3.12 (`cp312`) |

Wheels build on every **open** PR to `graph_api` (push/sync/reopen), when a PR is
**merged**, or when a maintainer starts the workflow manually
(`workflow_dispatch`). Closed PRs that were not merged are skipped.

### Triggering a build

**Automatic:** open or update a PR targeting `graph_api` (or merge it).

**Manual:** from a machine with write access to the repo:

```bash
gh workflow run torch-nntile-wheels.yml --ref graph_api
gh run watch
```

In the GitHub UI, **Run workflow** appears only when the workflow file with
`workflow_dispatch` exists on the repository **default branch** (see
[GitHub docs](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows#workflow_dispatch)).
Use `gh workflow run` if the button is missing.

Each matrix job uploads a **separate** artifact — there is no single bundle
with all platforms:

| Job | Artifact name |
|-----|---------------|
| Linux CUDA x86_64 | `torch-nntile-wheel-cp312-manylinux_x86_64` |
| macOS arm64 CPU | `torch-nntile-wheel-cp312-macosx_arm64` |

**Download (GitHub UI):** Actions → **torch_nntile wheels** → pick a run →
**Artifacts** at the bottom of the run page.

**Download (`gh` CLI):**

```bash
gh run list --workflow=torch-nntile-wheels.yml --limit 5
gh run download RUN_ID -D wheelhouse
# → wheelhouse/torch-nntile-wheel-cp312-manylinux_x86_64/*.whl
# → wheelhouse/torch-nntile-wheel-cp312-macosx_arm64/*.whl
```

### Linux (CUDA, torch 2.9.1)

Linux CUDA wheels are built against `torch==2.9.1`. **PyTorch may be CPU-only**
from default PyPI; a CUDA build of PyTorch is not required. NVIDIA math
libraries come from `nvidia-*-cu12` pip packages (declared as `torch_nntile`
dependencies on Linux x86_64), not from the wheel itself.

```bash
pip install torch==2.9.1
pip install /path/to/torch_nntile-0.0.1-cp312-cp312-manylinux_2_28_x86_64.whl
```

`pip install` of the wheel pulls the NVIDIA packages on Linux automatically.
You can also install them manually:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cusparse-cu12 \
    nvidia-cusolver-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-runtime-cu12
```

The wheel bundles `libstarpu` (CUDA-enabled, up to 8 devices, no FXT tracing),
`libnntile`, and small transitive deps (OpenBLAS, hwloc). A compatible NVIDIA
**driver** is required at runtime for CUDA StarPU workers (`ncuda > 0`).

### macOS arm64 (CPU-only, torch 2.9.1)

```bash
pip install torch==2.9.1
pip install /path/to/torch_nntile-0.0.1-cp312-cp312-macosx_14_0_arm64.whl
```

StarPU runs on CPU workers only (`ncuda=0`). macOS 14.0+ (arm64).

Publishing to PyPI is manual: download CI artifacts and run `twine upload` locally.
See [docs/build/README.md](../docs/build/README.md) for maintainer CI details.

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
| `torch.cat` | `tensor::concat` |
| `torch.cat` backward | `tensor::copy_intersection` (via `aten::narrow`) |
| `torch.split` / `torch.chunk` | `tensor::copy_intersection` |
| `torch.split` backward | `tensor::concat` (PyTorch `SplitWithSizesBackward`) |
| `F.linear` / `nn.Linear` (no bias) | `tensor::gemm` |
| `F.relu` / `nn.ReLU` | `tensor::relu` |
| ReLU backward | `tensor::relu_backward` (+ `tensor::clear` on output) |
| `F.layer_norm` / `nn.LayerNorm` | `native_layer_norm` / `native_layer_norm_backward` |
| `F.rms_norm` / `nn.RMSNorm` | custom autograd + `rms_norm_forward` / `rms_norm_backward` |
| `F.silu` / `nn.SiLU` | `tensor::silu` |
| SiLU in-place (`silu_`) | `tensor::silu_inplace` |
| SiLU backward | `tensor::silu_backward` (+ `tensor::clear` on output) |
| `F.gelu` / `nn.GELU` (`approximate='none'`) | `tensor::gelu` |
| `F.gelu` (`approximate='tanh'`) | `tensor::gelutanh` |
| GELU in-place (`gelu_`) | `tensor::gelu_inplace` / `tensor::gelutanh_inplace` |
| GELU backward | `tensor::gelu_backward` or `tensor::gelutanh_backward` |
| `F.softmax` / `nn.Softmax` | `tensor::maxsumexp` + `tensor::softmax` |
| Softmax backward | `tensor::sumprod_slice`, `tensor::add_slice`, `tensor::multiply_inplace` |
| `linear` backward / `mm` | `tensor::gemm` |
| `F.embedding` / `nn.Embedding` | `tensor::embedding` |
| Embedding backward | `tensor::embedding_backward` |
| `torch_nntile.training.cross_entropy` | `maxsumexp`, `logsumexp`, `total_sum_accum`, `softmax`, `subtract_indexed_outputs`; backward: chained `scale_slice`, `multiply_slice` |
| `torch_nntile.training.SGD` | `tensor::sgd_step` (fused SGD with momentum) |

PyTorch C-order shapes are converted to TensorGraph storage layout internally.
Gradients use **PyTorch autograd** (not `NNGraph` autograd).

**Embedding v1 limits:** `float32` weights only; `padding_idx` must be `-1`
(default); `scale_grad_by_freq=False` and `sparse=False` only. Indices may stay
on CPU while weights are on `device="nntile"`.

### Gradient accumulation

`torch_nntile` does **not** implement NNGraph-style `get_or_create_grad` /
`add_inplace` fan-in. PyTorch's autograd engine owns all gradient accumulation:

| Mechanism | When | torch_nntile op |
|-----------|------|-----------------|
| `AccumulateGrad` | Leaf `.grad` (params, inputs with `requires_grad=True`) | `add_.Tensor` (in-place `+=` on subsequent grads) or buffer steal on first grad |
| `InputBuffer` | Fan-in on intermediate tensors (diamond graphs) | `add_.Tensor` or `add.Tensor` |
| Optimizer / SGD | `param.add_(grad, alpha=-lr)`, `velocity.add_(grad)` | `add_.Tensor` |

Backward ATen ops (`linear_backward`, `silu_backward`, …) always **overwrite**
fresh grad buffers (`beta=0`). Accumulation is delegated to PyTorch; do not fold
`beta=1` into backward kernels unless profiling proves a fusion win.

**Grad buffer stealing:** backward return tensors must not be pinned for graph
recording (`pin_graph_op_output(..., false)`), so PyTorch can move the first
grad into `param.grad` without an extra copy.

**Training microbatches:** use the standard PyTorch pattern — scale loss, call
`loss.backward()` multiple times, then `optimizer.step()`. No special
`torch_nntile` API. Prefer `optimizer.zero_grad(set_to_none=True)` so the
first backward can steal into `.grad`; `grad.zero_()` is supported via
`zero_` / `fill_(0)` when `set_to_none=False`.

PyTorch does not fuse accumulation across the backward graph without
`torch.compile`; each `+=` dispatches to `add_` as a separate kernel.

Tests: `pytest -vv torch_nntile/tests/test_grad_accumulation.py`

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
``name_mnist_axis_groups`` (batch, features, classes, and ``hidden`` on each
linear weight/grad/velocity) and passes it to ``train_full_batch_step``.

CLI: ``--axis-tiling NAME=SIZES`` (repeatable), ``--print-axis-groups``,
``--restrict-cuda``, ``--verbose``.

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

Axis naming (`batch`, `features`, `hidden`, `classes`) is in the example script — see
[docs/torch_nntile.md](../docs/torch_nntile.md) for full run instructions and
expected output.

```bash
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib

# CPU StarPU workers (nntile path); reference PyTorch path is always CPU
STARPU_NCPU=4 STARPU_NCUDA=0 \
  python torch_nntile/examples/train_deep_relu_mnist.py \
    --runtime-mode graph --epochs 5

# CUDA StarPU workers only
STARPU_NCPU=0 STARPU_NCUDA=2 \
  python torch_nntile/examples/train_deep_relu_mnist.py \
    --runtime-mode graph --restrict-cuda --epochs 5 \
    --axis-tiling batch=15000,15000,15000,15000 \
    --axis-tiling features=392,392 \
    --axis-tiling hidden=128,128
```

**Parity expectations:** with CPU workers, per-epoch loss diffs are ~1e-6 or
smaller. With CUDA workers, **loss** diffs of ~1e-4 are acceptable; **weights**
should still agree to ~1e-8. See [docs/torch_nntile.md](../docs/torch_nntile.md)
for sample output.

Integration test (downloads MNIST, 3 epochs, compares losses and weights):

```bash
pytest -vv -m slow torch_nntile/tests/test_deep_relu_mnist_train.py
```

Cross-entropy parity (forward, backward, multi-D labels, `ignore_index`):

```bash
pytest -vv torch_nntile/tests/test_cross_entropy_parity.py
```

## Install from source (stub only)

Install `torch==2.9.1` first (same ABI as `install_requires`), then:

```bash
pip install 'torch==2.9.1'
CXX=g++ pip install -e ./torch_nntile --no-build-isolation
```

## Install from source (with libnntile / phase 2)

Build NNTile first (CPU-only example):

```bash
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -GNinja
cmake --build build -j$(nproc)
```

Then install the extension against that build (use the same `torch` version you
built NNTile against):

```bash
pip install 'torch==2.9.1'
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

When CUDA workers are enabled (`STARPU_NCUDA > 0`), use ``--restrict-cuda`` in
the MNIST example (or call ``restrict_cuda()``) and shut StarPU down at exit.
The example calls ``torch_nntile.wait()`` and ``torch_nntile.shutdown_context()``
in a ``finally`` block; ``init_context()`` also registers an ``atexit`` hook.

## macOS / PyTorch cpu_fallback ABI

PyTorch 2.12+ exports `at::native::cpu_fallback` with four arguments
(`OperatorHandle`, `Stack*`, `bool error_on_views`, `c10::DispatchKey`).
Older releases use a two-argument overload. The extension selects the
appropriate overload at compile time via `TORCH_VERSION_*`.

After upgrading PyTorch, reinstall the matching torch pin and rebuild:

```bash
pip install 'torch==2.9.1'
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
