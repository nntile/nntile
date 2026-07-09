# torch_nntile

PyTorch **PrivateUse1** backend registered as `device="nntile"`. When built
against `libnntile`, selected ops record into a shared `TensorGraph`, lower to
`TileGraph`, and run through `Runtime` (StarPU).

Package README: [`torch_nntile/README.md`](../torch_nntile/README.md).

## Prebuilt wheels

CI builds `torch_nntile` 0.0.2 wheels via the **`torch_nntile wheels`** workflow
(`.github/workflows/torch-nntile-wheels.yml`).

| Trigger | When wheels build |
|---------|-------------------|
| **Pull request → `graph_api`** | On open/update (and on merge close) |
| **`workflow_dispatch`** | Maintainer runs manually (Actions UI or `gh workflow run`) |

Closed PRs that were not merged are skipped.

Each platform is a **separate** artifact (no combined bundle):

| Platform | Artifact |
|----------|----------|
| Linux CUDA x86_64, Python 3.12 | `torch-nntile-wheel-cp312-manylinux_x86_64` |
| macOS arm64 CPU, Python 3.12 | `torch-nntile-wheel-cp312-macosx_arm64` |

Wheels are **not on PyPI**. Download from Actions → **torch_nntile wheels** →
Artifacts, or:

```bash
gh run list --workflow=torch-nntile-wheels.yml --limit 5
gh run download RUN_ID -D wheelhouse
```

Manual dispatch (write access required):

```bash
gh workflow run torch-nntile-wheels.yml --ref graph_api
```

Install the matching `torch` first, then the local wheel (see
[`torch_nntile/README.md`](../torch_nntile/README.md#prebuilt-wheels-001) for
full commands). Maintainer CI notes: [build/README.md](build/README.md#torch_nntile-wheels-ci).

## Install from source

Build NNTile first (see [build/README.md](build/README.md)), then install a
matching `torch` and the editable extension:

```bash
pip install 'torch==2.9.1'
export NNTILE_BUILD_DIR=$PWD/build
export NNTILE_SOURCE_DIR=$PWD
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CXX=g++ pip install -e ./torch_nntile --no-build-isolation
```

## TensorGraph execution

Ops append to one shared ``TensorGraph``. Flush with ``compile_graph()`` and
``run()`` (or legacy ``execute()``) before host readout.

```python
import torch
import torch_nntile

torch_nntile.init_context(ncpu=4, ncuda=0, cpu_fallback=False)
x = torch.randn(32, 128).to("nntile")
y = model(x)
loss.backward()
torch_nntile.compile_graph()
torch_nntile.run()
loss_cpu = loss.to("cpu")  # host readout after run
```

Data transfer uses **`.to("nntile")` in** and **`.to("cpu")` out** — there is no
`bind_data` in torch_nntile. Legacy `execute()` still compiles, runs, and resets
in one call.

Training helper `torch_nntile.training.train_full_batch_step` calls
`compile_graph()` and `run()`; read loss with `loss.to("cpu")`.

**Tile memory:** New nntile tensors are marked as graph outputs; `del` clears
the mark when the last Python reference dies. At `compile_graph()` only live
tensors (and their graph dependencies) stay marked. During `run()`,
`Runtime::execute()` releases intermediate StarPU tiles after their last
consumer when those tiles are not graph inputs/outputs.

## Axis-group naming and tiling

Tiling in NNTile is defined on **shared axis groups** (`AxisDescriptor` in C++),
not on individual `torch.Tensor` storage. The workflow mirrors GPT-2 graph
training (`name_gpt2_training_axis_groups` + `apply_flat_tiling_spec`):

1. **Name** selected dimensions of a tensor (partial naming is OK).
2. Record forward/backward into the pending graph (ops merge related axes).
3. **Set tiling** by axis group name.
4. Optionally **inspect** axis groups before lowering.
5. **`compile_graph()`** / **`run()`** — tiling is applied, then
   `TileGraph::from_tensor_graph` and `Runtime::execute()`. Host I/O is only via
   `.to("nntile")` / `.to("cpu")`.

### API

| Function | Description |
|----------|-------------|
| `set_axis_group_name(tensor, {dim: name, ...})` | Name axis groups for listed tensor dimensions. Names propagate through merged groups. |
| `set_axis_group_tiling(name, tile_sizes)` | `tile_sizes` is `int` (uniform) or `list[int]` (heterogeneous; must sum to extent). Stored until `compile_graph()`. |
| `format_axis_groups()` | Return a string summary of pending graph axis groups (like C++ `TensorGraph::to_string`). |
| `print_axis_groups()` | Print that summary to stdout. Shows `pending_tile=` when tiling is registered but not yet applied. |

**Axis-group tiling** applies across a full training step before ``compile_graph()``.

### Minimal example

```python
torch_nntile.init_context(ncpu=2, cpu_fallback=False)

x = torch.randn(4, 128).to("nntile")
torch_nntile.set_axis_group_name(x, {0: "batch", 1: "features"})

logits = model(x)  # ops merge axes across the network

torch_nntile.set_axis_group_tiling("batch", [1, 1, 2])
torch_nntile.set_axis_group_tiling("features", 64)
torch_nntile.print_axis_groups()
torch_nntile.compile_graph()
torch_nntile.run()
```

Example `format_axis_groups()` / `print_axis_groups()` output:

```text
Pending TensorGraph: data=24, ops=12, axis_groups=4, tiled=0/4
Axis groups:
  extent=4 name='batch' pending_tile=1,1,2 members=8
  extent=128 name='features' pending_tile=64 members=6
  extent=256 members=4
  extent=10 name='classes' members=2
```

After `compile_graph()` + `run()`, pending ops are cleared but the compiled
session may persist for tile reuse. Call `format_axis_groups()` only while a
graph is pending (`has_pending_graph()`).

### Training helper hooks

`train_full_batch_step` accepts optional hooks for graph-mode nntile training:

```python
from torch_nntile.training import train_full_batch_step

def name_axis_groups(x, logits):
    torch_nntile.set_axis_group_name(x, {0: "batch", 1: "features"})
    torch_nntile.set_axis_group_name(logits, {1: "classes"})

loss = train_full_batch_step(
    model,
    x,
    labels,
    lr=0.1,
    name_axis_groups=name_axis_groups,
    axis_group_tiling={"batch": [15000, 15000, 15000, 15000, 15000]},
    print_axis_groups=True,  # once, before execute
)
```

Models such as `DeepReLU` do **not** assign axis names internally; the caller or
example script provides naming.

## DeepReLU MNIST example

[`torch_nntile/examples/train_deep_relu_mnist.py`](../torch_nntile/examples/train_deep_relu_mnist.py)
trains `DeepReLU.mnist()` on the full MNIST training set (60k batch), comparing
CPU PyTorch vs `device="nntile"`.

Default model: **5 linear layers** (`--depth 5`), **4 hidden blocks** with output
width `--hidden-dim 256` (784→256, then three 256→256, then 256→10 logits).

Axis groups used by the example:

| Name | Meaning | Extent (MNIST defaults) |
|------|---------|-------------------------|
| `batch` | input / logits batch dim | 60000 |
| `features` | flattened image dim | 784 |
| `hidden` | hidden MLP width (`--hidden-dim`) on weights, grads, velocities | 256 |
| `classes` | logits class dim | 10 |

There are **four** separate `hidden` axis groups in the graph (one per hidden
linear output). The example names them explicitly on each linear weight, grad,
and SGD velocity matrix row/column of size `hidden_dim`.

### Prerequisites

```bash
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
# editable install with libnntile linked (see torch_nntile/README.md)
```

StarPU worker counts come from the environment (`STARPU_NCPU`, `STARPU_NCUDA`).
The script calls `init_context(ncpu=-1, ncuda=-1, …)` so those env vars apply.

### CPU workers only

Use CPU StarPU workers for the nntile path (reference PyTorch path always runs
on CPU tensors):

```bash
STARPU_NCPU=4 STARPU_NCUDA=0 \
  python torch_nntile/examples/train_deep_relu_mnist.py \
    --runtime-mode graph \
    --epochs 5
```

Optional graph tiling and axis-group dump:

```bash
STARPU_NCPU=4 STARPU_NCUDA=0 \
  python torch_nntile/examples/train_deep_relu_mnist.py \
    --runtime-mode graph \
    --epochs 5 \
    --print-axis-groups \
    --axis-tiling batch=15000,15000,15000,15000 \
    --axis-tiling features=392,392 \
    --axis-tiling hidden=128,128
```

**Expected tail output (CPU workers, graph mode, 5 epochs):**

```
Loss comparison (cpu vs nntile):
  epoch 1: cpu=2.302172  nntile=2.302172  diff=2.384e-07
  epoch 2: cpu=2.302079  nntile=2.302080  diff=4.768e-07
  epoch 3: cpu=2.301987  nntile=2.301987  diff=0.000e+00
  epoch 4: cpu=2.301894  nntile=2.301895  diff=4.768e-07
  epoch 5: cpu=2.301802  nntile=2.301802  diff=0.000e+00

Final weight max |cpu - nntile| = 1.118e-08
```

Per-epoch loss diffs at or below **~1e-6** are typical on CPU.

### CUDA workers only

Pin nntile kernels to CUDA workers (`--restrict-cuda`):

```bash
STARPU_NCPU=0 STARPU_NCUDA=2 \
  python torch_nntile/examples/train_deep_relu_mnist.py \
    --runtime-mode graph \
    --restrict-cuda \
    --epochs 5 \
    --axis-tiling batch=15000,15000,15000,15000 \
    --axis-tiling features=392,392 \
    --axis-tiling hidden=128,128
```

**Expected tail output (CUDA workers, graph mode, 5 epochs, with tiling above):**

```
Loss comparison (cpu vs nntile):
  epoch 1: cpu=2.302172  nntile=2.302095  diff=7.701e-05
  epoch 2: cpu=2.302079  nntile=2.301964  diff=1.152e-04
  epoch 3: cpu=2.301987  nntile=2.301834  diff=1.538e-04
  epoch 4: cpu=2.301894  nntile=2.301818  diff=7.653e-05
  epoch 5: cpu=2.301802  nntile=2.301495  diff=3.073e-04

Final weight max |cpu - nntile| = 1.583e-08
```

On CUDA, per-epoch **loss** diffs of order **1e-4** are normal (cuBLAS / TF32 /
reduction order vs CPU reference). **Weights** should still match to **~1e-8**.
Call `torch_nntile.wait()` before reading losses on the host; the example shuts
down StarPU cleanly in a `finally` block.

### Useful flags

| Flag | Purpose |
|------|---------|
| `--runtime-mode graph` | Record full step, then `compile_graph()` + `run()` |
| `--restrict-cuda` | `restrict_cuda()` — CUDA workers only |
| `--verbose` | Verbose StarPU / NNTile context logging |
| `--hidden-dim`, `--depth` | Model size (default 256, 5) |
| `--axis-tiling NAME=SIZES` | Repeatable; auto-switches to graph mode |
| `--print-axis-groups` | Dump axis groups after epoch 1 (graph mode) |

`--axis-tiling` and `--print-axis-groups` switch to graph mode automatically if
omitted from `--runtime-mode`.

Integration test (downloads MNIST, 3 epochs, CPU workers):

```bash
pytest -vv -m slow torch_nntile/tests/test_deep_relu_mnist_train.py
```

## Tests

```bash
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
pytest -vv torch_nntile/tests/test_axis_group_tiling.py
pytest -vv torch_nntile/tests/test_graph_execution.py
pytest -vv torch_nntile/tests/test_deep_relu_parity.py
```

## Relation to C++ graph API

| C++ (GPT-2 graph training) | torch_nntile |
|----------------------------|--------------|
| `name_gpt2_training_axis_groups(...)` | `set_axis_group_name(tensor, {...})` per tensor |
| `apply_flat_tiling_spec` / `tiling.json` | `set_axis_group_tiling(name, sizes)` |
| `TensorGraph::to_string()` axis section | `format_axis_groups()` / `print_axis_groups()` |
| `TileGraph::from_tensor_graph` + `Runtime` | `execute()` |

For full-model JSON tiling (GPT-2), use the Python `nntile` package and
`apply_gpt2_tiling_json` — see [graph-wip.md](graph-wip.md).
