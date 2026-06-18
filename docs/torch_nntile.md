# torch_nntile

PyTorch **PrivateUse1** backend registered as `device="nntile"`. When built
against `libnntile`, selected ops record into a shared `TensorGraph`, lower to
`TileGraph`, and run through `Runtime` (StarPU).

Package README: [`torch_nntile/README.md`](../torch_nntile/README.md).

## Install

Build NNTile first (see [build/README.md](build/README.md)), then:

```bash
export NNTILE_BUILD_DIR=$PWD/build
export NNTILE_SOURCE_DIR=$PWD
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CXX=g++ pip install -e ./torch_nntile --no-build-isolation
```

## Runtime modes

| Mode | Behavior |
|------|----------|
| `eager` (default) | Each op records a micro-graph and runs it immediately |
| `graph` | Ops append to one `TensorGraph` until `torch_nntile.execute()` |

```python
import torch
import torch_nntile

torch_nntile.init_context(ncpu=4, ncuda=0, cpu_fallback=False, runtime_mode="graph")
x = torch.randn(32, 128).to("nntile")
y = model(x)
loss.backward()
torch_nntile.execute()  # required before .cpu() / .item() on nntile tensors
```

Training helper `torch_nntile.training.train_full_batch_step` calls `execute()`
automatically in graph mode.

## Axis-group naming and tiling

Tiling in NNTile is defined on **shared axis groups** (`AxisDescriptor` in C++),
not on individual `torch.Tensor` storage. The workflow mirrors GPT-2 graph
training (`name_gpt2_training_axis_groups` + `apply_flat_tiling_spec`):

1. **Name** selected dimensions of a tensor (partial naming is OK).
2. Record forward/backward into the pending graph (ops merge related axes).
3. **Set tiling** by axis group name.
4. Optionally **inspect** axis groups before lowering.
5. **`execute()`** — tiling is applied, then `TileGraph::from_tensor_graph`.

### API

| Function | Description |
|----------|-------------|
| `set_axis_group_name(tensor, {dim: name, ...})` | Name axis groups for listed tensor dimensions. Names propagate through merged groups. |
| `set_axis_group_tiling(name, tile_sizes)` | `tile_sizes` is `int` (uniform) or `list[int]` (heterogeneous; must sum to extent). Stored until `execute()`. |
| `format_axis_groups()` | Return a string summary of pending graph axis groups (like C++ `TensorGraph::to_string`). |
| `print_axis_groups()` | Print that summary to stdout. Shows `pending_tile=` when tiling is registered but not yet applied. |

**Graph mode required** for axis-group tiling across a full training step.

### Minimal example

```python
torch_nntile.init_context(ncpu=2, runtime_mode="graph", cpu_fallback=False)

x = torch.randn(4, 128).to("nntile")
torch_nntile.set_axis_group_name(x, {0: "batch", 1: "features"})

logits = model(x)  # ops merge axes across the network

torch_nntile.set_axis_group_tiling("batch", [1, 1, 2])
torch_nntile.set_axis_group_tiling("features", 64)
torch_nntile.print_axis_groups()
torch_nntile.execute()
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

After `execute()`, the recorder resets; call `format_axis_groups()` only while a
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

Axis groups used by the example:

| Name | Tensor / dim | Extent (MNIST) |
|------|----------------|----------------|
| `batch` | input `x` dim 0, logits dim 0 | 60000 |
| `features` | input `x` dim 1 | 784 |
| `classes` | logits dim 1 | 10 |

Naming is defined in the example:

```python
def name_mnist_axis_groups(x, logits):
    torch_nntile.set_axis_group_name(x, {0: "batch", 1: "features"})
    torch_nntile.set_axis_group_name(logits, {1: "classes"})
```

CLI:

```bash
export LD_LIBRARY_PATH=$PWD/build/nntile:/opt/starpu/lib
python torch_nntile/examples/train_deep_relu_mnist.py --epochs 5
python torch_nntile/examples/train_deep_relu_mnist.py --epochs 5 --runtime-mode graph
python torch_nntile/examples/train_deep_relu_mnist.py --epochs 5 \
  --print-axis-groups \
  --axis-tiling batch=15000,15000,15000,15000,15000
```

`--axis-tiling` and `--print-axis-groups` switch to graph mode automatically.

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
