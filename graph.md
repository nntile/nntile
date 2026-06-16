# NNTile Graph System

This document describes the current graph implementation in NNTile. It reflects
the code in `include/nntile/` and `nntile/src/`. Eager execution (kernels,
StarPU, tile, tensor) lives in the **core** package (`include/nntile/`,
`nntile/src/`, CMake target `nntile`, namespace `nntile::core`).

## File layout

```
include/nntile/
├── nntile.hh          # core + graph umbrellas
├── core.hh
├── graph.hh
├── core/              # kernels, starpu, tile, tensor, context
└── graph/
    ├── dtype.hh
    ├── tensor.hh
    ├── nn.hh
    ├── compiled.hh
    ├── compiled/
    │   └── graph.hh
    ├── tensor/
    │   ├── graph.hh
    │   ├── graph_exec_ctx.hh
    │   ├── graph_op_node.hh
    │   ├── graph_tensor_node.hh
    │   ├── graph_ops.hh
    │   ├── add.hh
    │   ├── add_fiber.hh
    │   ├── add_fiber_inplace.hh
    │   ├── add_inplace.hh
    │   ├── clear.hh
    │   ├── fill.hh
    │   ├── multiply.hh
    │   ├── norm.hh
    │   ├── gemm.hh
    │   ├── gelu.hh
    │   ├── gelu_backward.hh
    │   └── sum_fiber.hh
    └── nn/
        ├── graph.hh
        ├── graph_op_node.hh
        ├── graph_tensor_node.hh
        ├── graph_ops.hh
        ├── add.hh
        ├── add_fiber.hh
        ├── gemm.hh
        ├── gelu.hh
        └── sum_fiber.hh

nntile/src/
├── dtype.cc
├── tensor/
│   ├── graph_data_node.cc
│   ├── add.cc
│   ├── add_fiber.cc
│   ├── ...
│   └── sum_fiber.cc
└── nn/
    ├── graph.cc
    ├── tensor_node.cc
    ├── add.cc
    ├── add_fiber.cc
    ├── gemm.cc
    ├── gelu.cc
    └── sum_fiber.cc
```

## TensorGraph

`TensorGraph` is a symbolic computation graph that operates on tensor data nodes.

- `TensorGraph::TensorNode` (TensorGraphNode) holds `shape`, `dtype`, `name`.
- `TensorGraph::OpNode` (TensorGraphOpNode) holds `inputs`, `outputs`, and implements
  `execute(ExecutionContext&)`.
- `data(shape, name, dtype)` creates a data node.
- `add_op(shared_ptr<TensorGraphOpNode>)` adds an operation to the graph.

### Input/output marking

Data nodes can be marked as graph input and/or output via `mark_input()` and
`mark_output()` on `TensorGraph::TensorNode`.

- **Input tensors** (`mark_input(true)`): Provided via `bind_data()`; never
  invalidated during execution.
- **Output tensors** (`mark_output(true)`): Retrieved via `get_output()`; never
  invalidated during execution.

`bind_data()` may only be called for tensors marked as input or output (or
both). This ensures that user-bound data is never invalidated unexpectedly.

When a graph executes, intermediate tensors that are no longer used by
remaining operations are automatically invalidated via `invalidate_submit()` to
free memory. Input and output tensors are never invalidated.

### Data types

`DataType` is defined in `dtype.hh` and includes:

- `FP32`, `FP32_FAST_TF32`, `FP32_FAST_FP16`, `FP32_FAST_BF16`
- `FP64`, `FP16`, `BF16`
- `INT64`, `BOOL`

### Tensor graph operations

Defined in `include/nntile/tensor/` and `graph_ops.hh`:

**Element-wise operations:**
- `add(alpha, x, beta, y, output_name)` — creates z = alpha*x + beta*y
- `add_inplace(alpha, x, beta, y)` — in-place y = alpha*x + beta*y
- `multiply(x, y, output_name)` — creates z = x*y
- `clear(x)` — in-place clear

**Reduction operations:**
- `sum_fiber(x, y, axis, batch_ndim, alpha, beta)` — sum along fibers

**Matrix operations:**
- `gemm(a, b, output_name, alpha, trans_a, trans_b, ndim, batch_ndim)` —
  creates a new output tensor.
- `gemm(a, b, c, alpha, beta, trans_a, trans_b, ndim, batch_ndim)` — in-place
  accumulation into `c`.

**Activation operations:**
- `gelu(x, output_name)` — creates GeLU output tensor
- `gelu_backward(x, dy, dx)` — backward pass for GeLU

**Utility operations:**
- `fill(x, value)` — fill tensor with scalar value

GEMM shape rules (see `gemm_output_shape` in `tensor/ops/gemm.hh`):

- `TensorGraph::TensorNode::shape()` uses **graph** (C-order) labels; tile
  kernels use **storage** shapes (`storage_shape()` / `graph_shape_to_storage`).
- **NNGraph** `gemm(a, b, trans_a, trans_b, ndim, batch_ndim)` forwards to
  `tensor::gemm(a, b, trans_a, trans_b, ndim, batch_ndim)` on graph shapes
  (no operand swap at the NN/tensor builder boundary).
- **Tile lowering** (`TensorGemmOp::lower_to_tile`) calls
  `tile::gemm(a, b, c, trans_a, trans_b, …)` with graph operand order (no swap).
- **Tile execute** (`TileGemmOp::execute`) maps graph-axis GEMM to Fortran
  `core::gemm` by swapping operands and transpose flags at the kernel boundary.
- `trans_a` / `trans_b` transpose the contracted ``ndim`` axes of ``a`` / ``b``
  (see `gemm_output_shape` for the exact index ranges).
- `ndim` is the number of contraction (K) dimensions.
- `batch_ndim` is the number of **leading** batch dimensions shared as the
  shape prefix `[0, batch_ndim)` in both `a` and `b`.

Example usages (not special cases in the op itself):

- `Linear` calls `gemm(input, weight, trans_b=true, ...)` with weight `[out, in]`.
- Attention Q/K/V call `gemm(x, w, trans_a=false, ...)` with weight
  `[hidden, head_size, n_heads]` and a following `transpose` to SDPA layout.

### Graph shape labels

`TensorGraph` and `NNGraph` expose **graph** shapes on `TensorNode::shape()`
(outermost / slowest dimension first, C-order). Tile storage and kernels use
reversed axis labels. Helpers in `include/nntile/tensor/shape_layout.hh`
(and re-exported from `include/nntile/nn/shape_layout.hh`) convert between the
two:

- `tensor::graph_shape_to_storage(graph_shape)` — graph label → tile storage shape
- `tensor::storage_shape_to_graph(storage_shape)` — tile storage → graph label
- `tensor::graph_axis_to_storage(graph_axis, ndim)` — graph axis (0 = outermost)
  → storage axis (0 = innermost)
- `tensor::storage_axis_to_graph(storage_axis, ndim)` — storage axis → graph axis

Most reduction and layout ops take **graph axis indices** at the public
TensorGraph / NNGraph API. `TensorGraph` op nodes store graph axes only;
`lower_to_tile` uses graph axes for tiling logic and calls
`layout_axis(graph_axis, ndim)` only when indexing `TensorAxisLayout` grid
coordinates (storage-indexed internally). Tile execute converts to storage
for `core::*`.

**Fiber tensors** (`batch_ndim > 0`): graph shape is
`[batch_0, …, batch_{batch_ndim-1}, fiber_dim]` (C-order: batch slower,
fiber faster). The fiber extent matches `tensor.shape[axis]`; leading batch
axes match `tensor.shape[0:batch_ndim)`.

**Exception — NNGraph `transpose`:** graph model code was written with
**storage-order** transpose axes (historical `graph_api` convention). The NN
layer maps model `ndim` to tensor `src->ndim() - ndim` so unchanged model
sources keep working. TensorGraph `tensor::transpose` always takes a **graph**
axis count.

#### Model tensor conventions

Graph model families (GPT-2, BERT, GPT-Neo, GPT-NeoX, Llama, RoBERTa, T5) use
these graph layouts:

| Role | Shape | Notes |
|------|-------|-------|
| Activations | `[batch, seq, hidden]` | Embedding output and block I/O |
| Linear weights | `[out, in]` | Q/K/V/O projections, MLP, LM head |
| Logits | `[batch, seq, vocab]` | Causal / MLM heads |
| `input_ids` | `[batch, seq]` | Token indices |
| `position_ids` | `[batch, seq]` | Position indices |
| Attention mask | `[seq, seq]` or `[batch, seq, seq]` | Bool or float mask |

Safetensors metadata records graph shapes (e.g. linear weights `{out, in}`);
payload bytes use explicit transposes in test generators (`as_bind_float32()`).

#### Example: GPT-2 attention

With activations `x` shaped `[batch, seq, hidden]` and Q weight
`[hidden, head_size, n_heads]`:

- `gemm(x, w_q, alpha, false, false, ndim=1, batch_ndim=0)` — Q projection
  → `[batch, seq, head_size, n_heads]`
- `transpose(q_proj, 1)` — model/storage-order axis; NN maps to graph cyclic
  shift → `[n_heads, batch, seq, head_size]` for SDPA
- `add_fiber(..., q_bias, ..., axis=3, batch_ndim=1)` — per-head bias
  (`q_bias` graph shape `[n_heads, head_size]`; `axis` is a **graph** axis)
- `sdpa_eager(q, k, v, mask, batch_ndim=2, redux=0)`
- `transpose(attn_out, 3)` — model/storage-order axis; maps to graph shift
  before output projection
- `gemm(attn_t, w_o, ..., false, false, ndim=2, batch_ndim=0)` — output projection
- `add_fiber(o_bias, ..., axis=2, batch_ndim=0)` — output bias on hidden axis

BERT/RoBERTa embeddings: sum word/position/token-type, then
`transpose(embed, 2)` (storage-order axis) → `[batch, seq, hidden]` before
`layer_norm(..., axis=2)` (graph axis).

## TileGraph

`TileGraph` is the tiled execution graph produced by lowering `TensorGraph`.
It mirrors TensorGraph's C-order shape labels at the public API while
`core::Tile` allocation and kernels remain Fortran-ordered.

- `TileGraph::TileNode::shape()` — **graph** (C-order) labels, matching the
  parent `TensorGraph::TensorNode::shape()` for the logical tensor.
- `TileGraph::TileNode::storage_shape()` — Fortran storage shape passed to
  `core::Tile` (via `tensor::graph_shape_to_storage`).
- `TensorDescriptor.tile_shape` in `from_tensor_graph` / append phases uses
  graph-order labels; `grid_shape` / `tile_coord` stay storage-indexed
  (`TensorAxisLayout` is unchanged).

Shape conversion helpers are re-exported from `include/nntile/tile/shape_layout.hh`
(same functions as `tensor::shape_layout.hh`).

### Tile op axis conventions

Axis-aware tile ops (`*_fiber*`, `*_slice*`, `softmax`, `maxsumexp`, `transpose`)
take **graph axis** indices (or graph leading-axis counts for `transpose`) at
the tile API. `execute()` converts with `graph_axis_to_storage(axis, ndim)`
before calling `core::*`, where `ndim` is the rank of the tensor the axis
refers to (the full tensor for reductions, not the reduced output). For
`transpose`, `ndim` is the number of leading **graph** axes in the cyclic
shift; execute maps it to storage via `src->ndim() - ndim`.

`TensorGraph` lowering passes graph axes to tile ops for slice/softmax-family ops.
Fiber and reduction ops also store graph axes; tile calls receive graph axes
directly.

Layout-parameter ops (`embedding`, `conv2d_*`, `copy_intersection`, `rope`,
`flash_sdpa`, `mask_scalar`) keep storage-indexed geometry from tensor lowering;
`mask_scalar` takes `batch_ndim` (leading graph batch rank), not an axis index.
No tile-layer shape relabeling beyond execute-time kernel conventions.

### TileGraph GEMM

- Tensor lowering: `tile::gemm(ta, tb, tc, trans_a, trans_b, ndim, batch_ndim)`
  (graph operand order).
- Tile execute: calls `core::gemm` with operand/transpose swap for Fortran kernels.

## NNGraph

`NNGraph` (in `nn/graph.hh` and `nn/graph.cc`) wraps `TensorGraph` and adds gradient tracking.

- `NNGraph::TensorNode` points to a `TensorGraph::DataNode` (via `.data()`) and
  tracks `grad` and `requires_grad`.
- `mark_input()` / `mark_output()` delegate to the underlying data node.
- `get_or_create_grad(tensor, grad_name)` returns `(grad_tensor, is_first_write)`.
  It does NOT add a CLEAR op. Use `is_first_write` to choose overwrite (beta=0)
  or accumulate (beta=1) in backward ops. Ops that use `+=` (e.g., gelu_backward)
  must add `clear(grad->data())` when `is_first_write` before the backward op.

Autograd operations use `TensorGraph` ops for forward. For `NNGraph::TensorNode* x`,
pass `x->data()` to tensor ops to get `TensorGraph::TensorNode*`.

### NN*Op structs (PyTorch-style)

Each autograd operation is defined as a struct (e.g., `NNAddOp`, `NNGemmOp`)
inheriting from `NNGraph::OpNode`:

- **Constructor**: Takes inputs only (no outputs). Outputs are created in `forward()`.
- **forward(output_name)**: Creates output(s), sets `outputs_`, adds tensor graph ops, returns primary output.
- **backward()**: Uses `output()->grad()` and propagates gradients to inputs via tensor ops.

This mirrors PyTorch: outputs and temporaries appear in the forward pass, not at construction.

### register_op

- `graph.register_op(op)` — when `graph.is_grad_enabled()` and any input requires grad, stores the op and sets `producer` on each output. The op's `outputs_` must be populated by `forward()` before registration. Use `graph.no_grad()` for a scope where grad recording is disabled.

## Adding new graph operations

### 1. Add a TensorGraph operation

**Header** (`include/nntile/tensor/<op>.hh`):

- Define `TensorXxxOp : TensorGraphOpNode` with `execute()` and `clone()`.
- Declare free functions for the builder API.

**Source** (`nntile/src/tensor_graph/<op>.cc`):

- Implement the builder: validate inputs, create output via `graph->data()`,
  build op, call `graph->add_op(op)`.
- Implement `TensorXxxOp::execute()`: dispatch on DataType and call
  `nntile::tensor::*` kernel.

Add to `graph_ops.hh` if needed.

### 2. Add an NNGraph (autograd) operation

**Header** (`include/nntile/nn/<op>.hh`):

- Define `NNXxxOp : NNGraph::OpNode` with constructor (inputs only), `forward(output_name)` returning `TensorNode*`, and `backward()`.
- Declare convenience free function.

**Source** (`nntile/src/nn_graph/<op>.cc`):

- `forward(output_name)`: create output via `graph.tensor()`, set `outputs_`, add tensor ops via `x->data()`, return output.
- `backward()`: use `output()->grad()`, `grad_x->data()`, etc. with tensor ops.
- `op = make_op(inputs); output = op->forward(output_name); graph.register_op(op); return output;`

See `docs/autograd_add_function.md` for a full guide. Add to `nn/graph_ops.hh`.

### 3. Build system

Update `src/CMakeLists.txt` and `include/CMakeLists.txt` if adding new files.

## Minimal example

Using NNGraph with gradients (see `nntile/examples/graph_mlp_example.cc` and
`nntile/examples/linear_layer_example.cc` for full examples):

```cpp
#include <nntile/context.hh>
#include <nntile/graph.hh>
#include <nntile/runtime.hh>

using namespace nntile;

nntile::Context context(
    1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0);

NNGraph graph("demo");
auto* x = graph.tensor({2, 3}, "x", DataType::FP32, true);
auto* w = graph.tensor({4, 3}, "w", DataType::FP32, true);  // out=4, in=3
auto* y = gemm(x, w, "y");  // shape (2, 4) with trans_b=true in Linear

x->mark_input(true);
y->mark_output(true);
y->backward();  // build backward pass

TileGraph tile_graph = TileGraph::from_tensor_graph(graph.tensor_graph());
Runtime runtime(tile_graph);
runtime.compile();
runtime.bind_data(x->data(), x_data);
runtime.bind_data(w->data(), w_data);
runtime.execute();
runtime.wait();
auto out = runtime.get_output<float>(y->data());
```
