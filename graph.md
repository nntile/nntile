# NNTile Graph System

This document describes the current graph implementation in NNTile. It reflects
the code in `include/nntile/graph/` and `src/graph/`.

## File layout

```
include/nntile/
├── graph.hh
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

src/graph/
├── dtype.cc
├── tensor_graph_node.cc
├── compiled_graph.cc
├── nn_graph.cc
├── tensor/
│   ├── add.cc
│   ├── add_fiber.cc
│   ├── ...
│   └── sum_fiber.cc
└── nn/
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

Defined in `include/nntile/graph/tensor/` and `graph_ops.hh`:

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

GEMM shape rules (see `gemm_output_shape` in `tensor/gemm.hh`):

- Tensor layout is column-major; dimensions are listed from inner to outer.
- A: `trans_a=false` → `[M..., K..., batch...]`
- B: `trans_b=false` → `[K..., N..., batch...]`
- Output: `[M..., N..., batch...]`
- `ndim` is the number of contraction (K) dimensions.
- `batch_ndim` is the number of trailing batch dimensions (must match between A
  and B).

## NNGraph

`NNGraph` (in `nn/graph.hh` and `nn_graph.cc`) wraps `TensorGraph` and adds gradient tracking.

- `NNGraph::TensorNode` points to a `TensorGraph::DataNode` (via `.data()`) and
  tracks `grad` and `requires_grad`.
- `mark_input()` / `mark_output()` delegate to the underlying data node.
- `get_or_create_grad()` creates a gradient tensor in the underlying
  `TensorGraph` and clears it via `clear()`.

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

- `register_op(graph, op)` — when `graph.is_grad_enabled()` and any input requires grad, stores the op and sets `producer` on each output. The op's `outputs_` must be populated by `forward()` before registration. Use `graph.no_grad()` for a scope where grad recording is disabled.

## Adding new graph operations

### 1. Add a TensorGraph operation

**Header** (`include/nntile/graph/tensor/<op>.hh`):

- Define `TensorXxxOp : TensorGraphOpNode` with `execute()` and `clone()`.
- Declare free functions for the builder API.

**Source** (`src/graph/tensor/<op>.cc`):

- Implement the builder: validate inputs, create output via `graph->data()`,
  build op, call `graph->add_op(op)`.
- Implement `TensorXxxOp::execute()`: dispatch on DataType and call
  `nntile::tensor::*` kernel.

Add to `graph_ops.hh` if needed.

### 2. Add an NNGraph (autograd) operation

**Header** (`include/nntile/graph/nn/<op>.hh`):

- Define `NNXxxOp : NNGraph::OpNode` with constructor (inputs only), `forward(output_name)` returning `TensorNode*`, and `backward()`.
- Declare convenience free function.

**Source** (`src/graph/nn/<op>.cc`):

- `forward(output_name)`: create output via `graph.tensor()`, set `outputs_`, add tensor ops via `x->data()`, return output.
- `backward()`: use `output()->grad()`, `grad_x->data()`, etc. with tensor ops.
- Free function: `op = make_op(inputs); output = op->forward(output_name); register_op(graph, op); return output;`

See `docs/autograd_add_function.md` for a full guide. Add to `nn/graph_ops.hh`.

### 3. Build system

Update `src/CMakeLists.txt` and `include/CMakeLists.txt` if adding new files.

## Minimal example

Using NNGraph with gradients (see `examples/graph_mlp_example.cc` and
`examples/linear_layer_example.cc` for full examples):

```cpp
#include <nntile/context.hh>
#include <nntile/graph.hh>

using namespace nntile::graph;

nntile::Context context(
    1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0);

NNGraph graph("demo");
auto* x = graph.tensor({2, 3}, "x", DataType::FP32, true);
auto* w = graph.tensor({3, 4}, "w", DataType::FP32, true);
auto* y = gemm(x, w, "y");  // y = x @ w

x->mark_input(true);
y->mark_output(true);
y->backward();  // build backward pass
CompiledGraph compiled(graph.tensor_graph());
compiled.compile();
compiled.bind_data("x", input_data);
compiled.bind_data("w", weight_data);
compiled.execute();
compiled.wait();
auto out = compiled.get_output<float>("y");
```
