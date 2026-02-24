# NNTile Graph System

This document describes the current graph implementation in NNTile. It reflects
the code in `include/nntile/graph/` and `src/graph/`.

## File layout

```
include/nntile/
├── graph.hh
└── graph/
    ├── logical_graph.hh
    ├── logical_graph_ops.hh
    ├── compiled_graph.hh
    ├── compiled_graph_ops.hh
    └── nn_graph.hh

src/graph/
├── logical_graph.cc
├── logical_graph_ops.cc
├── compiled_graph.cc
├── compiled_graph_ops.cc
└── nn_graph.cc
```

## LogicalGraph

`LogicalGraph` is a symbolic computation graph.

- `LogicalGraph::TensorNode` holds `shape`, `dtype`, `name`, and edges
  (producer/consumers).
- `LogicalGraph::OpNode` holds `OpType`, `OpAttrs`, input/output tensors.
- `tensor(shape, name, dtype)` creates an input tensor (no producer).
- `add_op(...)` is the public builder API used by free-function operations.

### Input/output marking

Tensors can be marked as graph input and/or output via `mark_input()` and
`mark_output()` on `LogicalGraph::TensorNode` (and `NNGraph::TensorNode`).

- **Input tensors** (`mark_input(true)`): Provided via `bind_data()`; never
  invalidated during execution.
- **Output tensors** (`mark_output(true)`): Retrieved via `get_output()`; never
  invalidated during execution.

When a compiled graph executes, intermediate tensors that are no longer used by
remaining operations are automatically invalidated via `invalidate_submit()` to
free memory. Input and output tensors are never invalidated.

### Data types

`DataType` is defined in `logical_graph.hh` and includes:

- `FP32`, `FP32_FAST_TF32`, `FP32_FAST_FP16`, `FP32_FAST_BF16`
- `FP64`, `FP16`, `BF16`
- `INT64`, `INT32`, `BOOL`

### Operation types

`OpType` currently includes:

**Element-wise unary operations:**
- `GELU`, `GELU_INPLACE`
- `GELUTANH`, `GELUTANH_INPLACE`
- `GELUTANH_BACKWARD`
- `RELU`, `RELU_INPLACE`, `RELU_BACKWARD`
- `SILU`, `SILU_INPLACE`, `SILU_BACKWARD`
- `SQRT`, `SQRT_INPLACE`
- `HYPOT`, `HYPOT_INPLACE`

**Element-wise binary operations:**
- `ADD`, `ADD_INPLACE`
- `MULTIPLY`, `MULTIPLY_INPLACE`
- `HYPOT_SCALAR_INVERSE`

**Reduction operations:**
- `SUM` (total sum)
- `SUM_FIBER`, `SUM_SLICE`
- `NORM_FIBER`, `NORM_FIBER_INPLACE`, `NORM_SLICE`, `NORM_SLICE_INPLACE`
- `LOGSUMEXP`, `MAXSUMEXP`
- `SUMPROD_FIBER`, `SUMPROD_SLICE`

**Scale operations:**
- `SCALE`, `SCALE_INPLACE`
- `SCALE_FIBER`, `SCALE_SLICE`

**Matrix operations:**
- `GEMM`
- `TRANSPOSE`

**Convolution operations:**
- `CONV2D_INPLACE`, `CONV2D_BWD_INPUT_INPLACE`, `CONV2D_BWD_WEIGHT_INPLACE`

**Embedding operations:**
- `EMBEDDING`, `EMBEDDING_BACKWARD`

**Mixed-dtype operations:**
- `MASK_SCALAR`
- `TOTAL_SUM_ACCUM`

**Optimizer operations:**
- `SGD_STEP`, `ADAM_STEP`, `ADAMW_STEP`

**Utility operations:**
- `CLEAR`
- `COPY`, `COPY_INTERSECTION`
- `GATHER`, `SCATTER`
- `FILL`
- `POW`, `LOG_SCALAR`
- `RANDN`

**Flash attention (CUDA-only):**
- `FLASH_SDPA_FWD_CUDNN`, `FLASH_SDPA_BWD_CUDNN`

**Rotary position embedding:**
- `ROPE`, `ROPE_BACKWARD`

Only a subset of these operations are currently implemented as graph builders.
See below for the implemented operations.

## Logical graph operations

Defined in `logical_graph_ops.hh/.cc` as free functions:

**Element-wise operations:**
- `gelu(x, output_name)` - creates GeLU output tensor
- `gelu_inplace(x)` - in-place GeLU
- `gelutanh(x, output_name)` - creates GeLU tanh output tensor
- `gelutanh_inplace(x)` - in-place GeLU tanh
- `relu(x, output_name)` - creates ReLU output tensor
- `relu_inplace(x)` - in-place ReLU
- `silu(x, output_name)` - creates SiLU output tensor
- `silu_inplace(x)` - in-place SiLU
- `sqrt(x, output_name)` - creates sqrt output tensor
- `sqrt_inplace(x)` - in-place sqrt
- `*_backward(x, dy, dx)` - backward operations that accumulate gradients

**Binary operations:**
- `add(alpha, x, beta, y, output_name)` - creates z = alpha*x + beta*y
- `add_inplace(alpha, x, beta, y)` - in-place y = alpha*x + beta*y
- `multiply(x, y, output_name)` - creates z = x*y
- `multiply_inplace(x, y)` - in-place y = x*y

**Reduction operations:**
- `sum(x, y, alpha=1.0, beta=0.0)` - total sum: y = alpha*sum(x) + beta*y
- `sum_fiber(x, y, axis=0, batch_ndim=0, redux=0, alpha=1.0, beta=0.0)` - sum along fibers

**Scale operations:**
- `scale(x, output_name, alpha=1.0)` - creates y = alpha*x
- `scale_inplace(x, alpha=1.0)` - in-place x = alpha*x

**Embedding operations:**
- `embedding(index, vocab, output_name, axis=0)` - embedding lookup
- `embedding_backward(index, vocab, embed, axis=0)` - embedding backward

**Matrix operations:**
- `gemm(a, b, output_name, alpha, trans_a, trans_b, ndim, batch_ndim)` -
  creates a new output tensor.
- `gemm(a, b, c, alpha, beta, trans_a, trans_b, ndim, batch_ndim)` - in-place
  accumulation into `c`.

**Utility operations:**
- `clear(x)` - in-place clear; output tensor is `x`, no inputs.

GEMM shape rules (see `compute_gemm_output_shape` in
`logical_graph_ops.cc`):

- Tensor layout is column-major; dimensions are listed from inner to outer.
- A:
  - `trans_a=false`: `[M..., K..., batch...]`
  - `trans_a=true`: `[K..., M..., batch...]`
- B:
  - `trans_b=false`: `[K..., N..., batch...]`
  - `trans_b=true`: `[N..., K..., batch...]`
- Output: `[M..., N..., batch...]`
- `ndim` is the number of contraction (K) dimensions.
- `batch_ndim` is the number of trailing batch dimensions (must match between A
  and B).

## NNGraph

`NNGraph` (in `nn_graph.hh/.cc`) wraps `LogicalGraph` and adds gradient tracking.

- `NNGraph::TensorNode` points to a logical tensor and tracks `grad` and
  `requires_grad`.
- `mark_input()` / `mark_output()` delegate to the underlying logical tensor.
- `get_or_create_grad()` creates a gradient tensor in the underlying
  `LogicalGraph` and clears it via `clear()`.

Logical operations still operate on `LogicalGraph::TensorNode`. When using
`NNGraph`, pass `tensor.data()` to logical ops as needed.

## CompiledGraph

`CompiledGraph` (in `compiled_graph.hh/.cc`) executes a `LogicalGraph`.

- `compile(logical)` validates operation data types, allocates NNTile tensors, and
  eliminates dead ops (ops whose outputs are never consumed and not marked output).
- Each logical tensor is allocated as a single tile (tile shape equals full
  shape).
- Execution order is the order of `logical.ops()`; build ops in topological
  order.
- `execute()` runs operations in order. After each op, tensors that are no
  longer used by remaining ops (and are not marked input/output) are
  invalidated via `invalidate_submit()` to free memory.
- `wait()` calls `starpu_task_wait_for_all()`.

### Data binding and output

- `bind_data(name, data, count)` and `bind_data(name, vector)` copy into the
  single tile with StarPU `STARPU_W`.
- `get_output<T>(name)` copies out with `STARPU_R`.
- Explicit instantiations are provided for
  `bind_data<float|double|long long>` and `get_output<float|double>`.

### Compiled operations

Implemented in `compiled_graph_ops.hh/.cc`:

- `execute_clear`
- `execute_gelu`
- `execute_gelu_backward`
- `execute_gemm`

Each dispatches on `DataType` and calls the corresponding
`nntile::tensor::*` operation.

## Adding new graph operations

This section mirrors the current extension flow for graph operations.

### 1. Add OpType and OpAttrs

In `include/nntile/graph/logical_graph.hh`:

- Add a new `OpType` entry.
- Add an attribute struct (if needed) and register it in `OpAttrs`.

In `src/graph/logical_graph.cc`, update `op_type_to_string()` for the new type.
If the operation has dtype restrictions, update
`validate_operation_data_types()` in `src/graph/compiled_graph.cc`.

### 2. Add the logical operation

Declare the builder in `include/nntile/graph/logical_graph_ops.hh` and implement
it in `src/graph/logical_graph_ops.cc`.

The logical operation should:

- Validate inputs (graph ownership, dtype compatibility, shape rules).
- Compute output shape (if creating a new output).
- Create output tensors with `graph.tensor(...)` (if needed).
- Call `graph.add_op(...)` to wire inputs and outputs.

For in-place or accumulation operations, pass the output tensor in both inputs
and outputs (see `gemm(..., c, ...)` or `gelu_backward(x, dy, dx)`).

### 3. Add the compiled operation

Declare the executor in `include/nntile/graph/compiled_graph_ops.hh` and
implement it in `src/graph/compiled_graph_ops.cc`.

The compiled operation should:

- Read input/output names from `OpExecutionInfo`.
- Extract attributes from `OpExecutionInfo::attrs`.
- Dispatch on `DataType` and call the corresponding `nntile::tensor::*` kernel
  using `CompiledGraph::get_tensor<T>()`.

### 4. Register the executor

Update `CompiledGraph::execute_op()` in `src/graph/compiled_graph.cc` to handle
the new `OpType` and call the new executor.

### 5. Tests

Add tests alongside the existing graph tests:

- Logical op tests: `tests/graph/logical_graph_ops.cc`
- Compiled op tests: `tests/graph/compiled_graph_ops.cc`

Compiled graph tests use `GraphTestFixture` for `nntile::Context` setup.

### 6. Build system updates

If you add new source or header files (instead of extending the existing ops
files), update `src/CMakeLists.txt` and `include/CMakeLists.txt`.

## Minimal example

```cpp
#include <nntile/context.hh>
#include <nntile/graph.hh>

using namespace nntile::graph;

nntile::Context context(
    1, 0, 0, "/tmp/nntile_ooc", 16777216, 0, "localhost", 5001, 0);

LogicalGraph g("demo");
auto& a = g.tensor({2, 3}, "a", DataType::FP32);
auto& b = g.tensor({3, 4}, "b", DataType::FP32);
auto& c = gemm(a, b, "c");
auto& y = gelu(c, "y");

auto compiled = CompiledGraph::compile(g);
compiled.bind_data("a", std::vector<float>{1, 2, 3, 4, 5, 6});
compiled.bind_data("b", std::vector<float>{
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
compiled.execute();
compiled.wait();
auto out = compiled.get_output<float>("y");
```
