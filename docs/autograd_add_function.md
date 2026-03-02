# How to Add a New Autograd Function

This guide describes how to add a new autograd (differentiable) operation to the NNGraph system. Autograd functions live in `include/nntile/graph/nn_graph/` and `src/graph/nn_graph/`.

**Important:** Autograd functions shall only use `LogicalGraph::TensorNode` operations. All forward and backward logic must be expressed via the logical graph API (e.g., `add`, `add_inplace`, `gemm`, `gelu_backward`, etc.). You access the underlying logical tensor from an `NNGraph::TensorNode*` via `.data()` — logical ops take `LogicalGraph::TensorNode*` and return `LogicalGraph::TensorNode*` where applicable.

---

## Overview

Each autograd function consists of:

1. **Header** (`include/nntile/graph/nn_graph/<op>.hh`) — struct with `build_forward` and `build_backward`, plus a convenience free function
2. **Source** (`src/graph/nn_graph/<op>.cc`) — implementation of forward and backward
3. **Logical ops** — existing or new operations in `include/nntile/graph/logical/` that operate on `LogicalGraph::TensorNode*`

---

## Step 1: Ensure Logical Ops Exist

Your autograd function must be built from `LogicalGraph::TensorNode` operations only. Check `include/nntile/graph/logical/` for existing ops (e.g., `add`, `add_inplace`, `gemm`, `gelu`, `gelu_backward`, `sum_fiber`, `add_fiber`, `add_fiber_inplace`). If you need a new logical op, add it first in the logical graph layer.

---

## Step 2: Create the Header

Create `include/nntile/graph/nn_graph/<op>.hh`:

```cpp
#pragma once

#include <string>

#include <nntile/graph/logical/<op>.hh>   // logical forward/backward ops
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! <Op>: build_forward does logical op + bookkeeping; build_backward for grad.
namespace MyOp
{
    NNGraph::TensorNode* build_forward(
        NNGraph::TensorNode* x,           // adjust params as needed
        const std::string& output_name);

    void build_backward(const NNGraph::OpNode* op);
}

//! Convenience free function
inline NNGraph::TensorNode* my_op(
    NNGraph::TensorNode* x,
    const std::string& output_name)
{
    return MyOp::build_forward(x, output_name);
}

} // namespace nntile::graph
```

---

## Step 3: Implement build_forward

In `src/graph/nn_graph/<op>.cc`:

1. **Validate inputs** — reject null pointers.
2. **Get graph** — `NNGraph& graph = x->graph();`
3. **Call logical op** — use `x->data()` to get `LogicalGraph::TensorNode*` and pass to the logical function:
   ```cpp
   LogicalGraph::TensorNode* y_data = my_logical_op(x->data(), output_name);
   ```
4. **Create tensor** — `NNGraph::TensorNode* y = graph.tensor(y_data, out_requires_grad);`
5. **Compute `out_requires_grad`** — `bool out_requires_grad = any_input_requires_grad({x, ...});`
6. **Register op** — `register_op(graph, {x, ...}, y, attrs, backward_fn, buffers);`

Example (from `add.cc`):

```cpp
NNGraph::TensorNode* Add::build_forward(  // Add is a namespace
    Scalar alpha,
    NNGraph::TensorNode* x,
    Scalar beta,
    NNGraph::TensorNode* y,
    const std::string& output_name)
{
    if (x == nullptr || y == nullptr)
        throw std::invalid_argument("Add::build_forward: x and y must be non-null");
    NNGraph& graph = x->graph();
    LogicalGraph::TensorNode* z_data =
        add(alpha, x->data(), beta, y->data(), output_name);
    bool out_requires_grad = any_input_requires_grad({x, y});
    NNGraph::TensorNode* z = graph.tensor(z_data, out_requires_grad);
    register_op(graph, {x, y}, z, std::make_shared<BinaryOpAttrs>(BinaryOpAttrs{alpha, beta}),
                [](const NNGraph::OpNode* op) { Add::build_backward(op); }, {});
    return z;
}
```

---

## Step 4: Implement build_backward

In `build_backward`, you receive the `OpNode` and must propagate gradients to each input that `requires_grad()`.

1. **Extract context** — `op->output()`, `op->output()->grad()`, `op->inputs()`, `op->attrs()`.
2. **For each input that requires grad:**
   - `graph.get_or_create_grad(input, input->name() + "_grad")` to get/create the gradient tensor
   - Call logical ops on `grad_out->data()` and `grad_x->data()` (and similar) to accumulate gradients

**Accumulation rules:**

- **First gradient** — use `graph.is_first_grad(input)` to decide whether to clear or accumulate. If first, use `beta=0` (or `clear`) so the result is overwritten; otherwise use `beta=1` to accumulate.
- **In-place ops** — use logical in-place ops (e.g., `add_inplace`) to add into the gradient tensor.

Example (from `add.cc`):

```cpp
void Add::build_backward(const NNGraph::OpNode* op)
{
    NNGraph& graph = op->output()->graph();
    NNGraph::TensorNode* grad_out = op->output()->grad();
    const auto& attrs = *std::static_pointer_cast<BinaryOpAttrs>(op->attrs());
    Scalar alpha = attrs.alpha;
    Scalar beta = attrs.beta;
    const auto& inputs = op->inputs();
    if (inputs.size() >= 2 && grad_out != nullptr) {
        NNGraph::TensorNode* x_nn = inputs[0];
        NNGraph::TensorNode* y_nn = inputs[1];
        if (x_nn != nullptr && x_nn->requires_grad()) {
            NNGraph::TensorNode* grad_x =
                graph.get_or_create_grad(x_nn, x_nn->name() + "_grad");
            add_inplace(alpha, grad_out->data(), Scalar(1.0), grad_x->data());
        }
        if (y_nn != nullptr && y_nn->requires_grad()) {
            NNGraph::TensorNode* grad_y =
                graph.get_or_create_grad(y_nn, y_nn->name() + "_grad");
            add_inplace(beta, grad_out->data(), Scalar(1.0), grad_y->data());
        }
    }
}
```

Example with first-grad handling (from `gelu.cc`):

```cpp
if (x_nn != nullptr && x_nn->requires_grad()) {
    bool first = graph.is_first_grad(x_nn);
    NNGraph::TensorNode* grad_x =
        graph.get_or_create_grad(x_nn, x_nn->name() + "_grad");
    if (first)
        clear(grad_x->data());
    gelu_backward(x_nn->data(), grad_out->data(), grad_x->data());
}
```

---

## Step 5: Attrs (if needed)

If your op needs to store parameters for backward (e.g., `alpha`, `beta`, `axis`), define an attrs struct (often in the logical op header, e.g., `BinaryOpAttrs`, `GemmAttrs`, `ReductionAttrs`) and pass it to `register_op`:

```cpp
register_op(graph, inputs, output,
            std::make_shared<MyOpAttrs>(MyOpAttrs{...}),
            [](const NNGraph::OpNode* op) { MyOp::build_backward(op); },
            {});
```

In `build_backward`, cast and use:

```cpp
const auto& attrs = *std::static_pointer_cast<MyOpAttrs>(op->attrs());
```

---

## Step 6: Register in nn_graph_ops.hh

Add an include for your new op in `include/nntile/graph/nn_graph_ops.hh`:

```cpp
#include <nntile/graph/nn_graph/my_op.hh>
```

---

## Step 7: Add to Build System

Ensure your new `.cc` file is added to the build (e.g., in `CMakeLists.txt` or equivalent).

---

## Summary Checklist

- [ ] Logical ops exist and operate on `LogicalGraph::TensorNode*` only
- [ ] Header: struct with `build_forward` and `build_backward`, plus free function
- [ ] Forward: validate inputs, call logical op via `x->data()` (returns pointer), create tensor, `register_op`
- [ ] Backward: use only logical ops on `grad_out->data()`, `grad_x->data()`, etc.
- [ ] Handle `is_first_grad` when accumulating gradients
- [ ] Attrs struct and `register_op` if backward needs parameters
- [ ] Include in `nn_graph_ops.hh`
- [ ] Add to build system

---

## Reference: Existing Autograd Functions

| Op        | Header              | Logical ops used (forward / backward)                    |
|-----------|---------------------|----------------------------------------------------------|
| Add       | `add.hh`            | `add` / `add_inplace`                                    |
| Gelu      | `gelu.hh`           | `gelu` / `clear`, `gelu_backward`                        |
| Gemm      | `gemm.hh`           | `gemm` / `gemm` (for grad_A, grad_B)                    |
| AddFiber  | `add_fiber.hh`      | `add_fiber` / `sum_fiber`, `add_inplace`                 |
| SumFiber  | `sum_fiber.hh`      | `clear`, `sum_fiber` / `add_fiber_inplace`               |
