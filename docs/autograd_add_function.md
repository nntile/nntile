# How to Add a New Autograd Function

This guide describes how to add a new autograd (differentiable) operation to the NNGraph system. Autograd functions live in `include/nntile/graph/nn/` and `src/graph/nn/`.

**Important:** Autograd functions use `TensorGraph::TensorNode` operations for forward and backward. All logic is expressed via the tensor graph API (e.g., `add`, `add_inplace`, `gemm`, `gelu_backward`, etc.). You access the underlying data node from an `NNGraph::TensorNode*` via `.data()` — tensor ops take `TensorGraph::TensorNode*` and return `TensorGraph::TensorNode*` where applicable.

---

## PyTorch-Style Design

Autograd ops follow a **PyTorch-style** design: **outputs and temporaries are created inside `forward()`**, not before. The op is constructed with inputs only; `forward(output_name)` creates the output(s), adds tensor graph ops, and returns the primary output.

---

## Overview

Each autograd function consists of:

1. **Header** (`include/nntile/graph/nn/<op>.hh`) — struct with `forward()` and `backward()`, plus a convenience free function
2. **Source** (`src/graph/nn/<op>.cc`) — implementation of forward and backward
3. **Tensor ops** — existing or new operations in `include/nntile/graph/tensor/` that operate on `TensorGraph::TensorNode*`

---

## Step 1: Ensure Tensor Ops Exist

Your autograd function must be built from `TensorGraph::TensorNode` operations only. Check `include/nntile/graph/tensor/` for existing ops (e.g., `add`, `add_inplace`, `gemm`, `gelu`, `gelu_backward`, `sum_fiber`, `add_fiber`, `add_fiber_inplace`). If you need a new tensor op, add it first in the tensor graph layer.

---

## Step 2: Create the Header

Create `include/nntile/graph/nn/<op>.hh`:

```cpp
#pragma once

#include <string>

#include <nntile/graph/tensor/<op>.hh>
#include <nntile/graph/nn.hh>

namespace nntile::graph
{

//! MyOp: PyTorch-style — outputs created in forward().
struct NNMyOp : NNGraph::OpNode
{
    // Parameters needed for backward (e.g., alpha, beta)
    Scalar alpha = 1.0;
    NNGraph::TensorNode* x = nullptr;

    NNMyOp() = default;
    explicit NNMyOp(NNGraph::TensorNode* x_)
        : alpha(1.0), x(x_)
    {
        inputs_ = {x};
    }

    NNGraph::TensorNode* forward(const std::string& output_name) override;
    void backward() const override;
};

NNGraph::TensorNode* my_op(
    NNGraph::TensorNode* x,
    const std::string& output_name);

} // namespace nntile::graph
```

**Key points:**
- Constructor takes **inputs only** (no outputs).
- `forward(output_name)` returns `TensorNode*` (the primary output).
- Use `output()` in backward to get the output tensor (set by forward).

---

## Step 3: Implement forward()

In `src/graph/nn/<op>.cc`:

1. **Validate inputs** — reject null pointers.
2. **Get graph** — `NNGraph& graph = x->graph();` (from any input).
3. **Create output** — `graph.tensor(shape, output_name, dtype, out_requires_grad)`.
4. **Set outputs_** — `outputs_ = {output};` (required for `register_op` and backward).
5. **Add tensor graph ops** — call tensor free functions (e.g., `graph::add`, `graph::gelu`) with `x->data()`, `output->data()`, etc.
6. **Return output** — `return output;`

Example (from `add.cc`):

```cpp
NNGraph::TensorNode* NNAddOp::forward(const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
        throw std::invalid_argument("NNAddOp::forward: x, y must be non-null");

    NNGraph& graph = x->graph();
    bool out_requires_grad = any_input_requires_grad({x, y});
    NNGraph::TensorNode* z = graph.tensor(
        x->shape(), output_name, x->dtype(), out_requires_grad);

    outputs_ = {z};
    graph::add(alpha, x->data(), beta, y->data(), z->data());
    return z;
}
```

**Temporaries:** For ops that need intermediate buffers (e.g., softmax), create them in `forward()` and store in `buffers_`:

```cpp
NNGraph::TensorNode* max_vals = graph.tensor(...);
NNGraph::TensorNode* sum_exp = graph.tensor(...);
buffers_ = {max_vals, sum_exp};
// ... add tensor ops ...
```

---

## Step 4: Implement backward()

In `backward()`, you receive gradients via `output()->grad()` and propagate to inputs.

1. **Get grad_out** — `NNGraph::TensorNode* grad_out = output()->grad();`
2. **Early exit** — if `grad_out == nullptr`, return (no gradient to propagate).
3. **For each input that requires grad:**
   - `graph.get_or_create_grad(input, input->name() + "_grad")` to get/create the gradient tensor
   - Call tensor ops on `grad_out->data()` and `grad_x->data()` to accumulate gradients

**Accumulation rules:**

- **First gradient** — use `graph.is_first_grad(input)` to decide whether to clear or accumulate. If first, use `beta=0` (or `clear`) so the result is overwritten; otherwise use `beta=1` to accumulate.
- **In-place ops** — use tensor in-place ops (e.g., `add_inplace`) to add into the gradient tensor.

Example (from `add.cc`):

```cpp
void NNAddOp::backward() const
{
    NNGraph& graph = x->graph();
    NNGraph::TensorNode* grad_out = output()->grad();
    if(grad_out == nullptr)
        return;

    if(x != nullptr && x->requires_grad()) {
        bool first = graph.is_first_grad(x);
        NNGraph::TensorNode* grad_x =
            graph.get_or_create_grad(x, x->name() + "_grad");
        Scalar grad_beta = first ? 0.0 : 1.0;
        graph::add_inplace(alpha, grad_out->data(), grad_beta, grad_x->data());
    }
    if(y != nullptr && y->requires_grad()) {
        bool first = graph.is_first_grad(y);
        NNGraph::TensorNode* grad_y =
            graph.get_or_create_grad(y, y->name() + "_grad");
        Scalar grad_beta = first ? 0.0 : 1.0;
        graph::add_inplace(beta, grad_out->data(), grad_beta, grad_y->data());
    }
}
```

Example with first-grad handling (from `gelu.cc`):

```cpp
if(x != nullptr && x->requires_grad()) {
    bool first = graph.is_first_grad(x);
    NNGraph::TensorNode* grad_x =
        graph.get_or_create_grad(x, x->name() + "_grad");
    if(first)
        graph::clear(grad_x->data());
    graph::gelu_backward(x->data(), grad_out->data(), grad_x->data());
}
```

---

## Step 5: Implement the Free Function

The free function creates the op, calls `forward()`, and registers it:

```cpp
NNGraph::TensorNode* add(
    Scalar alpha,
    NNGraph::TensorNode* x,
    Scalar beta,
    NNGraph::TensorNode* y,
    const std::string& output_name)
{
    if(x == nullptr || y == nullptr)
        throw std::invalid_argument("add: x and y must be non-null");

    NNGraph& graph = x->graph();
    auto op = std::make_shared<NNAddOp>(x, y, alpha, beta);
    NNGraph::TensorNode* z = op->forward(output_name);
    register_op(graph, std::move(op));
    return z;
}
```

**Pattern:** `op = make_op(inputs...); output = op->forward(output_name); register_op(graph, op); return output;`

---

## Step 6: Register in graph_ops.hh

Add an include for your new op in `include/nntile/graph/nn/graph_ops.hh`:

```cpp
#include <nntile/graph/nn_graph/my_op.hh>
```

---

## Step 7: Add to Build System

Ensure your new `.cc` file is added to the build (e.g., in `src/CMakeLists.txt`).

---

## Summary Checklist

- [ ] Tensor ops exist and operate on `TensorGraph::TensorNode*` only
- [ ] Header: struct with inputs only, `forward(output_name)` returns `TensorNode*`, `backward()` const
- [ ] Forward: validate inputs, create output via `graph.tensor()`, set `outputs_`, add tensor ops, return output
- [ ] Backward: use `output()->grad()`, propagate via tensor ops to `grad_x->data()`, etc.
- [ ] Handle `is_first_grad` when accumulating gradients
- [ ] Free function: create op with inputs, call `op->forward(output_name)`, `register_op`, return output
- [ ] Include in `nn/graph_ops.hh`
- [ ] Add to build system

---

## Reference: Existing Autograd Functions

| Op        | Header              | Tensor ops used (forward / backward)                    |
|-----------|---------------------|--------------------------------------------------------|
| Add       | `add.hh`            | `add` / `add_inplace`                                  |
| Gelu      | `gelu.hh`           | `gelu` / `clear`, `gelu_backward`                      |
| Gemm      | `gemm.hh`           | `clear`, `gemm` / `gemm` (for grad_A, grad_B)          |
| AddFiber  | `add_fiber.hh`      | `add_fiber` / `sum_fiber`, `add_inplace`               |
| SumFiber  | `sum_fiber.hh`      | `clear`, `sum_fiber` / `add_fiber_inplace`             |
