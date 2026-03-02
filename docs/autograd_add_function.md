# How to Add a New Autograd Function

This guide describes how to add a new autograd (differentiable) operation to the NNGraph system. Autograd functions live in `include/nntile/graph/nn_graph/` and `src/graph/nn_graph/`.

**Important:** Autograd functions use `TensorGraph::DataNode` operations for forward and backward. All logic is expressed via the tensor graph API (e.g., `add`, `add_inplace`, `gemm`, `gelu_backward`, etc.). You access the underlying data node from an `NNGraph::TensorNode*` via `.data()` — tensor ops take `TensorGraph::DataNode*` and return `TensorGraph::DataNode*` where applicable.

---

## Overview

Each autograd function consists of:

1. **Header** (`include/nntile/graph/nn_graph/<op>.hh`) — struct with `build_forward` and `backward`, plus a convenience free function
2. **Source** (`src/graph/nn_graph/<op>.cc`) — implementation of forward and backward
3. **Tensor ops** — existing or new operations in `include/nntile/graph/tensor/` that operate on `TensorGraph::DataNode*`

---

## Step 1: Ensure Tensor Ops Exist

Your autograd function must be built from `TensorGraph::DataNode` operations only. Check `include/nntile/graph/tensor/` for existing ops (e.g., `add`, `add_inplace`, `gemm`, `gelu`, `gelu_backward`, `sum_fiber`, `add_fiber`, `add_fiber_inplace`). If you need a new tensor op, add it first in the tensor graph layer.

---

## Step 2: Create the Header

Create `include/nntile/graph/nn_graph/<op>.hh`:

```cpp
#pragma once

#include <string>

#include <nntile/graph/tensor/<op>.hh>
#include <nntile/graph/nn_graph.hh>

namespace nntile::graph
{

//! <Op>: holds params, implements backward(); build_forward creates node.
struct NNMyOp : NNOpBase
{
    // Parameters needed for backward (e.g., alpha, beta)
    Scalar alpha = 1.0;

    void backward(const NNGraph::OpNode* op) override;

    static NNGraph::TensorNode* build_forward(
        NNGraph::TensorNode* x,
        const std::string& output_name);
};

inline NNGraph::TensorNode* my_op(
    NNGraph::TensorNode* x,
    const std::string& output_name)
{
    return NNMyOp::build_forward(x, output_name);
}

} // namespace nntile::graph
```

---

## Step 3: Implement build_forward

In `src/graph/nn_graph/<op>.cc`:

1. **Validate inputs** — reject null pointers.
2. **Get graph** — `NNGraph& graph = x->graph();`
3. **Call tensor op** — use `x->data()` to get `TensorGraph::DataNode*` and pass to the tensor function:
   ```cpp
   TensorGraph::DataNode* y_data = my_tensor_op(x->data(), output_name);
   ```
4. **Create tensor** — `NNGraph::TensorNode* y = graph.tensor(y_data, out_requires_grad);`
5. **Compute `out_requires_grad`** — `bool out_requires_grad = any_input_requires_grad({x, ...});`
6. **Register op** — `register_op(graph, {x, ...}, y, std::make_shared<NNMyOp>(...), {});`

Example (from `add.cc`):

```cpp
NNGraph::TensorNode* NNAddOp::build_forward(
    Scalar alpha,
    NNGraph::TensorNode* x,
    Scalar beta,
    NNGraph::TensorNode* y,
    const std::string& output_name)
{
    if (x == nullptr || y == nullptr)
        throw std::invalid_argument("NNAddOp::build_forward: x and y must be non-null");
    NNGraph& graph = x->graph();
    TensorGraph::DataNode* z_data =
        add(alpha, x->data(), beta, y->data(), output_name);
    bool out_requires_grad = any_input_requires_grad({x, y});
    NNGraph::TensorNode* z = graph.tensor(z_data, out_requires_grad);
    register_op(graph, {x, y}, z,
                std::make_shared<NNAddOp>(alpha, beta), {});
    return z;
}
```

---

## Step 4: Implement backward

In `backward()`, you receive the `OpNode` and must propagate gradients to each input that `requires_grad()`.

1. **Extract context** — `op->output()`, `op->output()->grad()`, `op->inputs()`.
2. **For each input that requires grad:**
   - `graph.get_or_create_grad(input, input->name() + "_grad")` to get/create the gradient tensor
   - Call tensor ops on `grad_out->data()` and `grad_x->data()` (and similar) to accumulate gradients

**Accumulation rules:**

- **First gradient** — use `graph.is_first_grad(input)` to decide whether to clear or accumulate. If first, use `beta=0` (or `clear`) so the result is overwritten; otherwise use `beta=1` to accumulate.
- **In-place ops** — use tensor in-place ops (e.g., `add_inplace`) to add into the gradient tensor.

Example (from `add.cc`):

```cpp
void NNAddOp::backward(const NNGraph::OpNode* op)
{
    NNGraph& graph = op->output()->graph();
    NNGraph::TensorNode* grad_out = op->output()->grad();
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

## Step 5: Register in nn_graph_ops.hh

Add an include for your new op in `include/nntile/graph/nn_graph_ops.hh`:

```cpp
#include <nntile/graph/nn_graph/my_op.hh>
```

---

## Step 6: Add to Build System

Ensure your new `.cc` file is added to the build (e.g., in `src/CMakeLists.txt`).

---

## Summary Checklist

- [ ] Tensor ops exist and operate on `TensorGraph::DataNode*` only
- [ ] Header: struct with `build_forward` and `backward`, plus free function
- [ ] Forward: validate inputs, call tensor op via `x->data()`, create tensor, `register_op`
- [ ] Backward: use only tensor ops on `grad_out->data()`, `grad_x->data()`, etc.
- [ ] Handle `is_first_grad` when accumulating gradients
- [ ] Include in `nn_graph_ops.hh`
- [ ] Add to build system

---

## Reference: Existing Autograd Functions

| Op        | Header              | Tensor ops used (forward / backward)                    |
|-----------|---------------------|----------------------------------------------------------|
| Add       | `add.hh`            | `add` / `add_inplace`                                    |
| Gelu      | `gelu.hh`           | `gelu` / `clear`, `gelu_backward`                       |
| Gemm      | `gemm.hh`           | `gemm` / `gemm` (for grad_A, grad_B)                    |
| AddFiber  | `add_fiber.hh`      | `add_fiber` / `sum_fiber`, `add_inplace`                 |
| SumFiber  | `sum_fiber.hh`      | `clear`, `sum_fiber` / `add_fiber_inplace`               |
