# Autograd API: Simple Helpers, No CRTP

## General API for Autograd Functions

### Helpers (register_op, any_input_requires_grad)

No CRTP. User performs all bookkeeping in build_forward:

```cpp
void register_op(
    NNGraph& graph,
    const std::vector<TensorNode*>& inputs,
    const std::vector<TensorNode*>& outputs,  // or single TensorNode*
    std::shared_ptr<void> attrs,  // opaque; only forward/backward know the type
    std::function<void(const OpNode*)> backward_fn,
    const std::vector<TensorNode*>& buffers = {});  // like ctx.save_for_backward

bool any_input_requires_grad(const std::vector<TensorNode*>& inputs);
```

- **Creates OpNode only when** GradMode enabled AND any input requires grad.
- **attrs**: opaque (`std::shared_ptr<void>`); decoupled from LogicalGraph. In `build_backward`: `std::static_pointer_cast<MyAttrs>(op->attrs())`.
- **Multi-input, multi-output** visible via ordinary API of register_op.

### Autograd Functors (Add, Gemm, AddFiber, Gelu, SumFiber)

```cpp
struct Add {
    static TensorNode* build_forward(Scalar alpha, TensorNode* x, Scalar beta,
                                    TensorNode* y, const std::string& output_name);
    static void build_backward(const OpNode* op);
};
```

- **build_forward**: user does logical op + bookkeeping (graph.tensor, register_op).
- **build_backward**: user does backward logical ops.
- **Free function**: `add(alpha, x, beta, y, "z")` calls `Add::build_forward(...)`.

**build_forward** – full bookkeeping:

```cpp
TensorNode* Add::build_forward(...) {
    NNGraph& graph = x->graph();
    LogicalGraph::TensorNode& z_data = add(alpha, x->data(), beta, y->data(), output_name);
    bool out_requires_grad = any_input_requires_grad({x, y});
    TensorNode* z = graph.tensor(z_data, out_requires_grad);
    register_op(graph, {x, y}, z, std::make_shared<BinaryOpAttrs>(BinaryOpAttrs{alpha, beta}),
                [](const OpNode* op) { Add::build_backward(op); }, {});
    return z;
}

// Multi-output: return std::vector<TensorNode*>, pass to register_op
```

### Modules (no CRTP, user does bookkeeping)

All modules inherit `ModuleBase` only. No template.

- **No custom backward**: implement `build_forward`, define `operator()` that calls it.
- **Custom backward**: implement `build_forward`, `backward_inputs()`, `build_backward(op)`.
  Define `operator()` that does: `GradMode::Guard`, `build_forward`, `wrap_with_module_op`.

### Usage Examples

```cpp
// Autograd functors (free function or build_forward)
auto* z = add(1.0, x, 1.0, y, "z");
auto* z2 = Add::build_forward(1.0, x, 1.0, y, "z");

// Modules
Linear linear(graph, "linear", 8, 4, true);
auto& out = linear(input);
```
