# Autograd API: Callable Functors and Modules

## General API for Autograd Functions

### AutogradFunction Base Class (PyTorch-like)

Base class handles OpNode creation, producer wiring, and requires_grad:

```cpp
struct ForwardResult {
    std::vector<LogicalGraph::TensorNode*> outputs;
    std::vector<TensorNode*> inputs;
    OpAttrs attrs;
};

template<typename Derived>
struct AutogradFunction : AutogradFunctionBase {
    // operator() does ALL bookkeeping: any_input_requires_grad, graph.tensor, register_op
    template<typename... Args>
    TensorNode* operator()(Args&&... args) const;
};

// User implements only:
//   static ForwardResult build_forward(...);  // logical ops only
//   static void build_backward(const OpNode* op);
```
- **Creates OpNode only when** GradMode enabled AND any input requires grad.
  When GradMode disabled (e.g. inside module forward with custom backward),
  no OpNode is created for small ops – only the module's wrap_with_module_op
  creates one OpNode for the whole forward.
- **Producer and backward_fn** only when GradMode enabled AND any input requires grad
  (gradients propagate to inputs, not outputs)
- **Multi-output** supported via `std::vector<TensorNode*> outputs`

### Autograd Functors (Add, Gemm, AddFiber, Gelu, SumFiber)

```cpp
struct Add : AutogradFunction<Add> {
    static ForwardResult build_forward(...);
    static void build_backward(const OpNode* op);
};
```

- **operator()**: base does all bookkeeping (requires_grad, graph.tensor, register_op).
- **build_forward**: user does only logical ops, returns `ForwardResult{out, inputs, attrs}`.
- **build_backward**: user does backward logical ops.
- **Callable**: `Add()(alpha, x, beta, y, "z")` or free function `add(...)`

**build_forward** – logical ops only (bookkeeping in operator()):

```cpp
ForwardResult Add::build_forward(...) {
    LogicalGraph::TensorNode& z_data = add(alpha, x->data(), beta, y->data(), output_name);
    return {{&z_data}, {x, y}, BinaryOpAttrs{alpha, beta}};
}

// Multi-output example:
ForwardResult MyOp::build_forward(...) {
    auto& out1 = logical_op1(...);
    auto& out2 = logical_op2(...);
    return {{&out1, &out2}, {x}, MyAttrs{}};
}
```

### Modules (CRTP, no fixed API)

```cpp
template<typename Derived>
class Module : public ModuleBase {
    template<typename... Args>
    decltype(auto) operator()(Args&&... args);  // forwards to build_forward(Args...)
};
```

- **build_forward** can have any signature: `(TensorNode&)`, `(Q&, K&, V&, mask*)`, etc.
- **operator()** forwards whatever args to `build_forward` – no fixed API.
- **No custom backward**: implement `build_forward` only.
- **Custom backward**: implement `build_forward`, `backward_inputs()`, `build_backward(op)`.

### Usage Examples

```cpp
// Functors (callable or free function)
graph::Add add_fn;
auto* z = add_fn(1.0, x, 1.0, y, "z");  // or add(1.0, x, 1.0, y, "z")

// Modules
Linear linear(graph, "linear", 8, 4, true);
auto& out = linear(input);  // or linear.build_forward(input)
```
