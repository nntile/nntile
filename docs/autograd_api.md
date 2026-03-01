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

### Modules

```cpp
// Base Module::operator() checks has_custom_backward():
// - If false: just calls build_forward(input) and returns
// - If true: GradMode::Guard, build_forward, wrap_with_module_op(backward_inputs, output, build_backward)

virtual bool has_custom_backward() const { return false; }
virtual std::vector<TensorNode*> backward_inputs() const { return {}; }
virtual TensorNode& build_forward(TensorNode& input);
virtual void build_backward(const OpNode* op) {}
TensorNode& operator()(TensorNode& input);  // in base
```

- **No custom backward** (Linear, Gelu, Mlp): implement only `build_forward`. operator() just calls it.
- **Custom backward** (LinearManual): override `has_custom_backward()=true`, `backward_inputs()`, `build_backward()`. operator() does GradMode::Guard, build_forward, wrap_with_module_op.

### Usage Examples

```cpp
// Functors (callable or free function)
graph::Add add_fn;
auto* z = add_fn(1.0, x, 1.0, y, "z");  // or add(1.0, x, 1.0, y, "z")

// Modules
Linear linear(graph, "linear", 8, 4, true);
auto& out = linear(input);  // or linear.build_forward(input)
```
