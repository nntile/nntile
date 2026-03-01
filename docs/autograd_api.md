# Autograd API: Callable Functors and Modules

## General API for Autograd Functions

### AutogradFunction Base Class (PyTorch-like)

Base class handles OpNode creation, producer wiring, and requires_grad:

```cpp
struct AutogradFunctionBase {
    static void register_op(graph, inputs, outputs, attrs, backward_fn);
    static void register_op(graph, inputs, output, attrs, backward_fn);
    static bool any_input_requires_grad(inputs);
};

template<typename Derived>
struct AutogradFunction : AutogradFunctionBase {
    template<typename... Args>
    auto operator()(Args&&... args) const;  // forwards to Derived::build_forward
};
```
- **Always creates OpNode** (via create_op)
- **Producer and backward_fn** only when GradMode enabled AND any input requires grad
  (gradients propagate to inputs, not outputs)
- **Multi-output** supported via `std::vector<TensorNode*> outputs`

### Autograd Functors (Add, Gemm, AddFiber, Gelu, SumFiber)

```cpp
struct Add : AutogradFunction<Add> {
    static TensorNode* build_forward(...);
    static void build_backward(const OpNode* op);
};
```

- **operator()**: base forwards to `build_forward`. User implements only `build_forward()` and `build_backward()`.
- **Callable**: `Add()(alpha, x, beta, y, "z")` or free function `add(...)`
- **OpNode**: base's `register_op()` creates OpNode and sets producer when GradMode enabled

**build_forward** – direct pattern (no wrappers):

```cpp
TensorNode* Add::build_forward(...) {
    LogicalGraph::TensorNode& z_data = add(alpha, x->data(), beta, y->data(), output_name);
    bool out_requires_grad = any_input_requires_grad({x, y});
    TensorNode* z = graph.tensor(z_data, out_requires_grad);
    register_op(graph, {x, y}, z, BinaryOpAttrs{alpha, beta},
                [](const OpNode* op) { Add::build_backward(op); });
    return z;
}
```

### Modules

```cpp
// No build_backward: uses functors, each appears as OpNode
TensorNode& operator()(TensorNode& input);

// With build_backward: single module OpNode (uses wrap_forward)
TensorNode& operator()(TensorNode& input);
```

- **Callable**: `linear(input)` or `mlp(input)` instead of `build_forward(input)`
- **No override**: each module implements `operator()` directly with its signature
- **Custom backward**: call `wrap_forward(input, forward_fn, inputs_fn, backward_fn)` helper

### Module Helper (no override)

```cpp
// Protected helper for modules with build_backward
TensorNode& wrap_forward(
    TensorNode& input,
    std::function<TensorNode&(TensorNode&)> forward_fn,
    std::function<std::vector<TensorNode*>()> inputs_fn,
    std::function<void(const OpNode*)> backward_fn);
```

Modules with custom backward use this; no need to override `forward_impl` or `has_custom_backward`.

### Usage Examples

```cpp
// Functors (callable or free function)
graph::Add add_fn;
auto* z = add_fn(1.0, x, 1.0, y, "z");  // or add(1.0, x, 1.0, y, "z")

// Modules
Linear linear(graph, "linear", 8, 4, true);
auto& out = linear(input);  // or linear.build_forward(input)
```
