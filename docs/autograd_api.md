# Autograd API: Callable Functors and Modules

## General API for Autograd Functions

### AutogradFunction Base Class (PyTorch-like)

Base class handles OpNode creation, producer wiring, and requires_grad:

```cpp
struct AutogradFunction {
    // Register OpNode and set producer when GradMode enabled
    static void register_op(graph, inputs, output, attrs, backward_fn);
    // Output requires_grad = any input requires grad
    static bool any_input_requires_grad(inputs);
};
```

Derived functors (Add, Gemm, etc.) inherit from AutogradFunction. Their
`build_forward` does the logical op, creates the output tensor, then calls
`register_op()`. The base handles GradMode check, create_op, set_producer.

### Autograd Functors (Add, Gemm, AddFiber, Gelu, SumFiber)

```cpp
struct Add : AutogradFunction {
    TensorNode* operator()(...) const;
    static TensorNode* build_forward(...);
    static void build_backward(const OpNode* op);
};
```

- **Callable**: `Add()(alpha, x, beta, y, "z")` or free function `add(...)`
- **Backward**: static `build_backward(op)` invoked by `output.backward()`
- **OpNode**: base's `register_op()` creates OpNode and sets producer when GradMode enabled

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
