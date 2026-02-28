# Autograd API: Callable Functors and Modules

## General API for Autograd Functions

Autograd functors and modules are **callable** via `operator()`. No override chain.

### Autograd Functors (Add, Gemm, AddFiber, Gelu, SumFiber)

```cpp
struct Add {
    TensorNode* operator()(Scalar alpha, TensorNode* x, Scalar beta,
                          TensorNode* y, const std::string& output_name) const;
    static TensorNode* build_forward(...);
    static void build_backward(const OpNode* op);
};
```

- **Callable**: `Add()(alpha, x, beta, y, "z")` or free function `add(alpha, x, beta, y, "z")`
- **Backward**: static `build_backward(op)` invoked by `output.backward()`
- **OpNode**: created when GradMode enabled; producer set on output

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
