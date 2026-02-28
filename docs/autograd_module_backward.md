# Module as OpNode (PyTorch-like Pattern)

## How PyTorch Does It

PyTorch supports backward at two levels:

1. **Small functions** (add, matmul, etc.): each creates a node in the autograd graph with its backward.
2. **Modules**: when using `torch.autograd.Function`, a module appears as a single node – its forward is wrapped in a Function whose `backward()` is the custom gradient.

## NNTile Implementation

We mimic both levels:

1. **Small functions** (add, gemm, add_fiber, gelu, sum_fiber): each creates an `NNGraph::OpNode` when `GradMode` is enabled.
2. **Modules** (Linear, Gelu): each appears as **one** `NNGraph::OpNode` via `Module::forward()`.

### Module Forward API

- `Module::forward(input)` – main entry point. When `has_custom_backward()` is true:
  1. Runs `forward_impl` inside `GradMode::Guard` (inner autograd ops don't set producer)
  2. Wraps output with `wrap_with_module_op` (single OpNode for the whole module)
- `forward_impl(input)` – override to implement forward
- `build_backward(op)` – override for custom gradient
- `backward_inputs()` – override to provide inputs for the module OpNode

### Modules as OpNodes

Linear and Gelu use `has_custom_backward() = true`. When you call `linear.build_forward(input)` (or `linear.forward(input)`), the output has exactly **one** producer – the Linear module's OpNode. Same for Gelu. Mlp chains them: `fc1.forward() -> gelu.forward() -> fc2.forward()`, so the graph has three OpNodes (one per submodule).

### Example

```cpp
Linear linear(graph, "linear", 8, 4, true);
auto& output = linear.build_forward(input);  // or linear.forward(input)
// output.has_producer() == true – one OpNode for the whole Linear
output.backward();  // invokes Linear::build_backward
```
