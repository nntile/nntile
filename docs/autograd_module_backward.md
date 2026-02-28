# Module vs Functor OpNodes (PyTorch-like Pattern)

## Two Modes

1. **Module provides only build_forward() (no build_backward())**:
   - Gradients come from each functor used in build_forward (gemm, add_fiber, gelu, etc.)
   - Each functor appears as an `NNGraph::OpNode`
   - TensorNode's producer is set to these small OpNodes
   - Example: Linear, Gelu – use autograd functors; output's producer is the last functor (add_fiber or gelu)

2. **Module provides build_backward()**:
   - Only one module-wise OpNode appears in the NNGraph
   - Forward runs in `GradMode::Guard` (inner functors don't set producer)
   - Output is wrapped with `wrap_with_module_op`; module is the producer
   - Example: LinearManual – custom backward, single OpNode for the whole module

### Module Forward API

- `Module::forward(input)` – when `has_custom_backward()` is false: calls `forward_impl` (GradMode enabled, functors set producer)
- When `has_custom_backward()` is true: `GradMode::Guard`, `forward_impl`, `wrap_with_module_op`
- `forward_impl(input)` – override to implement forward
- `build_backward(op)` – override only when module provides custom backward
- `backward_inputs()` – override for `wrap_with_module_op` (when has_custom_backward)
