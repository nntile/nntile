# Module vs Functor OpNodes (PyTorch-like Pattern)

## Module Forward Only

Modules provide only `build_forward()`. No backward support at module level.

- Gradients come from each autograd functor used in build_forward (gemm, add_fiber, gelu, etc.)
- Each functor appears as an `NNGraph::OpNode`
- TensorNode's producer is set to these small OpNodes
- Example: Linear, Gelu – use autograd functors; output's producer is the last functor (add_fiber or gelu)

### Module Forward API

- `module(input)` – calls `build_forward(input)`
- `build_forward(input)` – implement forward
