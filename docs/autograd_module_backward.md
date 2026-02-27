# Module Custom Backward (PyTorch-like Pattern)

## How PyTorch Does It

PyTorch's `nn.Module` does **not** use `torch.no_grad()` when it has a custom backward. Instead:

1. **Custom backward** is implemented via `torch.autograd.Function` – you wrap your forward in a custom Function whose `backward()` is your custom gradient logic.
2. **`torch.no_grad()`** disables gradient recording: ops inside don't build an autograd graph; outputs have `requires_grad=False`.
3. **`Function.forward`** runs the computation; when called in grad-enabled context, the Function becomes the only node in the graph (inner ops either run with no_grad or the output's `grad_fn` is overwritten to point to the Function).

## NNTile Implementation

We implement the pattern via:

1. **`GradMode`** – thread-local flag. When disabled (`GradMode::Guard`), autograd ops (add, gemm, add_fiber, gelu, sum_fiber) add to the logical graph but **do not** create `OpNode` or set `producer` on outputs.
2. **`wrap_with_module_op`** – attaches a single `OpNode` with custom `backward_fn` to an output that has no producer.

### Pattern for Modules with Custom Backward

```cpp
// 1. Run forward in no_grad – inner ops don't register producer
{
    GradMode::Guard g;
    output_tensor_ = graph::gemm(...);
    if (bias_tensor_) {
        output_tensor_ = graph::add_fiber(...);
    }
}

// 2. Wrap output with module op – our build_backward overrides autograd
graph_.wrap_with_module_op(
    {&input, weight_tensor_, bias_tensor_},
    output_tensor_,
    [this](const OpNode* op) { build_backward(op); });
```

When `output.backward()` is called, the traversal finds exactly one producer (the module op), and invokes `build_backward`.

### Example

See `LinearManual` in `include/nntile/module/linear_manual.hh` and `examples/linear_manual_example.cc`.
