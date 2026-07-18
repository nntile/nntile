# Design notes (`docs/dev`)

Internal notes for the **Graph API** / `torch_nntile` line on `graph_api`.
Start with the product docs if you are new:

- [Graph API overview](../graph.md)
- [torch_nntile](../torch_nntile.md)
- [C++ stack](../cpp/README.md)

## Current

| Doc | Topic |
|-----|-------|
| [torch_nntile_tensor_architecture.md](torch_nntile_tensor_architecture.md) | `TensorRef` / logical tiles, I/O, INVALIDATE, session debt |
| [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md) | PrivateUse1 aten op list; CUDA registration policy; untiled-only |
| [torch_starpu_kernels.md](torch_starpu_kernels.md) | Untiled torch kernels as StarPU codelets; match-CUDA dispatch guide |
| [graph_compiler_on_design.md](graph_compiler_on_design.md) | O(N) incremental TensorGraph → TileGraph → Runtime |
| [execution_json_schema.md](execution_json_schema.md) | Optional static `execution.json` schedule |
| [graph_compile_perf_mnist.md](graph_compile_perf_mnist.md) | Compile-perf measurements (MNIST dry-run) |
| [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md) | Tiny HF smokes: CPU vs nntile loss/wall table |
| [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md) | Tiny CNN smokes (LeNet / ResNet): CPU vs nntile |
| [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md) | Tiny Diffusers DiT: CPU vs nntile loss/wall |
| [cuda_wheel_single_nvidia_stack_plan.md](cuda_wheel_single_nvidia_stack_plan.md) | CUDA wheel disk / single NVIDIA stack (infra) |

## Historical

| Doc | Topic |
|-----|-------|
| [libtorch_nntile_migration.md](libtorch_nntile_migration.md) | Completed NNGraph → libtorch_nntile migration; deferred debt |

Obsolete NNGraph / eager-graph plans (autograd guide, `BaseOpNode` sketch,
static GPT-2 agent checklist, linear-bias plan, root `graph.md`) were removed.
Do not reintroduce those paths.
