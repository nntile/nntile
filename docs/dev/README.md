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
| [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md) | PrivateUse1 aten op list; CUDA registration policy; tiling mix rules |
| [torch_nntile_classic_kernels.md](torch_nntile_classic_kernels.md) | Classic kernels; C++ `torch_nntile::models` (not HF ports) |
| [torch_starpu_kernels.md](torch_starpu_kernels.md) | Untiled torch kernels as StarPU codelets; match-CUDA dispatch guide |
| [graph_compiler_on_design.md](graph_compiler_on_design.md) | O(N) incremental TensorGraph → TileGraph → Runtime |
| [execution_json_schema.md](execution_json_schema.md) | Optional static `execution.json` schedule |
| [graph_compile_perf_mnist.md](graph_compile_perf_mnist.md) | Compile-perf measurements (MNIST dry-run) |
| [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md) | Tiny HF smokes: CPU vs nntile loss/wall table |
| [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md) | Tiny CNN smokes (LeNet / ResNet): CPU vs nntile |
| [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md) | Tiny Diffusers DiT: CPU vs nntile loss/wall |
| [dit_hf_overhead_scale.md](dit_hf_overhead_scale.md) | DiT HF 10-step CUDA vs nntile overhead ladder (VRAM-matched to Llama) |
| [torch_native_middle_cpu_vs_nntile.md](torch_native_middle_cpu_vs_nntile.md) | Middle (~1 min) torch-native CPU vs nntile |
| [reproducibility.md](reproducibility.md) | Single-core CPU / GPU overhead measurement protocol |
| [cuda_vs_nntile_2gb.md](cuda_vs_nntile_2gb.md) | ≥2 GiB CUDA vs nntile GPU table (separate processes) |
| [gpt2_hf_overhead_scale.md](gpt2_hf_overhead_scale.md) | GPT-2 HF 10-step overhead; `seq_len = n_embd/2` (XS/S/M/L) |
| [cuda_wheel_single_nvidia_stack_plan.md](cuda_wheel_single_nvidia_stack_plan.md) | CUDA wheel disk / single NVIDIA stack (infra) |

## Historical

| Doc | Topic |
|-----|-------|
| [libtorch_nntile_migration.md](libtorch_nntile_migration.md) | Completed NNGraph → libtorch_nntile migration; deferred debt |

Obsolete NNGraph / eager-graph plans (autograd guide, `BaseOpNode` sketch,
static GPT-2 agent checklist, linear-bias plan, root `graph.md`) were removed.
Do not reintroduce those paths.
