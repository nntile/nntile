# NNTile Graph API (work in progress)

NNTile is developing a **graph-based** API for building and training models with
automatic differentiation. This stack is separate from the classic Python API
(`nntile.layer`, `nntile.model`) used in Hugging Face–style training scripts.

## What exists today

- C++ headers under [`nntile/include/nntile/`](../nntile/include/nntile/) and sources
  under [`nntile/src/`](../nntile/src/)
- Python bindings: `nntile` (see [`python/nntile/`](../python/nntile/))
- C++ examples under [`nntile/examples/`](../nntile/examples/) (e.g. autograd and MLP demos)
- Architecture overview: [`graph.md`](../graph.md) at the repository root

## What to use for production training

End-to-end training of BERT, GPT-2, Llama, T5, and similar models uses the
**classic API** documented in [python/README.md](python/README.md) and
[python/training.md](python/training.md). Those scripts do **not** depend on
`nntile.graph`.

## Static tiling and task scheduling (GPT-2)

- **Tiling:** `tiling.json` and `--tiling` in `gpt2_graph_training` (see
  `nntile/examples/README.md`).
- **Schedule:** `generate_round_robin_execution_schedule()` writes
  `execution.json` (`--execution-out`); same file loaded with `--execution`
  before `execute()`. `compile()` does not assign workers.
- **Schema:** [dev/execution_json_schema.md](dev/execution_json_schema.md).
- **Roadmap:** [dev/graph_static_execution_plan.md](dev/graph_static_execution_plan.md).
- **E2E:** `nntile/examples/run_gpt2_static_train.sh` (write then load
  `execution.json`). **CI:** `run_gpt2_graph_training_demo.sh` on `graph_api`.
- **Generators:** `generate_round_robin_execution_schedule`,
  `generate_affinity_batch_execution_schedule` (same JSON schema).

## torch_nntile (PyTorch bridge)

[`torch_nntile`](../torch_nntile/) exposes `device="nntile"` for PyTorch models.
When linked against `libnntile`, ops record into `TensorGraph` and run through
`Runtime`. In **graph mode**, you can name axis groups and set tiling before
`torch_nntile.execute()` — the same `AxisDescriptor` model as C++ graph
training.

See [torch_nntile.md](torch_nntile.md) for API reference, MNIST example, and
comparison with C++ `name_gpt2_training_axis_groups` / `tiling.json`.

## SGOC is not the Graph API

The **SGOC** scheduler ([sgoc/README.md](sgoc/README.md)) is a StarPU scheduling
policy for memory-aware replay of training batches. It is unrelated to
`TensorGraph` / `NNGraph` in the NNTile graph module.
