# NNTile Graph API

**Status:** work in progress (primary product path on `graph_api`)  
**Product entry:** [torch_nntile](torch_nntile.md) (`device="nntile"`)  
**C++ stack:** [cpp/README.md](cpp/README.md)

NNTile training runs through a **deferred** TensorGraph stack. Ops are recorded
into a shared graph, lowered to tiles, compiled, then submitted to StarPU.
There is no separate per-op “eager graph” path and no standalone NNGraph /
`python/nntile` bindings (removed).

```text
record ops → TensorGraph
     │  seal_phase + append_tensor_graph_phase
     ▼
TileGraph
     │  Runtime::compile + execute / execute_range
     ▼
Runtime → StarPU → kernels
```

## Libraries

| Library | Role |
|---------|------|
| **libnntile** | TensorGraph → TileGraph → Runtime (StarPU) |
| **libtorch_nntile** | LibTorch PrivateUse1 `device=nntile` + models |
| **torch_nntile** (Python) | Pip wheel / bindings over libtorch_nntile |

Apps (Python or C++) go through **libtorch_nntile**. Autograd is PyTorch’s;
libnntile holds the compute IR and executor.

## Execution model

1. **Record** — tensor ops append to one session-scoped `TensorGraph`.
2. **Compile** — seal the pending phase, lower to `TileGraph`, run
   `Runtime::compile()` (DCE / allocate).
3. **Run** — `Runtime::execute()` / `execute_range()` submits StarPU tasks
   (async).
4. **Wait** — `Runtime::wait()` (or host readout such as `.to("cpu")`) joins
   workers.

Python helpers: `torch_nntile.compile_graph()`, `run()`, `wait()`. Legacy
`execute()` is compile + run. Host `.to("cpu")` may auto-flush pending work.

Incremental compile aims for **O(work this call)** complexity — see
[dev/graph_compiler_on_design.md](dev/graph_compiler_on_design.md).

## Static tiling and schedules (optional)

| Artifact | Role |
|----------|------|
| Axis-group tiling / `tiling.json` | Tile geometry (`AxisDescriptor`, `tiling_spec_json.hh`) |
| `execution.json` | Optional static worker assignment for tile ops |

`Runtime::compile()` does **not** invent a schedule. If no schedule is set,
StarPU picks workers (`starpu_worker_hint = -1`). Round-robin and affinity-batch
generators live in `nntile/include/nntile/core/execution_schedule.hh`. Schema:
[dev/execution_json_schema.md](dev/execution_json_schema.md).

## Where to read next

| Doc | Contents |
|-----|----------|
| [torch_nntile.md](torch_nntile.md) | User-facing `device=nntile` API, wheels, models |
| [cpp/README.md](cpp/README.md) | libnntile layer map |
| [dev/README.md](dev/README.md) | Design notes index |
| [dev/torch_nntile_tensor_architecture.md](dev/torch_nntile_tensor_architecture.md) | `TensorRef`, I/O, INVALIDATE, session memory |
| [dev/graph_compiler_on_design.md](dev/graph_compiler_on_design.md) | O(N) incremental compile invariants |
| [dev/execution_json_schema.md](dev/execution_json_schema.md) | `execution.json` contract |

## Removed (do not revive)

- **NNGraph** (`include/nntile/nn/`, `CompiledGraph`, graph autograd)
- **`python/nntile`** package and NNGraph pybind
- C++ **`nntile/examples/gpt2_graph_training`** and related NNGraph demos

Training examples live under `torch_nntile/examples/` (DeepReLU, GPT-2, Llama,
…).
