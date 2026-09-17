# TensorGraph execution backend

**Status:** implementation detail of torch_nntile  
**Product entry:** [torch_nntile](torch_nntile.md) (`device="nntile"`, PyTorch autograd)  
**C++ stack:** [cpp/README.md](cpp/README.md)

torch_nntile does **not** ship its own autograd. Forward and backward are
PyTorch's; recorded ATen / classic ops append to a deferred **TensorGraph**,
lower to **TileGraph**, and run through **Runtime** (StarPU). There is no
separate per-op “eager graph” path and no standalone NNGraph /
`python/nntile` bindings (removed).

```text
PyTorch autograd (forward / backward)
     │  record ops
     ▼
TensorGraph
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
| **libtorch_nntile** | LibTorch PrivateUse1 `device=nntile` + C++ models |
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

Python helpers: `torch_nntile.compile_graph()`, `run()`, `wait()`.
Local lower (`compile_graph()`, legacy `execute()`, host `.to("cpu")`
auto-flush) requires `NNTILE_ENABLE_LOCAL_COMPILER=1` (off by default).

`nntile::RemoteExecutionDriver` submits an already-lowered `TileGraph`
over `NNTILE_DRIVER_SOCKET` (mode 0600). StarPU stays in
`ExecutionDaemon`. The wire allowlist is `TILE_ADD_INPLACE`,
`TILE_FILL`, and `TILE_RELU`. Other tile `op_name` values fail closed
(`UnknownOp`). Kernels must not have that socket path.

`nntile::tensor::encode_phase` / `decode_phase` emit Flush `PhaseIR`
JSON (`{nodes, ops}`). v1 allowlist: `TORCH_UNARY`, `TORCH_BINARY`,
`TORCH_TERNARY`, `GATHER`, `SCATTER`, `UNREGISTER`. Torch-native
aten kind is `attrs.kind`. Last `TensorRef` drop records
`UNREGISTER`. Unknown `op_name` fails closed (`UnknownOp`).

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
