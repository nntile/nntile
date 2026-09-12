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

Python helpers: `torch_nntile.compile_graph()`, `run()`, `wait()`. Legacy
`execute()` is compile + run. Host `.to("cpu")` may auto-flush pending work.

Incremental compile aims for **O(work this call)** complexity — see
[dev/graph_compiler_on_design.md](dev/graph_compiler_on_design.md).

## DDP (`ddp()`)

`torch_nntile.ddp(axis="batch")` is a session compile policy. Each
`compile_graph` splits the named axis into `count_execution_workers()`
tiles, lowers the pending suffix, then rewrites weight-grad-like writes
onto phase-local full-sized tiles and `ADD`s into the canonical dest.
Autograd fan-in `aten::add` on a classic-only graph records classic
`ADD` so those residual / QKV writes can be tiled too.
`TileGraph::OpNode::device_hint` pins sharded compute to logical workers
`0..N-1` (`-1` = StarPU dynamic). There is no `tiling.json` /
`execution.json`.

If the user never calls `ddp()`, each tensor stays one tile and StarPU
picks workers.

## Where to read next

| Doc | Contents |
|-----|----------|
| [torch_nntile.md](torch_nntile.md) | User-facing `device=nntile` API, wheels, models |
| [cpp/README.md](cpp/README.md) | libnntile layer map |
| [dev/README.md](dev/README.md) | Design notes index |
| [dev/torch_nntile_tensor_architecture.md](dev/torch_nntile_tensor_architecture.md) | `TensorRef`, I/O, INVALIDATE, session memory |
| [dev/graph_compiler_on_design.md](dev/graph_compiler_on_design.md) | O(N) incremental compile invariants |

## Removed (do not revive)

- **NNGraph** (`include/nntile/nn/`, `CompiledGraph`, graph autograd)
- **`python/nntile`** package and NNGraph pybind
- C++ **`nntile/examples/gpt2_graph_training`** and related NNGraph demos

Training examples live under `torch_nntile/examples/` (DeepReLU, GPT-2, Llama,
…).
