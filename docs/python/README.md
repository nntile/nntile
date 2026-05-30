# Python package (graph API)

The installable package lives under [`python/`](../../python/).

Build `libnntile` and the `nntile` extension with CMake (`BUILD_PYTHON_WRAPPERS=ON`), then set:

```bash
export PYTHONPATH="$(pwd)/build/python"
export LD_LIBRARY_PATH="$(pwd)/build/nntile:/opt/starpu/lib:${LD_LIBRARY_PATH}"
```

See [python/README.md](../../python/README.md) and [python/examples/README.md](../../python/examples/README.md).

## Public API (v1)

- `Context` — StarPU runtime init
- `NNGraph`, `nntile.nn` — graph construction and autograd ops
- `TensorGraph`, `TileGraph`, `Runtime` / `GraphRuntime` — lowering and execution
- `module.Mlp`, `Gpt2Causal`, `AdamW`, dataset helpers — as used by examples

Legacy eager `wrappers/python` has been removed on the `graph_api` branch.
