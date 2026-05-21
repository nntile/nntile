# Python package

The installable package lives under [`wrappers/python/nntile/`](../../wrappers/python/nntile/).
After building, use the copy in `build/wrappers/python/` on `PYTHONPATH`.

## Submodules

| Module | Purpose |
|--------|---------|
| `nntile.tensor` | Tensor types, traits, constructors, re-exports all `functions` |
| `nntile.functions` | Low-level async tensor ops (wrappers around C++) |
| `nntile.layer` | Differentiable layers (`BaseLayer` hierarchy) |
| `nntile.model` | Full models (`BaseModel` hierarchy) |
| `nntile.loss` | `CrossEntropy`, `Frob` |
| `nntile.optimizer` | `Adam`, `AdamW`, `SGD`, `Empty` |
| `nntile.pipeline` | Training loop with optional SGOC graph capture |
| `nntile.inference` | Sync/async LLM engines and simple API server — see [inference/README.md](../inference/README.md) |
| `nntile.graph` | Graph API (**work in progress** — see [graph-wip.md](../graph-wip.md)) |

Core runtime: `nntile.Context`, `nntile.starpu`, `nntile.tile`, `TransOp` / `trans` / `notrans`.

## Runtime initialization

```python
import nntile

context = nntile.Context(
    ncpu=-1,      # -1 = all CPUs
    ncuda=-1,     # -1 = all CUDA devices
    ooc=0,
    logger=False,
    verbose=0,
)
if use_cuda:
    context.restrict_cuda()
else:
    context.restrict_cpu()

# ... work ...

nntile.starpu.wait_for_all()
context.shutdown()
```

See [`gpt2_custom_training.py`](../../wrappers/python/examples/gpt2_custom_training.py)
for a full example with tiling and StarPU env vars.

## TensorMoments

Training uses `TensorMoments`: a tensor `value`, optional `grad` tensor, and
`grad_required` flag. Layers and models wire activations and parameters as
`TensorMoments` lists.

## Async execution

Most ops in `nntile.functions` are `*_async` and enqueue StarPU tasks. Call
`nntile.starpu.wait_for_all()` (or layer/model `forward()` / `backward()` which
wait internally) before reading results on the host.

## Documentation map

- [tensors.md](tensors.md) — dtypes, `TensorTraits`, constructors, `from_array` / `to_array`
- [functions.md](functions.md) — operation reference
- [layers.md](layers.md) — layer catalog and API
- [models.md](models.md) — model catalog and `from_pretrained`
- [training.md](training.md) — pipeline, examples, notebooks
- [data-preparation.md](data-preparation.md) — dataset scripts
- [inference/README.md](../inference/README.md) — generation, **nntile_gateway**, **nntile_tgbot**

## See also

- [build/README.md](../build/README.md) — build and `PYTHONPATH`
- [sgoc/README.md](../sgoc/README.md) — scheduler integration in `Pipeline`
