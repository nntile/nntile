# NNTile Graph API (work in progress)

NNTile is developing a **graph-based** API for building and training models with
automatic differentiation. This stack is separate from the classic Python API
(`nntile.layer`, `nntile.model`) used in Hugging Face–style training scripts.

## What exists today

- C++ headers under [`include/nntile/`](../include/nntile/) and sources
  under [`nntile/src/`](../nntile/src/)
- Python bindings: `nntile` (see [`wrappers/python/nntile/nntile.cc`](../wrappers/python/nntile/nntile.cc))
- C++ examples under [`examples/`](../examples/) (e.g. autograd and MLP demos)
- Architecture overview: [`graph.md`](../graph.md) at the repository root

## What to use for production training

End-to-end training of BERT, GPT-2, Llama, T5, and similar models uses the
**classic API** documented in [python/README.md](python/README.md) and
[python/training.md](python/training.md). Those scripts do **not** depend on
`nntile.graph`.

## SGOC is not the Graph API

The **SGOC** scheduler ([sgoc/README.md](sgoc/README.md)) is a StarPU scheduling
policy for memory-aware replay of training batches. It is unrelated to
`TensorGraph` / `NNGraph` in the NNTile graph module.
