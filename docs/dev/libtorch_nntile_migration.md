# Migration note: NNGraph → libtorch_nntile (complete)

**Status:** complete on `graph_api`  
**Related:** [torch_nntile_tensor_architecture.md](torch_nntile_tensor_architecture.md),
[graph_compiler_on_design.md](graph_compiler_on_design.md),
[../graph.md](../graph.md)

NNGraph autograd, `nntile::module` / `model` / `optim`, `python/nntile`, and
NNGraph C++ examples were removed. The product path is:

```text
C++ / Python apps
       │
 libtorch_nntile  (ATen PrivateUse1 + custom autograd + models)
       │
 libnntile        (TensorGraph → TileGraph → Runtime → StarPU → kernels)
```

| Library | Role |
|---------|------|
| **libnntile** | TensorGraph stack |
| **libtorch_nntile** | PrivateUse1 + models + custom ops |

Native C++ models use cyclic `model_transpose` for attention layout. They must
**not** call `swap_two_axes` / `aten::transpose` (HF bridge only). See
`.github/scripts/check-model-no-swap-axes.sh`.

## Deferred debt

### Untied embedding / LM-head weights

Local Python and C++ models keep **independent** embedding and output-projection
parameters. `tie_word_embeddings` (and BERT MLM decoder↔word-embedding sharing)
is **not** implemented as shared `Parameter` storage.

- Parity tests construct **untied** HF references so forward and backward grads
  match.
- HF loaders copy head weights by value into independent tensors.
- Exporters keep heads untied in the written state_dict.

Revisit when PrivateUse1 parameter aliasing is solid on `device=nntile` and
training needs true tied grads.

See also the debt table in
[torch_nntile_tensor_architecture.md](torch_nntile_tensor_architecture.md#technical-debt-future-fixes).
