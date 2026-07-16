# Migrate to libtorch_nntile (remove NNGraph)

**Status:** Phase 4 complete (NNGraph removed; `libnntile_tensorgraph` → `libnntile`)  
**Branch:** `cursor/libtorch-nntile-migration-160b`  
**Related:** [torch_nntile_tensor_architecture.md](torch_nntile_tensor_architecture.md),
[graph_compiler_on_design.md](graph_compiler_on_design.md)

## Decision

- **C++ entry point:** installable **libtorch_nntile** (LibTorch + PrivateUse1
  `device=nntile`) with `torch::nn::Module` models; Python `torch_nntile`
  binds the same backend.
- **Remove:** NNGraph autograd, `nntile::module` / `model` / `optim`,
  `python/nntile` graph bindings, and NNGraph C++ examples.
- **Rename:** `libnntile` → **`libnntile`** (TensorGraph stack
  only). No separate high-level `libnntile` remains.

```text
C++ / Python apps
       │
 libtorch_nntile  (ATen PrivateUse1 + custom autograd + models)
       │
 libnntile        (former libnntile)
       │
 TensorGraph → TileGraph → Runtime → StarPU → kernels
```

## Op classification

### A — Fully supported torch ops (`TORCH_LIBRARY_IMPL` PrivateUse1)

`add`, `mul`, `hypot`, `sum`, `cat`, `split`/`chunk`, `narrow`,
`mm`/`bmm`/`matmul`/`addmm`, `linear` (+ bias), `relu`/`silu`/`gelu`/
`gelutanh`, softmax, `native_layer_norm`, `embedding`, `transpose`/`t`
(HF layout via ``swap_two_axes`` only - not for native C++ models),
contiguous-preserving `view`/`permute`, `scaled_dot_product_attention`
(eager), `mse_loss` (via custom or ATen path using norm).

### B — Custom autograd on `device=nntile`

| API | Reason |
|-----|--------|
| RoPE | No first-class ATen rotary |
| `add_fiber` / selective fiber ops | Layout without broadcast expand |
| `model_transpose` | Cyclic NNTile attention layout (native models) |
| Fused CE | Class-dim-last fused path |
| Fused SGD / Adam / AdamW | StarPU fused steps |
| RMSNorm | Until ATen path is complete |
| SDPA helpers | Layout wrappers over ATen SDPA |

### C — Internal only (TensorGraph)

`gather`/`scatter`, `invalidate`, `clear`, `copy_intersection`,
`maxsumexp`, `logsumexp`, `total_sum_accum`, most `*_fiber`/`*_slice`
used inside ATen/custom implementations.

## Non-goals

- Tiling productization for transformer examples (later via config/JSON).
  Existing DeepReLU MNIST tiling must keep working.
- TensorGraph / O(N) compile redesign (separate track).
- Flash SDPA / conv2d productization.
- Mixing CUDA torch + `device=nntile` in one process.

## End-state libraries

| Library | Role |
|---------|------|
| **libnntile** | TensorGraph stack (renamed from libnntile) |
| **libtorch_nntile** | PrivateUse1 + models + custom ops |

## Model parity checklist

Native C++ models use cyclic ``model_transpose`` for attention layout.
They must **not** call ``swap_two_axes`` / ``aten::transpose`` (HF bridge
only; low performance). See ``.github/scripts/check-model-no-swap-axes.sh``.

| Model | Python | C++ torch::nn | Example |
|-------|--------|---------------|---------|
| DeepReLU | yes | yes | train_deep_relu_mnist (existing) |
| GPT-2 | yes | yes | train_gpt2* |
| Llama | yes | yes | train_llama |
| GPT-Neo / NeoX | yes | yes | train_gpt_neo* |
| BERT / RoBERTa | yes | yes | train_bert / train_roberta |
| T5 | yes | yes | train_t5 |

## Known debt / deferred work

### Untied embedding / LM-head weights

**Status:** deferred. Local Python and C++ models keep **independent**
embedding and output-projection parameters. `tie_word_embeddings` (and
BERT MLM decoder↔word-embedding sharing) is **not** implemented as shared
`Parameter` storage.

- Parity tests construct **untied** HF references (clone decoder / set
  `tie_word_embeddings=False`) so forward **and backward** grads match.
- HF loaders always **copy** head weights by value into independent
  tensors.
- Exporters keep heads untied in the written state_dict (no
  `hf.tie_weights()`).

Revisit when PrivateUse1 parameter aliasing / shared storage is solid on
`device=nntile` and training needs true tied grads.
