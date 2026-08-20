# PrivateUse1 aten ops on `device=nntile`

**Status:** temporary single-tile (untiled) only  
**Branch:** `graph_api_torch_kernels`  
**Related:** [torch_starpu_kernels.md](torch_starpu_kernels.md),
[torch_nntile_tensor_architecture.md](torch_nntile_tensor_architecture.md)

Under `NNTILE_TORCH_NATIVE_OPS`, each listed compute schema records a
torch-native TensorGraph op that lowers to the **same aten call on
`device=CPU` with no grad** inside a StarPU codelet (see
[torch_starpu_kernels.md](torch_starpu_kernels.md)). Classic NNTile
kernels are not used for compute on this path.

While torch-native StarPU codelets are introduced for untiled tensors,
**axis-group tiling is disabled** for the PrivateUse1 path:

- `torch_nntile.set_axis_group_tiling(...)` raises a C++ `std::runtime_error`.
- `compile_graph` / `execute` rejects any session whose axis groups are
  already tiled.
- Multi-tile Python tests under `torch_nntile/tests/` are skipped; keep
  **untiled** parity/smoke coverage only.

Libnntile C++ TensorGraph tiling tests are unchanged (they do not go through
PrivateUse1).

## Registration policy (match `device=cuda`)

Baseline is how stock PyTorch registers the same schema for CUDA. Full
write-up (layers, dump commands, worked examples):
[torch_starpu_kernels.md — Match device=cuda registration](torch_starpu_kernels.md#match-devicecuda-registration).

| CUDA pattern | What nntile should do |
|--------------|------------------------|
| Device kernel (`RegisterCUDA` / structured `.out`) | PrivateUse1 device impl; Autograd stays on generic VariableType |
| `CompositeImplicitAutograd` (`chunk`, `narrow`, `linear`, `matmul`, …) | **Do not** register PrivateUse1 — let the composite lower to primitives (`as_strided`, `addmm`, `mm`, …) |
| `CompositeExplicitAutograd` shared default (`select.int`, `alias`, …) | Prefer composite unless nntile storage needs a hook (`as_strided` / `alias` for `TensorRef`) |
| AutogradCUDA = VariableType formula (e.g. `rsqrt` → `result.pow(3)`) | **Do not** register AutogradPrivateUse1; implement the formula’s device ops (`pow`) |

### Quick audit

```python
import torch
# Before importing torch_nntile: stock CUDA / composite rows.
print(torch._C._dispatch_dump_table("aten::OP"))
# After import torch_nntile: confirm we did not shadow CompositeImplicit
# unless intentional; AutogradPrivateUse1 should match AutogradCUDA
# (usually VariableType or CompositeImplicit — not a custom kernel).
```

With `requires_grad=True`, `type(out.grad_fn).__name__` on nntile should
match CPU for the same call (`RsqrtBackward0`, `AddmmBackward0`,
`SplitBackward0`, …).

### Worked decisions

| Schema | CUDA | nntile choice |
|--------|------|---------------|
| `rsqrt` | device `.out` + VariableType (`pow(3)`) | PrivateUse1 `rsqrt`(+`.out`); `pow` exp 2/3 via mul; **no** AutogradPrivateUse1 |
| `linear` / `matmul` | CompositeImplicit → `addmm` / `mm` | **No** PrivateUse1; keep `addmm` / `mm` / `bmm` |
| `chunk` / `split` / `narrow` / `select.int` | Composite → views | **No** PrivateUse1; keep `as_strided` (+ `alias`) |
| `as_strided` / `alias` | device / shared composite | PrivateUse1 (keep `TensorRef`) |
| `contiguous` | CompositeImplicit | PrivateUse1 + AutogradPrivateUse1 densify |
| `rms_norm` | CompositeImplicit (`pow` / `mean` / `rsqrt` / `mul`) | **No** PrivateUse1 / AutogradPrivateUse1; match CUDA |

Intentional deviations (nntile storage / StarPU):

- `as_strided` / `alias` — PrivateUse1 so views keep `TensorRef` (CUDA has
  `as_strided_tensorimpl`; composite `alias` would drop our GC binding).
- `contiguous` on **AutogradPrivateUse1** — densify partial covers; CUDA’s
  CompositeImplicit `contiguous` can return a still-strided “contiguous”
  view that is not a full StarPU buffer cover.

`rms_norm` is not an intentional deviation: CUDA leaves it as
CompositeImplicitAutograd, so `device=nntile` does the same and relies on
the primitive ops (`pow` / `mean` / `rsqrt` / `mul`). LayerNorm remains fused
through `native_layer_norm` because PyTorch has that device primitive and
single-pass mean+variance matters.

Known gap vs CUDA view backward: ~~nntile→nntile `_copy_from` rebinds
`TensorRef` (SSA) instead of writing the parent at `storage_offset`.~~
**Fixed:** partial / strided fp32 and int64 destinations use
`TorchKind::CopyIntoView` (RW parent handle, packed view layout). Dense
full-cover copies still SSA-rebind. Bool still host-RMW. Slice / Select /
AsStrided / T Backward now match CPU grads.

## Registered aten schemas

All registrations are `TORCH_LIBRARY_IMPL(aten, PrivateUse1, …)` unless noted.
Sources live under `torch_nntile/csrc/`.

### Storage / views / copies (`nntile_kernels.cpp`)

| Schema |
|--------|
| `empty.memory_format` |
| `empty_strided` |
| `as_strided` |
| `alias` |
| `view` |
| `_unsafe_view` |
| `_reshape_alias` |
| `transpose.int` |
| `t` |
| `permute` |
| `contiguous` |
| `resize_` |
| `_copy_from` |
| `_copy_from_and_resize` |
| `_local_scalar_dense` |
| `fill_.Scalar` |
| `zero_` |
| `ones_like` |
| `set_.source_Tensor` |
| `set_.source_Storage` |
| `set_.source_Storage_storage_offset` |

Also: `contiguous` on **AutogradPrivateUse1**, and a boxed **`cpu_fallback`**
for unregistered ops when `cpu_fallback=True`.

Not registered (CUDA composite → our primitives): `narrow`, `select.int`,
`chunk`, `split` / `split_with_sizes`, `linear`, `matmul`.

### Elementwise / reductions / norms

| File | Schemas |
|------|---------|
| `nntile_add.cpp` | `add.Tensor`, `add.out`, `add_.Tensor`, `add.Scalar`, `add.Scalar_out`, `add_.Scalar` |
| `nntile_mul.cpp` | `mul.Tensor`, `mul.out`, `mul_.Tensor`, `mul.Scalar`, `mul.Scalar_out` |
| `nntile_relu.cpp` | `relu`, `relu.out`, `relu_` |
| `nntile_threshold_backward.cpp` | `threshold_backward` |
| `nntile_silu.cpp` | `silu`, `silu.out`, `silu_` |
| `nntile_silu_backward.cpp` | `silu_backward`, `silu_backward.grad_input` |
| `nntile_gelu.cpp` | `gelu`, `gelu.out`, `gelu_` |
| `nntile_gelu_backward.cpp` | `gelu_backward`, `gelu_backward.grad_input` |
| `nntile_softmax.cpp` | `_softmax`, `_softmax.out` |
| `nntile_softmax_backward.cpp` | `_softmax_backward_data` |
| `nntile_log_softmax.cpp` | `_log_softmax`, `_log_softmax.out`, `_log_softmax_backward_data` |
| `nntile_nll_loss.cpp` | `nll_loss_forward`, `nll_loss_backward` |
| `nntile_hypot.cpp` | `hypot`, `hypot.out` |
| `nntile_sum.cpp` | `sum.IntList_out`, `sum.dim_IntList` |
| `nntile_norm.cpp` | `linalg_vector_norm`, `linalg_vector_norm.out` |
| `nntile_avg_pool2d.cpp` | `avg_pool2d`, `avg_pool2d.out`, `avg_pool2d_backward`, `avg_pool2d_backward.grad_input` |
| `nntile_adaptive_avg_pool2d.cpp` | `_adaptive_avg_pool2d`, `_adaptive_avg_pool2d.out`, `_adaptive_avg_pool2d_backward`, `_adaptive_avg_pool2d_backward.out` |

### Linear algebra

| File | Schemas |
|------|---------|
| `nntile_mm.cpp` | `mm`, `mm.out` |
| `nntile_bmm.cpp` | `bmm`, `bmm.out` |
| `nntile_addmm.cpp` | `addmm`, `addmm.out` |
| `nntile_convolution.cpp` | `convolution_overrideable`, `convolution_backward_overrideable` |

`nntile_linear.cpp` / `nntile_gemm.cpp` keep StarPU helpers but do **not**
register `linear` / `matmul` (CUDA CompositeImplicit → `addmm` / `mm`).

### Norm / embedding / layout

| File | Schemas |
|------|---------|
| `nntile_layer_norm.cpp` | `native_layer_norm`, `native_layer_norm_backward` |
| `nntile_batch_norm.cpp` | `native_batch_norm`, `native_batch_norm_backward` |
| `nntile_embedding.cpp` | `embedding`, `embedding_dense_backward` |
| `nntile_cat.cpp` | `cat`, `cat.out` |
| `nntile_trig.cpp` | `cos`, `sin`, `neg`, `rsqrt`, `exp` (+ `.out`) |
| `nntile_repeat.cpp` | `repeat` |
| `nntile_max_pool2d.cpp` | `max_pool2d_with_indices`, `max_pool2d_with_indices.out`, `max_pool2d_with_indices_backward`, `max_pool2d_with_indices_backward.grad_input` |
| `nntile_upsample2d.cpp` | `upsample_nearest2d`, `upsample_nearest2d.out`, `upsample_nearest2d_backward`, `upsample_nearest2d_backward.grad_input`, `upsample_bilinear2d` (+ `.out` / `_backward` / `.grad_input`) |

`nntile_split.cpp` / `nntile_narrow.cpp` are reference helpers only (no
PrivateUse1 registration).

### Missing fused ops

See the table in
[torch_starpu_kernels.md — Missing fused ops](torch_starpu_kernels.md#missing-fused-ops).
Short list for product planning:

1. **`native_group_norm`** (+ backward) — GroupNorm models (ViT-style).
2. **`native_dropout`** — training.
3. Host leftovers: **`mean`**, general **`pow`/`div`/`where`**. RMSNorm
   correctly uses the composite path, so it currently reaches host `mean`.
4. **`nll_loss2d`** — optional ergonomics for NCHW segmentation CE
   (smokes flatten to 1D `nll_loss` instead).

HF rotary is **not** a fused-aten gap: `apply_rotary_pos_emb` lowers to
`mul` / `add` / views. Classic `_C.rope` is only for hand-written models
when torch-native ops are off.

### SDPA (`nntile_sdpa_aten.cpp`)

| Schema |
|--------|
| `_fused_sdp_choice` |
| `_scaled_dot_product_fused_attention_overrideable` |
| `_scaled_dot_product_fused_attention_overrideable_backward` |

## Skipped tiled tests (temporary)

| Test module | Ops covered when tiled |
|-------------|------------------------|
| `torch_nntile/tests/test_axis_group_tiling.py` | `add`, DeepReLU / CE ingress |
| `torch_nntile/tests/test_simple_matmul_tiling.py` | `matmul` / `mm` |
| `torch_nntile/tests/test_bmm_tiling.py` | `bmm` |

Untiled parity tests (e.g. `test_aten_ops_parity.py`, `test_add_parity.py`,
`test_bmm.py`, …) remain the supported coverage.

## Target direction

Replace per-op `nntile::kernel` compute for the **single-tile** case with
libtorch `*_out` kernels inside StarPU codelets. See
[torch_starpu_kernels.md](torch_starpu_kernels.md).
