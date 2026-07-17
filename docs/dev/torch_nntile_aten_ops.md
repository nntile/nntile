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

## Registered aten schemas

All registrations are `TORCH_LIBRARY_IMPL(aten, PrivateUse1, …)` unless noted.
Sources live under `torch_nntile/csrc/`.

### Storage / views / copies (`nntile_kernels.cpp`)

| Schema |
|--------|
| `empty.memory_format` |
| `empty_strided` |
| `as_strided` |
| `view` |
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

### Elementwise / reductions / norms

| File | Schemas |
|------|---------|
| `nntile_add.cpp` | `add.Tensor`, `add.out`, `add_.Tensor` |
| `nntile_mul.cpp` | `mul.Tensor`, `mul.out`, `mul_.Tensor`, `mul.Scalar`, `mul.Scalar_out` |
| `nntile_relu.cpp` | `relu`, `relu.out` |
| `nntile_threshold_backward.cpp` | `threshold_backward` |
| `nntile_silu.cpp` | `silu`, `silu.out`, `silu_` |
| `nntile_silu_backward.cpp` | `silu_backward`, `silu_backward.grad_input` |
| `nntile_gelu.cpp` | `gelu`, `gelu.out`, `gelu_` |
| `nntile_gelu_backward.cpp` | `gelu_backward`, `gelu_backward.grad_input` |
| `nntile_softmax.cpp` | `_softmax`, `_softmax.out` |
| `nntile_softmax_backward.cpp` | `_softmax_backward_data` |
| `nntile_hypot.cpp` | `hypot`, `hypot.out` |
| `nntile_sum.cpp` | `sum.IntList_out`, `sum.dim_IntList` |
| `nntile_norm.cpp` | `linalg_vector_norm`, `linalg_vector_norm.out` |

### Linear algebra / linear

| File | Schemas |
|------|---------|
| `nntile_mm.cpp` | `mm`, `mm.out` |
| `nntile_bmm.cpp` | `bmm`, `bmm.out` |
| `nntile_addmm.cpp` | `addmm`, `addmm.out` |
| `nntile_gemm.cpp` | `matmul` |
| `nntile_linear.cpp` | `linear`, `linear.out`, `linear_backward` |

### Norm / embedding / layout

| File | Schemas |
|------|---------|
| `nntile_layer_norm.cpp` | `native_layer_norm`, `native_layer_norm_backward` |
| `nntile_embedding.cpp` | `embedding`, `embedding_dense_backward` |
| `nntile_cat.cpp` | `cat`, `cat.out` |
| `nntile_split.cpp` | `split_with_sizes`, `split`, `split.Tensor`, `chunk` |
| `nntile_narrow.cpp` | `narrow` |
| `nntile_repeat.cpp` | `repeat` |

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
