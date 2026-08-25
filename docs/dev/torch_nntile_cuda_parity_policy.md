# torch_nntile namespace and CUDA parity policy

**Status:** active design rule (Aug 2026)  
**Related:** [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md),
[torch_starpu_kernels.md](torch_starpu_kernels.md)

## Goal

`torch.nn` / `torch.nn.functional` on `device=nntile` must behave like
`device=cuda`: same dispatch decomposition, same autograd nodes, same numerics
(up to benign FP reordering). NNTile-specific tiled / hand-written kernels live
under **`torch_nntile.nn.functional`** (alias: ``torch_nntile.kernels`` for
classic builds). Do **not** monkey-patch stock ``torch.nn.functional``.

## Three layers

| Layer | API | Role |
|-------|-----|------|
| **Stock PyTorch** | `torch.nn`, `F.*`, `tensor` ops on `device=nntile` | Same formulas and autograd as CUDA. PrivateUse1 hooks exist only to **record** ops into TensorGraph and **execute** the matching ATen kernel on StarPU/CUDA — no custom math, no forced `.contiguous()` unless CUDA would too. |
| **Device runtime** | `torch_nntile.init_context`, `compile_graph`, `run`, `wait` | Deferred execution / StarPU scheduling. Not a substitute for `aten`. |
| **NNTile kernels** | `torch_nntile.nn.functional.*` (classic builds) | Tiled GEMM, RoPE interleaved pairs, sum-slice reductions. **Never** registered on `aten::` in torch-native builds. |

## What may stay on `aten::PrivateUse1`

1. **Storage / views / I/O** — `empty`, `as_strided`, `alias`, `_copy_from`,
   `contiguous` (densify partial covers), `_local_scalar_dense`, etc.
2. **CUDA device primitives** where PyTorch registers a real device kernel on
   CUDA (`native_layer_norm`, `mm`, `gelu`, …) — PrivateUse1 impl must call
   the **same** `aten::*` inside a StarPU codelet with **identical** tensor
   layouts/strides as CUDA (no legacy NNLayerNorm reduced-stats layout, no
   blanket `contiguous()` in the wrapper).
3. **Explicit `cpu_fallback`** — unregistered schemas only when
   `init_context(cpu_fallback=True)`; not the default training path.

## What must NOT be on `aten::PrivateUse1`

| Pattern | CUDA | nntile rule |
|---------|------|-------------|
| `CompositeImplicitAutograd` (`linear`, `matmul`, `chunk`, `narrow`, `rms_norm`, …) | Composite lowers to primitives | **Do not** register PrivateUse1 |
| VariableType-only autograd (`rsqrt` → `pow`) | No custom AutogradCUDA kernel | **Do not** register AutogradPrivateUse1 |
| Classic NNTile tiled ops (`gemm`, `rope`, `sum_slice`, …) | N/A | **`torch_nntile.nn.functional` only** |
| Host-only reimplementations (`div`, `where`, … in `nntile_host_aten.cpp`) | CUDA device kernels | **Remove or replace** with torch-native StarPU aten codelets |

## LayerNorm parity (fixed Aug 2026)

Previous `nntile_layer_norm.cpp` forced `.contiguous()` and allocated **reduced**
(non-keepdim) mean/rstd buffers (legacy `NNLayerNormOp`). CUDA uses keepdim
`[..., 1]` stats and accepts strided inputs. The PrivateUse1 wrapper now:

- passes tensors with their **actual strides** into `pack_tensor_layout`;
- allocates mean/rstd with **keepdim** shapes matching CUDA;
- lets the StarPU codelet call `at::native_layer_norm` without reshaping stats.

## Layout / densification policy (Aug 2026)

| Path | Rule |
|------|------|
| **`aten::PrivateUse1` (torch-native)** | No hidden `.contiguous()` in wrappers. Match CUDA: strided tensors flow into StarPU codelets via `pack_tensor_layout`. GEMM (`mm`/`bmm`/`addmm`) records real operand layouts and executes `at::mm_out` / `at::bmm_out` / `at::addmm_out` in the codelet (ATen/cuBLAS infers transpose from strides). |
| **`torch_nntile.nn.functional` (classic)** | `require_nntile_kernel_dense`: `is_contiguous() && storage_offset==0` or explicit error. User must call `.contiguous()` (autograd-tracked) before classic ops. |
| **Registered `aten::contiguous`** | Allowed — goes through `ContiguousFn` (`AutogradPrivateUse1`). |
| **`_to_copy` / host I/O** | Explicit staging only. |

**CUDA ops that densify internally (not in our wrappers):** PyTorch's
`_scaled_dot_product_efficient_attention` backward on CUDA may require
contiguous layouts inside cuDNN/Flash — we no longer pre-densify in
`run_sdpa_efficient_backward`; the codelet uses strided `from_blob` views and
falls back to math SDPA on failure (same decomposition as eager attention).

Helper: `torch_nntile/csrc/nntile_layout_checks.h`.

## Audit

Run before adding a new PrivateUse1 registration:

```bash
python3 torch_nntile/tools/audit_cuda_parity_registration.py \
  --ops native_layer_norm linear rms_norm contiguous
```

Compare `torch._C._dispatch_dump_table("aten::OP")` before and after
`import torch_nntile`. New PrivateUse1 rows on composite-only ops are
regressions unless documented.

## Migration checklist (ongoing)

- [x] LayerNorm: CUDA keepdim stats, no wrapper `contiguous()`
- [x] `torch_nntile.nn.functional` namespace for classic `_C` ops (`kernels` alias)
- [x] `pow` / `div` / `triu` host wrappers call matching `aten::*_out` codelets
- [x] Remove blanket `contiguous()` from unary/binary PrivateUse1 wrappers (mul, trig, …)
- [x] Remove hidden densify from GEMM layout prep, SDPA, classic linear/gemm
- [ ] CI: `test_aten_ops_parity` compares nntile vs **cuda** (not only cpu) for
      registered device primitives
- [ ] Registration audit in pre-commit or CI
