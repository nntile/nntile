<!-- SPDX-License-Identifier: MIT -->
# Kernel scope: transformers only

NNTile's default focus is transformer training and inference. Kernel
families outside that scope are **disabled** by default: their public
operations throw, and their numerical/performance tests are not
registered in ctest, so they no longer contribute to the suites or to
benchmark tracking. Every source and header of a disabled family carries an
**INTENTIONALLY DISABLED** banner at the top of the file. No product path
calls the family: the public entry points throw through
`require_kernel_family()` before any codelet is submitted. A shared A/B test
that also covers enabled families (`chunked_swap_conv_randn_ab`) keeps its
disabled-family sections as Catch2 skips. The sources remain in the tree and
can be re-enabled with `-DNNTILE_DISABLED_KERNELS=` (empty).

The list lives in `NNTILE_DISABLED_KERNELS` (see
`cmake/NNTileKernelScope.cmake`) and is exported to `nntile/defs.h`;
`nntile::require_kernel_family()` enforces it at the public op entry
points. Tests-only builds adopt the list from the installed product's
`nntileConfig.cmake`.

## Disabled now

| Family | Why |
|---|---|
| `conv2d_inplace` | CNN path; never numerically validated in the transformer suites |
| `conv2d_bwd_input_inplace` | CNN path; same |
| `conv2d_bwd_weight_inplace` | CNN path; same |
| `isfinite` | diagnostic utility, no transformer op path |
| `pow` | generic math, no transformer op path |
| `sqrt` | generic math; LayerNorm/RMSNorm use `hypot`/`norm_*` |
| `sqrt_inplace` | generic math; same |

Re-enable everything with:

```bash
cmake -S . -B build -DNNTILE_DISABLED_KERNELS=
```

## Kept (exercise by transformer models/training)

- GEMM and layout: `gemm`, `transpose`, `swap_two_axes`, `copy`,
  `subcopy`, `clear`, `fill`, `mask_scalar`
- Embeddings: `embedding`, `embedding_backward`
- Normalization: `norm`, `norm_slice`, `norm_slice_inplace`,
  `norm_fiber`, `norm_fiber_inplace`, `hypot`, `hypot_inplace`,
  `hypot_scalar_inverse`
- Activations: `gelu*`, `gelutanh*`, `silu*`, `relu*`
- Attention/loss math: `softmax`, `softmax_inplace`, `maxsumexp`,
  `logsumexp`, `total_sum_accum`, `subtract_indexed_outputs`, `sum*`,
  `sumprod_*`, `scale*`, `add*`, `multiply*`, `rope`, `rope_backward`
- Training: `sgd_step`, `adam_step`, `adamw_step`, `randn`

The transformer coverage is exercised by the classic model/test suites
(GPT-2, Llama, GPT-Neo/NeoX, BERT/RoBERTa, T5) and the GPT-2 training path.
