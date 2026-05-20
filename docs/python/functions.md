# Tensor function wrappers

[`wrappers/python/nntile/functions.py`](../../wrappers/python/nntile/functions.py)
wraps C++ tensor kernels. Import via `nntile.functions` or `nntile.tensor`.

Almost all operations are **async** (StarPU tasks). Exceptions: blocking `gelu`,
and synchronous fused optimizer steps. Call `nntile.starpu.wait_for_all()` before
using host-side results.

`is_tensor_of(tensors, dtype)` checks that all tensors share one dtype class.

## Linear algebra

| Function | Description |
|----------|-------------|
| `gemm_async(alpha, trans_a, A, trans_b, B, beta, C)` | General matrix multiply: C = $\alpha$·op(A)·op(B) + $\beta$·C |
| `transpose_async(alpha, src, dst, ndim)` | Transpose leading `ndim` dimensions |

## Activations

| Function | Description |
|----------|-------------|
| `relu_async` / `relu_inplace_async` | ReLU forward / in-place |
| `relu_backward_async` | ReLU backward |
| `silu_async` / `silu_inplace_async` / `silu_backward_async` | SiLU (Swish) |
| `gelu_async` / `gelu` / `gelu_inplace_async` / `gelu_backward_async` | GELU (exact); `gelu` is blocking |
| `gelutanh_async` / `gelutanh_inplace_async` / `gelutanh_backward_async` | Tanh-approx GELU |

## Elementwise and scalars

| Function | Description |
|----------|-------------|
| `fill_async(val, x)` | Fill tensor `x` with scalar `val` |
| `clear_async(x)` | Zero tensor `x` |
| `pow_async(alpha, exp, x)` | Power $x := \alpha x^{\exp}$ |
| `multiply_async` / `multiply_inplace_async` | Elementwise multiply |
| `add_async` / `add_inplace_async` | z = $\alpha$·x + $\beta$·y |
| `scale_async` / `scale_inplace_async` | Scale tensor |
| `sqrt_async` / `sqrt_inplace_async` | Square root |
| `hypot_async` / `hypot_inplace_async` | Hypot norm |
| `hypot_scalar_inverse_async(eps, alpha, x)` | Inverse hypot for stability |
| `mask_scalar_async(mask, alpha, x, val)` | Apply bool mask |

## Reductions along axis (fiber / slice)

| Function | Description |
|----------|-------------|
| `sum_async` | Sum entire tensor into output |
| `sum_slice_async` | Sum along axis (slice layout) |
| `sum_fiber_async` | Sum along axis (fiber layout) |
| `sumprod_slice_async` / `sumprod_fiber_async` | Sum of products along axis |
| `norm_async` | Norm into output |
| `norm_slice_async` / `norm_slice_inplace_async` | Norm along slice |
| `norm_fiber_async` / `norm_fiber_inplace_async` | Norm along fiber |
| `add_slice_async` / `add_slice_inplace_async` | Broadcast-add slice to tensor |
| `scale_slice_async` | Broadcast-scale slice |
| `add_fiber_async` / `add_fiber_inplace_async` | Broadcast-add fiber |
| `scale_fiber_async` | Broadcast-scale fiber |
| `multiply_slice_async` | Broadcast-multiply slice |
| `multiply_fiber_async` / `multiply_fiber_inplace_async` | Broadcast-multiply fiber |

## Softmax and attention statistics

| Function | Description |
|----------|-------------|
| `maxsumexp_async` | Max and sum-exp along axis (stable softmax building block) |
| `softmax_async` / `softmax_inplace_async` | Softmax using maxsumexp workspace |
| `logsumexp_async` | Log-sum-exp from maxsumexp buffer |

## Data movement and random

| Function | Description |
|----------|-------------|
| `copy_async` | Copy tensor |
| `copy_intersection_async` | Copy overlapping subregion with offsets |
| `gather_async` / `scatter_async` | MPI gather / scatter |
| `embedding_async` | Lookup rows by int64 indices |
| `embedding_backward_async` | Embedding backward |
| `randn_async` | Gaussian noise in subregion |
| `log_scalar_async(name, value)` | Log scalar metric (debug) |

## Loss helpers

| Function | Description |
|----------|-------------|
| `total_sum_accum_async` | Cross-entropy style accumulation with labels |
| `subtract_indexed_outputs_async` | Subtract target class from logits |

## Fused optimizers (synchronous)

| Function | Description |
|----------|-------------|
| `fused_adam_step` | Adam update step |
| `fused_adamw_step` | AdamW update step|
| `fused_sgd_step` | SGD with momentum / Nesterov |

## Transformer and convolution

| Function | Description |
|----------|-------------|
| `rope_async` / `rope_backward_async` | Rotary position embedding |
| `conv2d_inplace_async` | 2D convolution in-place |
| `conv2d_bwd_input_inplace_async` | Conv backward w.r.t. input |
| `conv2d_bwd_weight_inplace_async` | Conv backward w.r.t. weights |
| `flash_sdpa_fwd_cudnn_async` | cuDNN flash scaled dot-product attention forward (fp16/bf16) |
| `flash_sdpa_bwd_cudnn_async` | cuDNN flash SDPA backward |

## See also

- [tensors.md](tensors.md) — allocation and I/O
- [cpp/README.md](../cpp/README.md) — underlying C++ ops
