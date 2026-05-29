# Tests labeled `NotImplemented`

These CTest entries are registered but the underlying ops are not implemented yet.
CI skips them with `ctest -LE NotImplemented` (see `.github/workflows/build-test.yml`).

Regenerate the list from a configured build:

```bash
.github/scripts/list-not-implemented-tests.sh build
```

Source of truth: `TESTS_NOT_IMPLEMENTED` in:

- `nntile/tests/kernel/CMakeLists.txt` → `tests_kernel_*`
- `nntile/tests/starpu/CMakeLists.txt` → `tests_starpu_*`
- `nntile/tests/core/CMakeLists.txt` → `tests_core_tile_*` (currently empty)

## Kernel (`tests_kernel_*`, 7)

| CTest name |
|------------|
| `tests_kernel_conv2d_bwd_input_inplace` |
| `tests_kernel_conv2d_bwd_weight_inplace` |
| `tests_kernel_conv2d_inplace` |
| `tests_kernel_scale_inplace` |
| `tests_kernel_subtract_indexed_outputs` |
| `tests_kernel_sumprod_fiber` |
| `tests_kernel_total_sum_accum` |

## StarPU (`tests_starpu_*`, 49)

| CTest name |
|------------|
| `tests_starpu_accumulate` |
| `tests_starpu_accumulate_hypot` |
| `tests_starpu_accumulate_maxsumexp` |
| `tests_starpu_adam_step` |
| `tests_starpu_adamw_step` |
| `tests_starpu_add` |
| `tests_starpu_add_fiber` |
| `tests_starpu_add_fiber_inplace` |
| `tests_starpu_add_inplace` |
| `tests_starpu_add_slice` |
| `tests_starpu_codelet` |
| `tests_starpu_conv2d_bwd_input_inplace` |
| `tests_starpu_conv2d_bwd_weight_inplace` |
| `tests_starpu_conv2d_inplace` |
| `tests_starpu_copy` |
| `tests_starpu_embedding` |
| `tests_starpu_embedding_backward` |
| `tests_starpu_gelu_backward` |
| `tests_starpu_gelutanh` |
| `tests_starpu_gelutanh_backward` |
| `tests_starpu_handle` |
| `tests_starpu_hypot` |
| `tests_starpu_hypot_inplace` |
| `tests_starpu_hypot_scalar_inverse` |
| `tests_starpu_log_scalar` |
| `tests_starpu_logsumexp` |
| `tests_starpu_multiply` |
| `tests_starpu_multiply_fiber` |
| `tests_starpu_multiply_fiber_inplace` |
| `tests_starpu_multiply_slice` |
| `tests_starpu_pow` |
| `tests_starpu_relu` |
| `tests_starpu_relu_backward` |
| `tests_starpu_rope` |
| `tests_starpu_rope_backward` |
| `tests_starpu_scale` |
| `tests_starpu_scale_fiber` |
| `tests_starpu_scale_inplace` |
| `tests_starpu_scale_slice` |
| `tests_starpu_sgd_step` |
| `tests_starpu_silu` |
| `tests_starpu_silu_backward` |
| `tests_starpu_silu_inplace` |
| `tests_starpu_softmax` |
| `tests_starpu_sqrt` |
| `tests_starpu_subtract_indexed_outputs` |
| `tests_starpu_sum_fiber` |
| `tests_starpu_sumprod_fiber` |
| `tests_starpu_total_sum_accum` |

## Core / tile (`tests_core_tile_*`)

None (`TESTS_NOT_IMPLEMENTED` is empty in `nntile/tests/core/CMakeLists.txt`).

**Total: 56** (7 kernel + 49 starPU; use the script after CMake configure to verify).
