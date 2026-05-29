# Tests labeled `NotImplemented`

These CTest entries are registered but the underlying ops are not implemented yet.
CI skips them with `ctest -LE NotImplemented` (see `.github/workflows/build-test.yml`).

Regenerate the list from a configured build:

```bash
.github/scripts/list-not-implemented-tests.sh build
```

Source of truth: `TESTS_NOT_IMPLEMENTED` in:

- `nntile/tests/kernel/CMakeLists.txt` → `tests_core_kernel_*`
- `nntile/tests/starpu/CMakeLists.txt` → `tests_core_starpu_*`
- `nntile/tests/core/CMakeLists.txt` → `tests_core_tile_*` (currently empty)

## Kernel (`tests_core_kernel_*`, 7)

| CTest name |
|------------|
| `tests_core_kernel_conv2d_bwd_input_inplace` |
| `tests_core_kernel_conv2d_bwd_weight_inplace` |
| `tests_core_kernel_conv2d_inplace` |
| `tests_core_kernel_scale_inplace` |
| `tests_core_kernel_subtract_indexed_outputs` |
| `tests_core_kernel_sumprod_fiber` |
| `tests_core_kernel_total_sum_accum` |

## StarPU (`tests_core_starpu_*`, 50)

| CTest name |
|------------|
| `tests_core_starpu_accumulate` |
| `tests_core_starpu_accumulate_hypot` |
| `tests_core_starpu_accumulate_maxsumexp` |
| `tests_core_starpu_adam_step` |
| `tests_core_starpu_adamw_step` |
| `tests_core_starpu_add` |
| `tests_core_starpu_add_fiber` |
| `tests_core_starpu_add_fiber_inplace` |
| `tests_core_starpu_add_inplace` |
| `tests_core_starpu_add_slice` |
| `tests_core_starpu_codelet` |
| `tests_core_starpu_conv2d_bwd_input_inplace` |
| `tests_core_starpu_conv2d_bwd_weight_inplace` |
| `tests_core_starpu_conv2d_inplace` |
| `tests_core_starpu_copy` |
| `tests_core_starpu_embedding` |
| `tests_core_starpu_embedding_backward` |
| `tests_core_starpu_gelu_backward` |
| `tests_core_starpu_gelutanh` |
| `tests_core_starpu_gelutanh_backward` |
| `tests_core_starpu_handle` |
| `tests_core_starpu_hypot` |
| `tests_core_starpu_hypot_inplace` |
| `tests_core_starpu_hypot_scalar_inverse` |
| `tests_core_starpu_log_scalar` |
| `tests_core_starpu_logsumexp` |
| `tests_core_starpu_multiply` |
| `tests_core_starpu_multiply_fiber` |
| `tests_core_starpu_multiply_fiber_inplace` |
| `tests_core_starpu_multiply_slice` |
| `tests_core_starpu_pow` |
| `tests_core_starpu_relu` |
| `tests_core_starpu_relu_backward` |
| `tests_core_starpu_rope` |
| `tests_core_starpu_rope_backward` |
| `tests_core_starpu_scale` |
| `tests_core_starpu_scale_fiber` |
| `tests_core_starpu_scale_inplace` |
| `tests_core_starpu_scale_slice` |
| `tests_core_starpu_sgd_step` |
| `tests_core_starpu_silu` |
| `tests_core_starpu_silu_backward` |
| `tests_core_starpu_silu_inplace` |
| `tests_core_starpu_softmax` |
| `tests_core_starpu_sqrt` |
| `tests_core_starpu_subtract_indexed_outputs` |
| `tests_core_starpu_sum_fiber` |
| `tests_core_starpu_sumprod_fiber` |
| `tests_core_starpu_total_sum_accum` |

## Core / tile (`tests_core_tile_*`)

None (`TESTS_NOT_IMPLEMENTED` is empty in `nntile/tests/core/CMakeLists.txt`).

**Total: 57** (as of the lists above; use the script after CMake configure to verify).
