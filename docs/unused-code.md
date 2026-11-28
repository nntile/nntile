<!--
 @copyright (c) 2022-present Skolkovo Institute of Science and Technology
                              (Skoltech), Russia. All rights reserved.
                2023-present Artificial Intelligence Research Institute
                              (AIRI), Russia. All rights reserved.

 NNTile is software framework for fast training of big neural networks on
 distributed-memory heterogeneous systems with GPU acceleration.

 @file docs/unused-code.md
 @version 2.0.0
-->

<!-- SPDX-License-Identifier: MIT -->
# Unused and reference-only code in NNTile 2.0

NNTile 2.0 is a breaking release: it is CUDA-only and NNHaul-only. Code that
is no longer on the product path is either **deleted** or **explicitly marked**
here with the reason it is retained. The rule is:

- **Delete** if it was 1.x-only and is *not* planned to return.
- **Keep and mark** if it is test/reference code, an optional deployment path,
  or is planned to be revived.
- **Keep active** if a product, test, tool, or optional service actually uses
  it.

This file is the registry of everything that is not obviously on the default
`torch_nntile` product path.

## Kept because it is planned to be revived

### Disabled kernel families

`cmake/NNTileKernelScope.cmake` lists exactly seven families:
`conv2d_inplace`, `conv2d_bwd_input_inplace`, `conv2d_bwd_weight_inplace`,
`isfinite`, `pow`, `sqrt`, `sqrt_inplace`.

- **Status:** disabled, not called by any product path.
- **Why kept:** the files remain for future re-enablement; every source/header
  carries an `INTENTIONALLY DISABLED` banner, their public ops throw through
  `require_kernel_family()`, and their tests are filtered out.
- **Plan:** revivable with `-DNNTILE_DISABLED_KERNELS=`; see
  [kernel-scope.md](kernel-scope.md).

### Placeholder tests

`nntile/tests/kernel/sumprod_fiber.cc` and
`nntile/tests/kernel/total_sum_accum.cc` return `-1` and are registered with
the `NotImplemented` label. The kernel families themselves are active chunked
kernels; only these tests are unwritten.

- **Status:** excluded from the default ctest run.
- **Why kept:** planned test work, not 1.x residue.

## Kept as reference / non-product paths

### Contiguous CPU kernels

`nntile/include/nntile/kernel/*/cpu.hh` and `nntile/src/kernel/*/{cpu.cc,cpu.hh}`.

- **Status:** compiled into `libnntile`, used by the kernel A/B tests and by
  the CPU bodies of the NNHaul codelets, but NNHaul creates no CPU worker, so
  the product runtime never schedules them.
- **Why kept:** reference implementations for the tests, and the basis for a
  future NNHaul CPU-worker enablement. Chunked CPU kernels are planned for
  that enablement; none exist yet.
- **Rule:** they are contiguous-only and must not be used as a product fallback.
  See [cpp/README.md](cpp/README.md).

### Optional remote execution transport

`nntile/src/remote_execution_driver.cc`, `nntile/src/remote_tile_codec.cc`,
`nntile/include/nntile/remote_execution_driver.hh`,
`nntile/include/nntile/remote_tile_codec.hh`, and the tile tests
`remote_execution_driver` / `daemon_robustness`.

- **Status:** optional out-of-process transport (`nntile-executiond`); not used
  by the default in-process product path.
- **Why kept:** it runs the same NNHaul stack on a server and is exercised by
  the tile tests; it is not a second backend.

### Platform client bindings

`torch_nntile/torch_nntile/_platform.py` and
`torch_nntile/csrc/nntile_platform_bind.h`.

- **Status:** inactive unless `NNTILE_SERVER_SOCKET` is set.
- **Why kept:** closed-platform integration; it ships a TensorGraph to the
  server instead of running it locally.

### Logger service and logger codelets

`logger/` (the small Python logging server), `nntile/src/logger/`, and the
`log_scalar` codelet.

- **Status:** active only when logging is enabled; `log_scalar` is the only
  host-side codelet, forwarding one scalar to the logger thread.
- **Why kept:** diagnostics support.

## Development-only code

`nnhaul/examples/`, `nnhaul/tools/`, and `nnhaul/tests/` are engine development
and trace-analysis tooling. They are built only with `NNHAUL_BUILD_TESTS` /
`NNHAUL_BUILD_EXAMPLES` (the embedded NNTile build follows `BUILD_TESTING`).

## Deleted in the 2.0 cleanup

These were 1.x or intermediate-2.0 code with no planned use; they were removed
rather than marked:

- The universal **staged fallback** in `nntile/include/nntile/nnhaul/stage.hh`
  and `nntile/src/nnhaul/stage.cc`: gather-into-contiguous-scratch /
  run / scatter machinery, its `StagedBuf` state, the scratch live set
  (`stash_scratches`, `drain_scratches`), and the staged-submit probe counters.
  Every product driver now has a chunk-aware path; the docs already described
  this fallback as removed.
  **This is not GEMM staging.** GEMM keeps its own dedicated workspace
  staging (`gemm_workspace_staged` in `nntile/src/nnhaul/gemm.cc` and the
  `Workspace` contract in `nntile/include/nntile/kernel/gemm/cuda.hh`,
  including C2W/W2C counters and the high-water mark); it was not touched.
- `core::Tile::acquire_async` — an alias of the blocking `acquire` with no
  callers.
- `cudnn_get_local_cudnn_handle` — only the deleted flash-SDPA path consumed
  it; no NNTile kernel uses cuDNN directly now (NNHaul still creates its own
  cuDNN handle internally).
- Manual 1.x/early-2.0 scripts `torch_nntile/tests/torch_{add_step,
  gemm_add_step,linear_step,linear_train}.py`; they were not collected by
  pytest and are superseded by the parity tests.
- Unreferenced benchmark images `images/gpt_short8_perf.*`.
- `.github/scripts/deepen-pr-checkout.sh` — replaced by `fetch-depth: 0` in
  the lint workflow.
- `nntile/include/nntile/tensor/tiling_spec_json.hh` and
  `nntile/src/tensor/tiling_spec_json.cc` — the `tiling.json` loader/saver had
  no caller. The one helper still used by `torch_nntile`
  (`apply_tiling_to_axis`) moved to `axis_descriptor.{hh,cc}`.
- `nntile/include/nntile/kernel.hh` — legacy umbrella header with no include
  site anywhere in the tree.
- Zero-caller `Runtime` scheduling/reclaim variants
  (`apply_execution_schedule_from_file`, `clear_execution_schedule_file_cache`,
  `compile_with_round_robin_schedule`, `build_tile_last_consumer_map`,
  `release_dead_tiles_after_op`, `queue_dead_tiles_after_op`,
  `flush_queued_dead_tiles`) and their private state.
- Zero-caller accessors: `TensorGraph::data_names`,
  `TileGraph::data_names`, `has_tiled_axis_group`, `num_untiled_groups`,
  `reset_phase_seal_cursor`, `TileNode::clear_tensor_info`,
  `TensorNode::is_compatible`, `AxisDescriptor::num_tiles`,
  `TensorAxisLayout::tile_nelems_at`,
  `RemoteExecutionDriver::gather_int64/gather_bool`.
- Removed compile switches and backend selection earlier in the 2.0 series:
  StarPU/CPU/macOS backends, `USE_CUDA`, `USE_CUDA_FP8`,
  `TORCH_NNTILE_USE_CUDA`, `NNTILE_FORCE_CHUNKED_KERNELS`, flash-SDPA,
  ATen-on-nntile dispatch.

## Re-auditing

Useful checks:

```bash
# Kernel-family coverage (chunked / driver / disabled)
python3 tools/chunked_coverage.py

# Admits no removed runtime identifier
git grep -inE 'starpu|access_redux'

# Disabled families: every file has the banner
grep -RL 'INTENTIONALLY DISABLED' nntile/include/nntile/kernel/{conv2d_inplace,conv2d_bwd_input_inplace,conv2d_bwd_weight_inplace,isfinite,pow,sqrt,sqrt_inplace} nntile/src/kernel/{conv2d_inplace,conv2d_bwd_input_inplace,conv2d_bwd_weight_inplace,isfinite,pow,sqrt,sqrt_inplace}

# Declarations with no in-tree caller (heuristic; verify before deleting)
git grep -n 'acquire_async\|staged_insert\|drain_scratches'
```
