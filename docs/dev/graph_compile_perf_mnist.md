# Graph compilation performance: MNIST investigation (VM)

Measured on the Cloud Agent VM (CPU-only), `torch==2.9.1+cpu`,
`torch_nntile` built with `--no-build-isolation` against libnntile
(`-DUSE_CUDA=OFF`).

## Controls

| Knob | Value |
|------|-------|
| StarPU workers | `--ncpu 1 --ncuda 0 --restrict-cpu` |
| Host threads | `torch.set_num_threads(1)`, `OMP/MKL/OPENBLAS_NUM_THREADS=1` |
| Kernels | `STARPU_DISABLE_KERNELS=1` (nntile only) |
| Logging | `--train-log-every 50 --test-every 50` |
| Seed | `0` |

Script: `torch_nntile/examples/reproduce_google_five_layer_relu_mnist.py`.

## Before vs after (real compile reductions)

Not a timer rename. Changes that cut CPU:

1. **`TensorGraph::drop_all_ops()` after each `wait()`** — drops sealed
   non-`SCATTER` ops (keeps ingress scatters + any unsealed next-phase ops)
   so compile stays O(phase) without wiping a pending step or corrupting
   host-ingressed inputs on the next run.
2. **Remove full `tile_map_` scan in `Runtime::sync_tile_marks_from_logical`**
   — reclaim uses `invalidate_logical_tiles` / pending_output_reclaim instead.
3. Reverted the earlier fake “async compile = move work into wait” approach.

### Wall ms/step (nntile, no-compute)

| steps | before | after | change | cpu eager |
|------:|-------:|------:|-------:|----------:|
| 100 | 4.52 | 4.63 | +2% | 1.61 |
| 200 | 4.96 | 4.71 | **-5%** | 1.57 |
| 300 | 5.45 | 4.53 | **-17%** | 1.54 |
| 500 | 6.34 | 4.78 | **-25%** | 1.62 |

Session growth is gone: 500-step ms/step no longer climbs past ~4.8.

### Bucket totals at 500 steps (seconds)

| bucket | before | after |
|--------|-------:|------:|
| record | 1.492 | 0.561 |
| compile | 1.294 | 1.180 |
| run | 0.229 | 0.440 |
| wait | 0.095 | 0.154 |
| readout | 0.059 | 0.048 |

### `print_info()` compile avg (ms/call)

| steps | before compile avg | after | before runtime.compile | after |
|------:|-------------------:|------:|-----------------------:|------:|
| 100 | 2.26 | 2.07 | 0.30 | 0.16 |
| 500 | 2.42 | 2.20 | 0.31 | 0.17 |

`append_phase` stays ~1.1 ms/call (still lowers ~75 tensor ops every step).
That is the remaining gap vs PyTorch (~1.6 ms/step total for real compute).

## Record-path follow-up (graph capture)

After the compile fixes above, **record** was still high and
`linear_backward` / gemm capture avg ms grew with session length on the
Google five-layer ReLU script (all MNIST batches preloaded → ~1230 retained
`SCATTER` ops).

Root causes and fixes:

1. **`ensure_metadata_fill_if_unproduced` scanned every TensorGraph op**
   (including all retained scatters) on each gemm record. Replaced with
   O(1) `TensorNode::has_producer()` (set in `TensorGraph::add_op`) plus
   `is_input()` short-circuit.
2. **`merge_axis(fresh, huge_persistent_group)` walked the large group**
   when the first argument was the smaller side. `merge_axis` now uses
   union-by-size so capture stays O(small) as historical members accumulate.
3. **Pin dedup** uses an `unordered_set<TensorImplKey>` instead of a linear
   scan of `g_pinned_tensors`.

### Record bucket at 500 steps (`STARPU_DISABLE_KERNELS=1`)

| metric | before record fix | after |
|--------|------------------:|------:|
| record total | 0.506 s | **0.280 s** |
| gemm record avg | 0.0100 ms | **0.0025 ms** |
| linear_backward avg | 0.061 ms (grew w/ steps) | **0.014 ms** (flat) |

### Second session-scaling fix (seal / drop / CE)

After the capture fixes above, **compile** still grew with preloaded
batches. Two host-side causes:

1. **`seal_phase()`** carried every historical `mark_input` (all MNIST
   ingress tensors) into each phase, so append refreshed marks on
   O(session) tiles every step. It now carries only tensors referenced by
   the sealed op slice.
2. **`drop_all_ops()`** rebuilt the full `ops_` vector (all retained
   `SCATTER`s) every wait. It now keeps a SCATTER prefix length and erases
   only the sealed non-SCATTER middle (O(phase) after the first compact).

On the **record** path:

1. **Cross-entropy** reuses forward `maxsumexp` in backward and folds a
   constant unit `ones_like(loss)` scale (skips broadcast + `multiply_slice`).
2. **`set_axes`** unifies via `merge_axis` (union-by-size) instead of a
   linear erase from `AxisDescriptor::members`.

Residual compile growth vs preload size still comes from the retained
tile-graph history of ingress `SCATTER`s (runtime DCE / last-consumer over
`execution_order_`). Record ms/step stays flat with session length.

## Async API contract

### `torch_nntile` (Python)

`compile_graph()` / `run()` / `execute()` / `wait()`:

- **`compile_graph()` / `run()` / `execute()`** — host work on the calling
  thread (may enqueue StarPU tasks). They do **not** join StarPU.
  `execute()` is compile+run only (same as the split API).
- **`wait()`** — the only API that blocks on StarPU completion and runs
  post-run reclaim / session compact.

### C++ `nntile::Runtime`

- **`execute()` / `execute_range()`** — submit compiled tile ops only (async).
- **`wait()`** — `starpu_task_wait_for_all` + flush last-consumer reclaim.

Do **not** treat “async compile” as moving host CPU work onto a background
thread and joining in `wait()` — that only renames timers.

Note: a later `compile_graph()` / `execute()` still finishes a prior async
`run()` before sealing the next phase (correctness). That is not a wait
hidden inside a standalone `execute()` of a fresh phase.

## Remaining work to approach PyTorch

- Activation buffer pool / stable logical nodes (skip `build_tile_nodes` + layout
  rebuild for identical shapes).
- Prune or avoid retaining unmarked tensors in `AxisDescriptor::members`
  (memory; capture cost is already union-by-size).
- Single-tile fast-path in `lower_to_tile` (GEMM etc.).
- True “compile once, replay” needs scalar lifting for Adam `lr` / `num_iter`.

## Logs

- Baseline: `/opt/cursor/artifacts/mnist_compile_bench/baseline_logs/`
- Final: `/opt/cursor/artifacts/mnist_compile_bench/final_logs/`
- Record-path: `/opt/cursor/artifacts/mnist_record_bench/`
