# Graph compilation performance: MNIST investigation (VM)

Measured on the Cloud Agent VM (CPU-only). Prefer the tree’s supported
`torch==2.9.1` ABI (`torch_nntile`); do not use torch 2.12.
`torch_nntile` built with `--no-build-isolation` against libnntile
(`-DUSE_CUDA=OFF`).

## Controls

| Knob | Value |
|------|-------|
| StarPU workers | `--ncpu 1 --ncuda 0 --restrict-cpu` |
| Host threads | `torch.set_num_threads(1)`, `OMP/MKL/OPENBLAS_NUM_THREADS=1` |
| Kernels | `STARPU_DISABLE_KERNELS=1` (nntile only; still submits tasks) |
| No submit / I/O | `TORCH_NNTILE_SKIP_STARPU=1` (skip StarPU task insert + staging acquire/memcpy; still advances execute watermark + last-consumer reclaim so compile stays O(pending); accuracy meaningless) |
| Logging | `--train-log-every 50 --test-every 50` |
| Seed | `0` (accuracy) / `42` (dry-run scaling) |

Script: `torch_nntile/examples/reproduce_google_five_layer_relu_mnist.py`.

Design: [graph_compiler_on_design.md](graph_compiler_on_design.md).

## `TORCH_NNTILE_SKIP_STARPU` dry-run

Set `TORCH_NNTILE_SKIP_STARPU=1` to measure **record + compile** without StarPU
compute or host↔tile copies.

| Still runs | Skipped |
|------------|---------|
| TensorGraph capture (record) | `OpNode::execute` (StarPU task insert) |
| Seal / lower / `Runtime::compile` (incl. allocate) | Staging `acquire` + memcpy |
| `execute_range(..., submit_tasks=false)` — advances `executed_op_end_` and queues last-consumer reclaim | Kernel work / meaningful numerics |
| `wait()` reclaim / `invalidate_logical_tiles` | |

**Do not** skip `execute_range` entirely: leaving `Runtime::executed_op_end_`
unchanged makes every later `compile()` treat full history as pending
(O(session) DCE/allocate — tens of seconds on this script).

```bash
STARPU_WORKERS_NOBIND=1 TORCH_NNTILE_SKIP_STARPU=1 \
  python torch_nntile/examples/reproduce_google_five_layer_relu_mnist.py \
    --steps 500 --batch-size 100 --seed 42 --device nntile \
    --train-log-every 50 --test-every 50 --ncpu 1 --skip-accuracy-floor
```

## After O(N) dense mapping redesign (2026-07)

Hot-path `std::map` / `std::set` bridges replaced with dense `NodeId` tables;
`TileNode::payload_` replaces `Runtime::tile_map_`; last-consumer reclaim is
O(#dying) per op. Fully tiled `lower_to_tile` paths unchanged.

Later: pending-window last-consumer map, then TileGraph/`execution_order_`
history drop after `wait()` (mirror of TensorGraph `drop_all_ops`).

### Wall ms/step comparison (`TORCH_NNTILE_SKIP_STARPU=1`, batch=100, seed=42)

| steps | before history fixes¹ | + last-consumer fix² | + history drop³ |
|------:|----------------------:|---------------------:|----------------:|
| 100 | 1.09 | 1.04 | 1.05 |
| 1000 | 1.20 | 0.96 | **0.90** |
| 10000 | **2.44** | 1.33 | **0.86** |

¹ Dense `NodeId` maps only (`after_skip_*.log`).
² Pending-window last-consumer (`28d12e4f`, `after_fix_skip_*.log`).
³ `drop_fully_executed_history` + `TileGraph::clear_ops` (`c006e0b3`,
`after_hist_skip_*.log`).

`runtime.compile` avg (ms/call) at 10k steps: 1.40 → 0.42 → **0.069**
(flat across 100→10000 after history drop). After treating sealed ingress
`SCATTER` as ordinary droppable history, session `tensor_graph_ops` and
`executed_tile_ops` both report `0` after each `wait()` (previously
`tensor_graph_ops` stayed ≈1230 from the retained SCATTER prefix).

### Wall ms/step (nntile dry-run after history drop, batch=100, seed=42)

| steps | ms/step | PyTorch CPU eager (500 steps) |
|------:|--------:|------------------------------:|
| 100 | 1.05 | — |
| 1000 | 0.90 | — |
| 10000 | 0.86 | — |
| 500 (earlier dense-map run) | ~1.1 | **1.52** |

Dry-run is **faster than** single-threaded PyTorch CPU eager on this script
and **flat** with step count. Prior baseline after earlier fixes was
~4.8 ms/step at 500 steps.

### `print_info()` compile avg (ms/call), dry-run (after history drop)

| steps | runtime.compile avg |
|------:|--------------------:|
| 100 | 0.081 |
| 1000 | 0.069 |
| 10000 | 0.069 |

### Batch-size sensitivity (300 steps, dry-run, dense-map era)

| batch | ms/step |
|------:|--------:|
| 50 | 1.28 |
| 100 | ~1.1 |
| 200 | 0.94 |

### StarPU accuracy (seed=0, 1000 steps, `--ncpu 1 --restrict-cpu`)

| metric | value |
|--------|------:|
| max / final test accuracy | **0.9706** |
| floor (≥0.97) | met |
| train ms/step (real compute, after history drop) | 3.42 |

## Historical notes (pre-dense redesign)

Earlier work removed O(session) scans (`drop_all_ops` compact, seal carry,
`has_producer`, union-by-size `merge_axis`, etc.). See git history and older
artifact logs under `/opt/cursor/artifacts/mnist_compile_bench/`.

## Remaining work

- Further constant-factor cuts inside individual `lower_to_tile` bodies /
  buffer pooling (only if dry-run gap grows again under multi-tile workloads).
- True “compile once, replay” needs scalar lifting for Adam `lr` / `num_iter`.

## Logs (this redesign)

- `/opt/cursor/artifacts/mnist_on_bench/`
