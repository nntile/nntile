# Graph compilation performance: MNIST investigation (VM)

Measured on the Cloud Agent VM (CPU-only), `torch==2.12.0+cpu`,
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

### Wall ms/step (nntile dry-run, batch=100, seed=42)

| steps | ms/step | PyTorch CPU eager (500 steps) |
|------:|--------:|------------------------------:|
| 100 | 1.12 | — |
| 200 | 1.05 | — |
| 500 | 1.11 | **1.52** |
| 1000 | 1.18 | — |

Session growth is gone; dry-run is **faster than** single-threaded PyTorch CPU
eager on this script (~1.1 vs ~1.5 ms/step). Prior baseline after earlier
fixes was ~4.8 ms/step at 500 steps.

### `print_info()` compile avg (ms/call), dry-run

| steps | compile avg | append_phase avg | runtime.compile avg |
|------:|------------:|-----------------:|--------------------:|
| 100 | 0.30 | 0.037 | 0.12 |
| 500 | 0.34 | 0.034 | 0.17 |
| 1000 | 0.41 | 0.035 | 0.23 |

`append_phase` is flat (~0.035 ms/call; previously ~1.1 ms/call). Mild growth
in `runtime.compile` avg with step count is residual SCATTER-history work in
the tile execution order, not map lookups.

### Batch-size sensitivity (300 steps, dry-run)

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
| train ms/step (real compute) | 3.67 |

## Historical notes (pre-dense redesign)

Earlier work removed O(session) scans (`drop_all_ops` compact, seal carry,
`has_producer`, union-by-size `merge_axis`, etc.). See git history and older
artifact logs under `/opt/cursor/artifacts/mnist_compile_bench/`.

## Remaining work

- Further constant-factor cuts inside individual `lower_to_tile` bodies /
  buffer pooling (only if dry-run gap grows again under multi-tile workloads).
- True “compile once, replay” needs scalar lifting for Adam `lr` / `num_iter`.
- Residual `runtime.compile` growth from retained ingress `SCATTER` tile-op
  history when all MNIST batches are preloaded.

## Logs (this redesign)

- `/opt/cursor/artifacts/mnist_on_bench/`
