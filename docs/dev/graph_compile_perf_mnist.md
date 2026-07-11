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

## Baseline (pre-optimization)

Train-step wall excludes eval; nntile breakdown is script timers.

| steps | cpu ms/step | nntile ms/step | record s | compile s | run s | wait s | readout s |
|------:|------------:|---------------:|---------:|----------:|------:|-------:|----------:|
| 100 | 1.61 | 4.52 | 0.119 | 0.247 | 0.059 | 0.017 | 0.013 |
| 200 | 1.57 | 4.96 | 0.335 | 0.496 | 0.105 | 0.036 | 0.022 |
| 300 | 1.54 | 5.45 | 0.651 | 0.749 | 0.145 | 0.051 | 0.042 |
| 500 | 1.51 | 6.34 | 1.492 | 1.294 | 0.229 | 0.095 | 0.059 |

`print_info()` compile averages (ms/call) grow only mildly; **record** and
session metadata growth dominate end-to-end ms/step slope.

| steps | compile avg ms | seal | tiling | append | runtime.compile | tensor_graph_ops | host_readout calls |
|------:|---------------:|-----:|-------:|-------:|----------------:|-----------------:|-------------------:|
| 100 | 2.26 | 0.50 | 0.43 | 1.02 | 0.30 | 8749 | 66 |
| 200 | 2.30 | 0.51 | 0.45 | 1.03 | 0.30 | 15937 | 110 |
| 300 | 2.34 | 0.51 | 0.45 | 1.05 | 0.32 | 23125 | 154 |
| 500 | 2.42 | 0.53 | 0.49 | 1.10 | 0.31 | 37501 | 242 |

### Ranked bottlenecks (no-compute)

1. **`compile_graph` CPU** (~50% of nntile step at 100; still large later) —
   `append_phase` is the largest sub-bucket, then seal/tiling.
2. **Op recording growth** — `record` ms/step rises with session length
   (graph / pin / autograd bookkeeping over a growing `TensorGraph`).
3. **Host readout (D1)** — every-50 `.cpu()` appends gather/staging nodes
   (`host_readout` ~2.1 ms/call including nested compile/run/wait).
4. **`run` submit** — small; **`wait`** — tiny with kernels disabled
   (as intended for this matrix).

### Async contract (baseline)

| API | Blocks? |
|-----|---------|
| `compile_graph()` | Yes (seal/tiling/append/`Runtime::compile`); also auto-waits prior `run` via `finish_run_locked` |
| `run()` | No (submit-only) |
| `wait()` | Yes |

Target: **only `wait()` blocks**; compile and run return immediately.

## After changes (same VM controls)

Changes landed:

- `compile_graph()` / `run()` enqueue work on a background pipeline; **only
  `wait()` joins** compile + StarPU + reclaim.
- Removed compile-time full `finish_run_locked()`; prior StarPU is drained
  without reclaim (reclaim stays in `wait()`).
- `pending_output_reclaim` dedup uses `unordered_set` (was O(n²)).
- MNIST script forces `torch.set_num_threads(1)`.

Python-visible step breakdown (nntile, `STARPU_DISABLE_KERNELS=1`):

| steps | ms/step | record s | compile s | run s | wait s | readout s |
|------:|--------:|---------:|----------:|------:|-------:|----------:|
| 100 | 5.03 | 0.124 | 0.004 | 0.000 | 0.366 | 0.013 |
| 200 | 5.40 | 0.349 | 0.007 | 0.000 | 0.703 | 0.022 |
| 300 | 5.64 | 0.622 | 0.009 | 0.000 | 1.030 | 0.030 |
| 500 | 6.55 | 1.481 | 0.014 | 0.000 | 1.729 | 0.047 |

`print_info()` still reports CPU compile time (~2.3 ms/call); that work now
runs off the Python thread and is joined inside `wait()` (script `wait`
bucket). Async contract: **pass**.

End-to-end ms/step is similar because this loop still `wait()`s every step;
absolute compile CPU is not yet much smaller. Remaining compile reduction
work: D1 staging reuse, record-path session growth, append_phase cost.

## Logs

- Baseline: `/opt/cursor/artifacts/mnist_compile_bench/baseline_logs/`
- After: `/opt/cursor/artifacts/mnist_compile_bench/after_logs/`
