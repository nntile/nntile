# Tiny HF train showcase: CPU vs `device=nntile`

Short smoke runs of stock HuggingFace models through `torch_nntile`
PrivateUse1 (`device=nntile`) versus plain PyTorch CPU. Goal: show
**numerical parity** (matching loss) and how to read the printed
timings — not a large-scale throughput benchmark.

Scripts live under [`torch_nntile/examples/`](../../torch_nntile/examples/).
Each uses a tiny JSON config (`*_hf_tiny_config.json`) and the same
`train` / `compare` CLI shape as [`train_gpt2_hf.py`](../../torch_nntile/examples/train_gpt2_hf.py).

## How to run

Environment (CPU StarPU build; no CUDA in this Cloud Agent image)::

```bash
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig
export LD_LIBRARY_PATH=$PWD/build/nntile:$PWD/build/torch_nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NNTILE_BUILD_DIR=$PWD/build TORCH_NNTILE_BUILD_DIR=$PWD/build
export NNTILE_SOURCE_DIR=$PWD
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
```

Single model (example: Llama)::

```bash
# CPU reference
python torch_nntile/examples/train_llama_hf.py train \
  --device cpu --seed 0 --steps 1 --seq-len 16 --batch-size 1 \
  --output-dir /tmp/llama_cpu

# nntile (one StarPU CPU worker)
python torch_nntile/examples/train_llama_hf.py train \
  --device nntile --seed 0 --steps 1 --seq-len 16 --batch-size 1 \
  --ncpu 1 --output-dir /tmp/llama_nntile

# Optional weight compare (relative Frobenius per tensor)
python torch_nntile/examples/train_llama_hf.py compare \
  --checkpoint-a /tmp/llama_cpu/checkpoint.pt \
  --checkpoint-b /tmp/llama_nntile/checkpoint.pt
```

Batch all HF smokes and print a markdown table::

```bash
python torch_nntile/examples/bench_hf_tiny_cpu_vs_nntile.py \
  --ncpu 1 --steps 1 --seq-len 16 --batch-size 1 --seed 0 \
  --markdown-out /tmp/hf_tiny_cpu_vs_nntile/table.md
```

Models covered by the bench: GPT-2, GPT-Neo, GPT-NeoX, Llama, BERT,
RoBERTa, T5 (all stock HF, tiny JSON configs).

## What the outputs mean

### Loss

Each script prints one line per train step, for example::

```text
[llama] step 1/1  loss=4.915326
```

- **Loss** is next-token (causal) or MLM / T5 CE on a **synthetic**
  mini-batch (deterministic from `--seed`).
- Same seed + same config ⇒ CPU and nntile should match to printing
  precision when the aten graph is covered. In the table below, **Δ loss
  is exactly 0** for every model (bit-identical printed values).

### Wall time (train loop)

Tiny HF smokes print::

```text
[llama] wall=0.024s  OK
```

That is **only the train loop** (forward → backward → optimizer step →
host loss readout), after the model and batch are already on device.
It does **not** include Python import, HuggingFace construction, or
StarPU `init_context` / shutdown.

For GPT-2 (`train_gpt2_hf.py`) the analogous line is::

```text
timing torch train wall (incl. per-iter sync): 0.005s (1 epochs)
timing nntile train wall (incl. per-iter loss sync): 0.015s (1 epochs)
```

On **nntile**, each step also `compile_graph` / `run`s and syncs loss
with `.cpu()` / `.to("cpu")` so StarPU reclaim stays in the same phase
(see [torch_nntile_tensor_architecture.md](torch_nntile_tensor_architecture.md)).
That sync is part of the reported wall.

### Process elapsed (not in the table)

Spawning a fresh Python process per run costs ~3 s here (imports +
TF/transformers side effects). Do **not** treat process elapsed as a
device comparison — use the printed train-loop wall.

## Results (CPU vs nntile, `ncpu=1`)

Measured with `bench_hf_tiny_cpu_vs_nntile.py` on the Cloud Agent VM
(CPU-only StarPU / `USE_CUDA=OFF`, `ncpu=1`, `steps=1`, `seq-len=16`,
`batch-size=1`, `seed=0`, date 2026-07-18). Tiny configs: ~64-wide
hidden, 1–2 layers — **overhead-dominated**, not a speed contest.

| Model | CPU loss | nntile loss | CPU wall (s) | nntile wall (s) | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---|
| gpt2 | 5.615653 | 5.615653 | 0.005 | 0.015 | 0.000e+00 | OK |
| gpt-neo | 4.827158 | 4.827158 | 0.004 | 0.013 | 0.000e+00 | OK |
| gpt-neox | 4.835960 | 4.835960 | 0.004 | 0.020 | 0.000e+00 | OK |
| llama | 4.915326 | 4.915326 | 0.004 | 0.024 | 0.000e+00 | OK |
| bert | 4.833461 | 4.833461 | 0.004 | 0.013 | 0.000e+00 | OK |
| roberta | 4.735972 | 4.735972 | 0.004 | 0.013 | 0.000e+00 | OK |
| t5 | 5.692081 | 5.692081 | 0.005 | 0.033 | 0.000e+00 | OK |

**Takeaways for demos**

1. **Correctness:** losses match on CPU and nntile for every model above.
2. **Timing at this scale:** nntile wall is higher (StarPU submit +
   compile/run + host sync) while the math is tiny; expect the gap to
   shrink (and reverse) on larger seq / batch / depth or with CUDA
   workers (`--ncuda`).
3. **Checkpoints:** each successful `--output-dir` run writes
   `checkpoint.pt` (`model_state_dict` + HF `config` dict + seed /
   step). Use `compare` for relative Frobenius norms.

## Nntile-native model scripts

Hand-written stacks (`train_llama.py`, `train_bert.py`, …) also take
JSON `--config` / `--checkpoint`, but always train on `device=nntile`
and use `torch_nntile.models.*` (plus `_C` helpers such as `gemm` /
`rms_norm_forward`). They are **not** in the CPU column of the table.

Example::

```bash
python torch_nntile/examples/train_llama.py train \
  --seed 0 --config llama_tiny_config.json \
  --ncpu 1 --steps 2 --output-dir /tmp/llama_native
```

On a libtorch-only / incomplete `_C` build these may fail with missing
symbols; use a full torch_nntile extension build (CI wheel or local
`cmake --build … --target torch_nntile` with model bindings enabled).

## Related

- CNN counterpart: [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md)
- DiT counterpart: [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md)
- [torch_nntile_tensor_architecture.md](torch_nntile_tensor_architecture.md)
- [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md)
- [torch_starpu_kernels.md](torch_starpu_kernels.md)
- Product overview: [../torch_nntile.md](../torch_nntile.md)
