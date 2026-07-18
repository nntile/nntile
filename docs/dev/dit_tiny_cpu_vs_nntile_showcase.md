# Tiny DiT HF train showcase: CPU vs `device=nntile`

Short smoke runs of HuggingFace Diffusers
[`DiTTransformer2DModel`](https://huggingface.co/docs/diffusers/api/models/dit_transformer2d)
through `torch_nntile` PrivateUse1 (`device=nntile`) versus plain PyTorch
CPU. Goal: show **numerical parity** (matching loss) for AdaLN-Zero DiT
on a tiny CIFAR-10 batch — not a large-scale throughput benchmark.

Scripts live under [`torch_nntile/examples/`](../../torch_nntile/examples/):

| Script | Role |
|--------|------|
| [`train_dit_hf.py`](../../torch_nntile/examples/train_dit_hf.py) | Stock Diffusers DiT, JSON config, `train` / `compare` |
| [`dit_hf_tiny_config.json`](../../torch_nntile/examples/dit_hf_tiny_config.json) | Tiny DiT (16×16, 2 layers, 2 heads) |
| [`dit_hf_tiny_train_common.py`](../../torch_nntile/examples/dit_hf_tiny_train_common.py) | CIFAR-10 (`datasets`) batch + MSE noise loss |
| [`bench_dit_hf_tiny_cpu_vs_nntile.py`](../../torch_nntile/examples/bench_dit_hf_tiny_cpu_vs_nntile.py) | CPU vs nntile table helper |

Ops exercised beyond the LM/CNN smokes: StarPU `aten::exp` (sinusoidal
timestep frequencies), contiguous densify for AdaLN `LayerNorm`
(`elementwise_affine=False`), plus existing Conv2d patch embed, SiLU,
GELU-tanh MLP, SDPA, Linear, and host `mean` for MSE.

## How to run

Environment (CPU StarPU build; no CUDA in this Cloud Agent image)::

```bash
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig
export LD_LIBRARY_PATH=$PWD/build/nntile:$PWD/build/torch_nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NNTILE_BUILD_DIR=$PWD/build TORCH_NNTILE_BUILD_DIR=$PWD/build
export NNTILE_SOURCE_DIR=$PWD
export PYTHONPATH=$PWD/torch_nntile${PYTHONPATH:+:$PYTHONPATH}
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
```

Requires `diffusers` and `datasets` (CIFAR-10)::

```bash
pip install 'diffusers==0.32.2'  # or newer; API matches DiTTransformer2DModel
```

Single run::

```bash
# CPU reference
python torch_nntile/examples/train_dit_hf.py train \
  --device cpu --seed 0 --steps 1 --batch-size 2 \
  --output-dir /tmp/dit_cpu

# nntile (one StarPU CPU worker)
python torch_nntile/examples/train_dit_hf.py train \
  --device nntile --seed 0 --steps 1 --batch-size 2 \
  --ncpu 1 --output-dir /tmp/dit_nntile

# Optional weight compare
python torch_nntile/examples/train_dit_hf.py compare \
  --checkpoint-a /tmp/dit_cpu/checkpoint.pt \
  --checkpoint-b /tmp/dit_nntile/checkpoint.pt
```

Batch CPU vs nntile and print a markdown table::

```bash
python torch_nntile/examples/bench_dit_hf_tiny_cpu_vs_nntile.py \
  --ncpu 1 --steps 1 --batch-size 2 --seed 0 \
  --markdown-out /tmp/dit_hf_tiny_cpu_vs_nntile/table.md
```

## What the outputs mean

### Loss

```text
[dit] step 1/1  loss=1.603470
```

- **Loss** is MSE between predicted noise and the synthetic noise target
  on a **CIFAR-10** mini-batch (first `batch_size` images from
  `train[:64]`, resized to `sample_size`, mapped to `[-1, 1]`).
- Timesteps / noise are drawn from `--seed` (deterministic).
- CFG label dropout is disabled for parity (`dropout_prob=0`).
- Same seed + same JSON config ⇒ CPU and nntile match to printing
  precision. In the table below, **Δ loss is exactly 0**.

### Wall time (train loop)

```text
[dit] wall=0.042s  OK
```

That is **only the train loop** (forward → backward → optimizer step →
host loss readout), after the model and batch are already on device.
It does **not** include Python import, Diffusers / datasets download, or
StarPU `init_context` / shutdown.

## Results (CPU vs nntile, `ncpu=1`)

Measured with `bench_dit_hf_tiny_cpu_vs_nntile.py` on the Cloud Agent VM
(CPU-only StarPU / `USE_CUDA=OFF`, `ncpu=1`, `steps=1`, `batch-size=2`,
`seed=0`, date 2026-07-18). Tiny config: 16×16, 2 layers, hidden 16 —
**overhead-dominated**, not a speed contest.

| Model | CPU loss | nntile loss | CPU wall (s) | nntile wall (s) | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---|
| dit | 1.603470 | 1.603470 | 0.007 | 0.042 | 0.000e+00 | OK |

**Takeaways for demos**

1. **Correctness:** noise-prediction MSE matches on CPU and nntile.
2. **Timing at this scale:** nntile wall is higher (StarPU submit +
   compile/run + host sync) while the math is tiny; expect the gap to
   shrink on larger resolution / depth or with CUDA workers.
3. **Checkpoints:** each successful `--output-dir` run writes
   `checkpoint.pt` (`model_state_dict` + Diffusers config dict + seed /
   step). Relative Frobenius after one step stays ~1e-9.

## Related

- HF language-model counterpart:
  [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md)
- CNN counterpart:
  [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md)
- [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md) (`exp`, LayerNorm)
- [torch_starpu_kernels.md](torch_starpu_kernels.md)
- Product overview: [../torch_nntile.md](../torch_nntile.md)
