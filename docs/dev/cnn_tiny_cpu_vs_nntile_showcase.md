# Tiny CNN train showcase: CPU vs `device=nntile`

Short smoke runs of tiny Conv / Pool / BN stacks through
`torch_nntile` PrivateUse1 (`device=nntile`) versus plain PyTorch CPU.
Goal: show **numerical parity** (matching loss) for the StarPU-backed
CNN aten path — not a large-scale throughput benchmark.

Scripts live under [`torch_nntile/examples/`](../../torch_nntile/examples/).
Shared helpers: [`cnn_tiny_train_common.py`](../../torch_nntile/examples/cnn_tiny_train_common.py).
CLI shape matches the HF smokes (`train` / `compare`).

| Script | Model | Ops exercised |
|--------|--------|----------------|
| [`train_lenet_tiny.py`](../../torch_nntile/examples/train_lenet_tiny.py) | Tiny LeNet | `convolution_overrideable`, `max_pool2d_with_indices`, ReLU, Linear |
| [`train_resnet_tiny.py`](../../torch_nntile/examples/train_resnet_tiny.py) | Tiny ResNet | Conv, `native_batch_norm`, inplace `relu_`, residual `add`, AdaptiveAvgPool2d |
| [`train_vgg_tiny.py`](../../torch_nntile/examples/train_vgg_tiny.py) | Tiny VGG | Stacked Conv / ReLU / MaxPool + AdaptiveAvgPool head |
| [`train_mobilenet_tiny.py`](../../torch_nntile/examples/train_mobilenet_tiny.py) | Tiny MobileNet | Depthwise (`groups=C`) + pointwise 1×1 + BN |
| [`train_unet_tiny.py`](../../torch_nntile/examples/train_unet_tiny.py) | Classic U-Net | Encoder / decoder, skip `cat`, `ConvTranspose2d`, pixel CE |
| [`train_unet_modern_tiny.py`](../../torch_nntile/examples/train_unet_modern_tiny.py) | Modern U-Net | Same, but upsample via `F.interpolate` (`upsample_bilinear2d` / nearest) |

Classic U-Net uses learnable **`ConvTranspose2d`**. Modern U-Net uses
**`F.interpolate(..., mode="bilinear"|"nearest")`** then a 1×1 reduce
before the skip `cat` — the common post-2018 pattern. Pixel CE is still
flattened to 1D (`nll_loss`) because `nll_loss2d` is not registered.

## How to run

Environment (CPU StarPU build; no CUDA in this Cloud Agent image)::

```bash
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig
export LD_LIBRARY_PATH=$PWD/build/nntile:$PWD/build/torch_nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NNTILE_BUILD_DIR=$PWD/build TORCH_NNTILE_BUILD_DIR=$PWD/build
export NNTILE_SOURCE_DIR=$PWD
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
```

Single model (example: modern U-Net)::

```bash
# CPU reference
python torch_nntile/examples/train_unet_modern_tiny.py train \
  --device cpu --seed 0 --steps 1 --batch-size 2 \
  --output-dir /tmp/unet_modern_cpu

# nntile (one StarPU CPU worker)
python torch_nntile/examples/train_unet_modern_tiny.py train \
  --device nntile --seed 0 --steps 1 --batch-size 2 \
  --ncpu 1 --output-dir /tmp/unet_modern_nntile

# Optional weight compare (relative Frobenius per tensor)
python torch_nntile/examples/train_unet_modern_tiny.py compare \
  --checkpoint-a /tmp/unet_modern_cpu/checkpoint.pt \
  --checkpoint-b /tmp/unet_modern_nntile/checkpoint.pt
```

Batch all CNN smokes and print a markdown table::

```bash
python torch_nntile/examples/bench_cnn_tiny_cpu_vs_nntile.py \
  --ncpu 1 --steps 1 --batch-size 2 --seed 0 \
  --markdown-out /tmp/cnn_tiny_cpu_vs_nntile/table.md
```

## What the outputs mean

### Loss

Each script prints one line per train step, for example::

```text
[unet_modern] step 1/1  loss=1.164545
```

- **Classification** (LeNet / ResNet / VGG / MobileNet): cross-entropy on
  a synthetic image mini-batch.
- **U-Net / modern U-Net:** flattened pixel-wise CE on synthetic NCHW
  logits vs NHW labels (same seed ⇒ same batch).
- Same seed + same config ⇒ CPU and nntile should match to printing
  precision when the aten graph is covered. In the table below, **Δ loss
  is exactly 0** for every model (bit-identical printed values).

### Wall time (train loop)

Tiny CNN smokes print::

```text
[unet_modern] wall=0.043s  OK
```

That is **only the train loop** (forward → backward → optimizer step →
host loss readout), after the model and batch are already on device.
It does **not** include Python import, model construction, or StarPU
`init_context` / shutdown.

On **nntile**, each step also `compile_graph` / `run`s and syncs loss
with `.cpu()` so StarPU reclaim stays in the same phase (see
[torch_nntile_tensor_architecture.md](torch_nntile_tensor_architecture.md)).
That sync is part of the reported wall.

### Process elapsed (not in the table)

Spawning a fresh Python process per run costs ~1–3 s here (imports).
Do **not** treat process elapsed as a device comparison — use the
printed train-loop wall.

## Results (CPU vs nntile, `ncpu=1`)

Measured with `bench_cnn_tiny_cpu_vs_nntile.py` on the Cloud Agent VM
(CPU-only StarPU / `USE_CUDA=OFF`, `ncpu=1`, `steps=1`, `batch-size=2`,
`seed=0`, `OMP_NUM_THREADS=1` / `torch.set_num_threads(1)`, date
2026-07-18). Tiny configs (16–28 spatial, 8–16 channels) —
**overhead-dominated**, not a speed contest. Single-core protocol:
[reproducibility.md](reproducibility.md).

| Model | CPU loss | nntile loss | CPU wall (s) | nntile wall (s) | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---|
| lenet | 2.230573 | 2.230573 | 0.004 | 0.006 | 0.000e+00 | OK |
| resnet | 1.905449 | 1.905449 | 0.004 | 0.008 | 0.000e+00 | OK |
| vgg | 2.474704 | 2.474704 | 0.005 | 0.009 | 0.000e+00 | OK |
| mobilenet | 2.080944 | 2.080944 | 0.004 | 0.010 | 0.000e+00 | OK |
| unet | 1.160912 | 1.160912 | 0.012 | 0.021 | 0.000e+00 | OK |
| unet_modern | 1.164545 | 1.164545 | 0.009 | 0.019 | 0.000e+00 | OK |

**Takeaways for demos**

1. **Correctness:** losses match on CPU and nntile for every model above
   (classic transpose U-Net and interpolate-based modern U-Net).
2. **Timing at this scale:** nntile wall is higher (StarPU submit +
   compile/run + host sync) while the math is tiny; middle CNN recipes
   shrink the gap to ~1.1–1.4× — see
   [torch_native_middle_cpu_vs_nntile.md](torch_native_middle_cpu_vs_nntile.md).
3. **Checkpoints:** each successful `--output-dir` run writes
   `checkpoint.pt` (`model_state_dict` + config dict + seed / step).
   Use `compare` for relative Frobenius norms.

## Related

- Middle (~1 min) overhead table:
  [torch_native_middle_cpu_vs_nntile.md](torch_native_middle_cpu_vs_nntile.md)
- Measurement protocol (CPU / GPU):
  [reproducibility.md](reproducibility.md)
- HF language-model counterpart:
  [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md)
- DiT counterpart:
  [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md)
- [torch_nntile_aten_ops.md](torch_nntile_aten_ops.md) (CNN PrivateUse1 list)
- [torch_starpu_kernels.md](torch_starpu_kernels.md)
- Product overview: [../torch_nntile.md](../torch_nntile.md)
