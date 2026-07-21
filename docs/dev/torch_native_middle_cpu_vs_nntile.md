# Middle torch-native train: CPU vs `device=nntile`

Larger stock-model recipes (HF / CNN / Diffusers DiT) sized so each
**train-loop wall is ~1 minute on a single CPU core**. Goal: show that
StarPU / graph overhead seen on [tiny smokes](hf_tiny_cpu_vs_nntile_showcase.md)
becomes a smaller fraction of wall time as models and batches grow.

Configs: `torch_nntile/examples/*_middle_config.json`  
Recipes: [`torch_native_middle_recipes.json`](../../torch_nntile/examples/torch_native_middle_recipes.json)  
Bench: [`bench_torch_native_middle_cpu_vs_nntile.py`](../../torch_nntile/examples/bench_torch_native_middle_cpu_vs_nntile.py)  
Protocol: [reproducibility.md](reproducibility.md)

## How to run

```bash
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig
export LD_LIBRARY_PATH=$PWD/build/nntile:$PWD/build/torch_nntile:/opt/starpu/lib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1

# CPU + nntile (ncpu=1)
python torch_nntile/examples/bench_torch_native_middle_cpu_vs_nntile.py \
  --families hf,cnn,dit \
  --markdown-out /tmp/torch_native_middle.md

# nntile with two StarPU CPU workers (compare to CPU column above)
python torch_nntile/examples/bench_torch_native_middle_cpu_vs_nntile.py \
  --families hf,cnn,dit --devices nntile --ncpu 2 \
  --markdown-out /tmp/torch_native_middle_ncpu2.md
```

Host BLAS / PyTorch stay single-threaded (`OMP_NUM_THREADS=1`); only
StarPU worker count changes via `--ncpu`.

## Results (CPU vs nntile `ncpu=1` / `ncpu=2`)

Measured on the Cloud Agent VM (CPU-only StarPU / `USE_CUDA=OFF`,
`host_threads=1`, `seed=0`, date 2026-07-18). Walls are **train-loop
only**. CPU is always single-thread torch.

**Acceleration ratios** (higher is better for nntile):

- `Accel@1` = `CPU_wall / nntile_ncpu1_wall` (>1 ⇒ nntile faster than CPU)
- `Accel@2` = `CPU_wall / nntile_ncpu2_wall`
- `Accel(1→2)` = `nntile_ncpu1_wall / nntile_ncpu2_wall`
  (>1 ⇒ second StarPU worker speeds up nntile)

Tiny-smoke Accel@1 values were typically **0.15–0.35**; middle recipes
land near **0.5–1.0** at `ncpu=1`, and several models exceed **1.0** at
`ncpu=2`.

### hf

| Model | steps | batch | seq | CPU (s) | nntile₁ (s) | nntile₂ (s) | Accel@1 | Accel@2 | Accel(1→2) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 32 | 8 | 256 | 51.849 | 52.486 | 38.980 | 0.99x | 1.33x | 1.35x | OK |
| gpt-neo | 28 | 8 | 256 | 43.360 | 61.457 | 50.614 | 0.71x | 0.86x | 1.21x | OK |
| gpt-neox | 30 | 8 | 256 | 42.430 | 84.284 | 63.636 | 0.50x | 0.67x | 1.32x | OK |
| llama | 24 | 8 | 256 | 44.777 | 77.397 | 60.698 | 0.58x | 0.74x | 1.28x | OK |
| bert | 32 | 8 | 256 | 45.654 | 59.186 | 40.995 | 0.77x | 1.11x | 1.44x | OK |
| roberta | 32 | 8 | 256 | 45.867 | 59.874 | 40.891 | 0.77x | 1.12x | 1.46x | OK |
| t5 | 12 | 4 | 192 | 15.496 | 39.751 | 33.875 | 0.39x | 0.46x | 1.17x | OK |

Final losses (CPU / nntile) match the prior `ncpu=1` run to printing
precision for these seeds (BERT/RoBERTa keep the same small FP drift).
T5 uses fewer steps than other HF models: many micro-steps inflate
per-step `compile_graph` cost on the encoder–decoder graph.

### cnn

| Model | steps | batch | CPU (s) | nntile₁ (s) | nntile₂ (s) | Accel@1 | Accel@2 | Accel(1→2) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lenet | 30 | 32 | 49.763 | 61.999 | 52.094 | 0.80x | 0.96x | 1.19x | OK |
| resnet | 12 | 16 | 55.368 | 66.583 | 69.375 | 0.83x | 0.80x | 0.96x | OK |
| vgg | 64 | 8 | 47.119 | 51.857 | 53.652 | 0.91x | 0.88x | 0.97x | OK |
| mobilenet | 40 | 16 | 45.060 | 64.550 | 64.982 | 0.70x | 0.69x | 0.99x | OK |
| unet | 64 | 4 | 47.571 | 59.755 | 58.969 | 0.80x | 0.81x | 1.01x | OK |
| unet_modern | 60 | 4 | 49.367 | 57.875 | 59.086 | 0.85x | 0.84x | 0.98x | OK |

### dit

| Model | steps | batch | CPU (s) | nntile₁ (s) | nntile₂ (s) | Accel@1 | Accel@2 | Accel(1→2) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dit | 12 | 8 | 44.826 | 64.525 | 51.863 | 0.69x | 0.86x | 1.24x | OK |

## Takeaways

1. **Correctness:** most models match to printing precision; a few
   multi-step runs (BERT / LeNet / MobileNet) show small FP drift in
   final loss after tens of steps.
2. **`ncpu=1` vs tiny:** Accel@1 for GPT-2 is essentially **0.99×**; CNN
   middle recipes are **0.7–0.9×**; causal LMs land **0.5–0.8×**. Tiny
   smokes were often **0.15–0.35×**.
3. **`ncpu=2`:** Accel(1→2) is **1.2–1.5×** for most HF models (GPT-2 /
   BERT / RoBERTa beat single-thread torch at Accel@2 **>1**). Some CNNs
   (ResNet / VGG / MobileNet) do not speed up — limited parallelism in
   the untiled graph at this size.
4. **GPU follow-up:** re-run on a CUDA build — results in
   [torch_native_cuda_vs_nntile.md](torch_native_cuda_vs_nntile.md)
   (checklist in [reproducibility.md](reproducibility.md)).

## Related

- [reproducibility.md](reproducibility.md)
- [torch_native_cuda_vs_nntile.md](torch_native_cuda_vs_nntile.md)
- [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md)
- [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md)
- [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md)
