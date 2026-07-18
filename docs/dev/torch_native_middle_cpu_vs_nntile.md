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
only**. Ratios are `nntile_wall / cpu_wall` (CPU is always single-thread
torch). Tiny-smoke nntile/CPU ratios were typically **3–6×**; middle
recipes land near **1.0–2.0×** at `ncpu=1`, and several models drop
below **1×** at `ncpu=2`.

### hf

| Model | steps | batch | seq | CPU wall (s) | nntile ncpu=1 (s) | nntile ncpu=2 (s) | ncpu1/CPU | ncpu2/CPU | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 32 | 8 | 256 | 51.849 | 52.486 | 38.980 | 1.01x | 0.75x | OK |
| gpt-neo | 28 | 8 | 256 | 43.360 | 61.457 | 50.614 | 1.42x | 1.17x | OK |
| gpt-neox | 30 | 8 | 256 | 42.430 | 84.284 | 63.636 | 1.99x | 1.50x | OK |
| llama | 24 | 8 | 256 | 44.777 | 77.397 | 60.698 | 1.73x | 1.36x | OK |
| bert | 32 | 8 | 256 | 45.654 | 59.186 | 40.995 | 1.30x | 0.90x | OK |
| roberta | 32 | 8 | 256 | 45.867 | 59.874 | 40.891 | 1.31x | 0.89x | OK |
| t5 | 12 | 4 | 192 | 15.496 | 39.751 | 33.875 | 2.57x | 2.19x | OK |

Final losses (CPU / nntile) match the prior `ncpu=1` run to printing
precision for these seeds (BERT/RoBERTa keep the same small FP drift).
T5 uses fewer steps than other HF models: many micro-steps inflate
per-step `compile_graph` cost on the encoder–decoder graph.

### cnn

| Model | steps | batch | CPU wall (s) | nntile ncpu=1 (s) | nntile ncpu=2 (s) | ncpu1/CPU | ncpu2/CPU | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| lenet | 30 | 32 | 49.763 | 61.999 | 52.094 | 1.25x | 1.05x | OK |
| resnet | 12 | 16 | 55.368 | 66.583 | 69.375 | 1.20x | 1.25x | OK |
| vgg | 64 | 8 | 47.119 | 51.857 | 53.652 | 1.10x | 1.14x | OK |
| mobilenet | 40 | 16 | 45.060 | 64.550 | 64.982 | 1.43x | 1.44x | OK |
| unet | 64 | 4 | 47.571 | 59.755 | 58.969 | 1.26x | 1.24x | OK |
| unet_modern | 60 | 4 | 49.367 | 57.875 | 59.086 | 1.17x | 1.20x | OK |

### dit

| Model | steps | batch | CPU wall (s) | nntile ncpu=1 (s) | nntile ncpu=2 (s) | ncpu1/CPU | ncpu2/CPU | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dit | 12 | 8 | 44.826 | 64.525 | 51.863 | 1.44x | 1.16x | OK |

## Takeaways

1. **Correctness:** most models match to printing precision; a few
   multi-step runs (BERT / LeNet / MobileNet) show small FP drift in
   final loss after tens of steps.
2. **`ncpu=1` vs tiny:** GPT-2 is essentially **1.01×**; CNN middle
   recipes are **1.1–1.4×**; causal LMs land **1.3–2.0×**. Tiny smokes
   were often **3–6×**.
3. **`ncpu=2`:** HF models improve further (GPT-2 / BERT / RoBERTa
   beat single-thread torch). Some CNNs (ResNet / VGG / MobileNet) do
   not speed up — limited parallelism in the untiled graph at this size.
4. **GPU follow-up:** re-run the same recipes on a CUDA build using the
   checklist in [reproducibility.md](reproducibility.md).

## Related

- [reproducibility.md](reproducibility.md)
- [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md)
- [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md)
- [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md)
