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

python torch_nntile/examples/bench_torch_native_middle_cpu_vs_nntile.py \
  --families hf,cnn,dit \
  --markdown-out /tmp/torch_native_middle.md
```

## Results (CPU vs nntile, `ncpu=1`)

Measured on the Cloud Agent VM (CPU-only StarPU / `USE_CUDA=OFF`,
`host_threads=1`, `seed=0`, date 2026-07-18). Walls are **train-loop
only**. Tiny-smoke nntile/CPU ratios were typically **3–6×**; middle
recipes land near **1.0–2.0×** for most models.

### hf

| Model | steps | batch | seq | CPU loss | nntile loss | CPU wall (s) | nntile wall (s) | nntile/CPU | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 32 | 8 | 256 | 7.709977 | 7.709976 | 51.849 | 52.486 | 1.01x | 1.000e-06 | OK |
| gpt-neo | 28 | 8 | 256 | 7.620250 | 7.620250 | 43.360 | 61.457 | 1.42x | 0.000e+00 | OK |
| gpt-neox | 30 | 8 | 256 | 7.678890 | 7.678890 | 42.430 | 84.284 | 1.99x | 0.000e+00 | OK |
| llama | 24 | 8 | 256 | 7.612030 | 7.612030 | 44.777 | 77.397 | 1.73x | 0.000e+00 | OK |
| bert | 32 | 8 | 256 | 7.593865 | 7.578587 | 45.654 | 59.186 | 1.30x | 1.528e-02 | OK |
| roberta | 32 | 8 | 256 | 7.595626 | 7.595358 | 45.867 | 59.874 | 1.31x | 2.680e-04 | OK |
| t5 | 12 | 4 | 192 | 8.274581 | 8.274585 | 15.496 | 39.751 | 2.57x | 4.000e-06 | OK |

T5 uses fewer steps than other HF models: many micro-steps inflate
per-step `compile_graph` cost on the encoder–decoder graph (70 tiny
steps previously reached ~12×). Prefer heavier steps over long step
counts for T5 overhead reads.

### cnn

| Model | steps | batch | seq | CPU loss | nntile loss | CPU wall (s) | nntile wall (s) | nntile/CPU | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lenet | 30 | 32 | — | 0.493463 | 0.483910 | 49.763 | 61.999 | 1.25x | 9.553e-03 | OK |
| resnet | 12 | 16 | — | 2.240705 | 2.240705 | 55.368 | 66.583 | 1.20x | 0.000e+00 | OK |
| vgg | 64 | 8 | — | 2.151951 | 2.151951 | 47.119 | 51.857 | 1.10x | 0.000e+00 | OK |
| mobilenet | 40 | 16 | — | 0.259883 | 0.263021 | 45.060 | 64.550 | 1.43x | 3.138e-03 | OK |
| unet | 64 | 4 | — | 1.104982 | 1.104982 | 47.571 | 59.755 | 1.26x | 0.000e+00 | OK |
| unet_modern | 60 | 4 | — | 1.107599 | 1.107598 | 49.367 | 57.875 | 1.17x | 1.000e-06 | OK |

### dit

| Model | steps | batch | seq | CPU loss | nntile loss | CPU wall (s) | nntile wall (s) | nntile/CPU | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dit | 12 | 8 | — | 1.305994 | 1.305994 | 44.826 | 64.525 | 1.44x | 0.000e+00 | OK |

## Takeaways

1. **Correctness:** most models match to printing precision; a few
   multi-step runs (BERT / LeNet / MobileNet) show small FP drift in
   final loss after tens of steps.
2. **Overhead vs tiny:** GPT-2 is essentially **1.01×**; CNN middle
   recipes are **1.1–1.4×**; causal LMs land **1.3–2.0×**. Tiny smokes
   were often **3–6×** — larger work amortizes StarPU submit +
   compile/run + host sync.
3. **GPU follow-up:** re-run the same recipes on a CUDA build using the
   checklist in [reproducibility.md](reproducibility.md) to see whether
   nntile still shows overhead vs torch CUDA at this size.

## Related

- [reproducibility.md](reproducibility.md)
- [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md)
- [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md)
- [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md)
