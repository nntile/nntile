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
`host_threads=1`, `seed=0`, date 2026-07-18). Walls are train-loop only.

_Results table filled after the middle bench run on this branch._

## Takeaways

1. **Correctness:** losses should match to printing precision (same as tiny).
2. **Overhead:** `nntile/CPU` wall ratios should be much closer to **1×** than
   on tiny smokes (where ratios of 3–6× are common).
3. **GPU follow-up:** re-run the same recipes on a CUDA build using the
   checklist in [reproducibility.md](reproducibility.md) to see whether
   nntile still shows overhead vs torch CUDA at this size.

## Related

- [reproducibility.md](reproducibility.md)
- [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md)
- [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md)
- [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md)
