# Torch-native train: CUDA vs `device=nntile`

Same tiny and middle recipes as the CPU showcases. Compares plain
PyTorch `--device cuda` to `device=nntile` with StarPU CUDA workers
(`--ncpu 0 --ncuda {1|2} --restrict-cuda`). TF32 is disabled on both
sides for full FP32 parity.

Bench: [`bench_torch_native_cuda_vs_nntile.py`](../../torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py)  
Protocol: [reproducibility.md](reproducibility.md)  
CPU counterparts: [hf_tiny…](hf_tiny_cpu_vs_nntile_showcase.md),
[cnn_tiny…](cnn_tiny_cpu_vs_nntile_showcase.md),
[dit_tiny…](dit_tiny_cpu_vs_nntile_showcase.md),
[middle CPU](torch_native_middle_cpu_vs_nntile.md)

## How to run

```bash
export LD_LIBRARY_PATH=$PWD/install/lib:/opt/conda/envs/nntile/lib:$LD_LIBRARY_PATH
# plus pip nvidia-* /lib entries if needed for cuBLAS / cuDNN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1

# nntile with 1 CUDA worker (one physical GPU)
export CUDA_VISIBLE_DEVICES=1
python torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py \
  --suite tiny --families hf,cnn,dit --ncpu 0 --ncuda 1

# nntile with 2 CUDA workers (two physical GPUs)
export CUDA_VISIBLE_DEVICES=1,2
python torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py \
  --suite tiny --families hf,cnn,dit --devices nntile --ncpu 0 --ncuda 2

# Same for middle recipes
python torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py \
  --suite middle --families hf,cnn,dit --ncpu 0 --ncuda 1
export CUDA_VISIBLE_DEVICES=1,2
python torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py \
  --suite middle --families hf,cnn,dit --devices nntile --ncpu 0 --ncuda 2
```

HF / CNN / DiT train commons accept `--device cuda` the same way as GPT-2
HF. Walls are **train-loop only** (printed `wall=…s` / GPT-2
`timing … train wall`).

## Host

Measured 2026-07-21 on NVIDIA A40 ×4 (driver 550.54.15, 46 GiB each),
`torch==2.9.1+cu129`, `USE_CUDA=ON` install prefix `install/lib`,
`seed=0`, `host_threads=1`.

- **CUDA** and **nntile₁** (`ncuda=1`): `CUDA_VISIBLE_DEVICES=1`
- **nntile₂** (`ncuda=2`): `CUDA_VISIBLE_DEVICES=1,2`

**Acceleration ratios** (higher is better for nntile):

- `Accel@1` = `CUDA_wall / nntile_ncuda1_wall` (>1 ⇒ nntile faster than CUDA)
- `Accel@2` = `CUDA_wall / nntile_ncuda2_wall`
- `Accel(1→2)` = `nntile_ncuda1_wall / nntile_ncuda2_wall`
  (>1 ⇒ second StarPU CUDA worker speeds up nntile)

## Tiny suite — acceleration

`steps=1` (HF `seq-len=16`, `batch=1`; CNN/DiT `batch=2`). Losses match
CUDA vs nntile to printing precision on every model below.

### hf

| Model | CUDA (s) | nntile₁ (s) | nntile₂ (s) | Accel@1 | Accel@2 | Accel(1→2) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 0.546 | 0.222 | 0.223 | 2.46x | 2.45x | 1.00x | OK |
| gpt-neo | 0.250 | 0.213 | 0.233 | 1.17x | 1.07x | 0.91x | OK |
| gpt-neox | 0.223 | 0.262 | 0.287 | 0.85x | 0.78x | 0.91x | OK |
| llama | 0.265 | 0.274 | 0.624 | 0.97x | 0.42x | 0.44x | OK |
| bert | 0.216 | 0.215 | 0.242 | 1.00x | 0.89x | 0.89x | OK |
| roberta | 0.213 | 0.217 | 0.337 | 0.98x | 0.63x | 0.64x | OK |
| t5 | 0.317 | 0.344 | 0.349 | 0.92x | 0.91x | 0.99x | OK |

### cnn

| Model | CUDA (s) | nntile₁ (s) | nntile₂ (s) | Accel@1 | Accel@2 | Accel(1→2) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| lenet | 0.557 | 0.297 | 0.279 | 1.87x | 2.00x | 1.06x | OK |
| resnet | 0.700 | 0.283 | 0.277 | 2.47x | 2.53x | 1.02x | OK |
| vgg | 0.624 | 0.295 | 0.688 | 2.12x | 0.91x | 0.43x | OK |
| mobilenet | 0.340 | 0.286 | 0.649 | 1.19x | 0.52x | 0.44x | OK |
| unet | 0.613 | 0.362 | 0.730 | 1.69x | 0.84x | 0.50x | OK |
| unet_modern | 0.561 | 0.403 | 0.396 | 1.39x | 1.42x | 1.02x | OK |

### dit

| Model | CUDA (s) | nntile₁ (s) | nntile₂ (s) | Accel@1 | Accel@2 | Accel(1→2) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| dit | 0.559 | 0.395 | 0.399 | 1.42x | 1.40x | 0.99x | OK |

## Middle suite — acceleration

Same configs / steps / batches as
[`torch_native_middle_recipes.json`](../../torch_nntile/examples/torch_native_middle_recipes.json)
(~1 min on one CPU core; much shorter on A40).

### hf

| Model | steps | batch | seq | CUDA (s) | nntile₁ (s) | nntile₂ (s) | Accel@1 | Accel@2 | Accel(1→2) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 32 | 8 | 256 | 1.238 | 2.708 | 3.555 | 0.46x | 0.35x | 0.76x | OK |
| gpt-neo | 28 | 8 | 256 | 1.106 | 30.854 | 32.395 | 0.04x | 0.03x | 0.95x | OK |
| gpt-neox | 30 | 8 | 256 | 0.880 | 54.297 | 59.339 | 0.02x | 0.01x | 0.92x | OK |
| llama | 24 | 8 | 256 | 1.250 | 47.716 | 50.935 | 0.03x | 0.02x | 0.94x | OK |
| bert | 32 | 8 | 256 | 1.046 | 11.486 | 12.819 | 0.09x | 0.08x | 0.90x | OK |
| roberta | 32 | 8 | 256 | 0.995 | 10.821 | 12.696 | 0.09x | 0.08x | 0.85x | OK |
| t5 | 12 | 4 | 192 | 0.731 | 37.118 | 36.222 | 0.02x | 0.02x | 1.02x | OK |

Final losses (CUDA / nntile) match to printing precision for these seeds
except small FP drift on BERT / RoBERTa after many steps (same pattern
as the CPU middle suite).

### cnn

| Model | steps | batch | CUDA (s) | nntile₁ (s) | nntile₂ (s) | Accel@1 | Accel@2 | Accel(1→2) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lenet | 30 | 32 | 0.946 | 1.636 | 1.672 | 0.58x | 0.57x | 0.98x | OK |
| resnet | 12 | 16 | 1.085 | 2.329 | 2.631 | 0.47x | 0.41x | 0.89x | OK |
| vgg | 64 | 8 | 0.829 | 2.939 | 4.248 | 0.28x | 0.20x | 0.69x | OK |
| mobilenet | 40 | 16 | 1.135 | 6.049 | 7.773 | 0.19x | 0.15x | 0.78x | OK |
| unet | 64 | 4 | 1.488 | 11.391 | 13.590 | 0.13x | 0.11x | 0.84x | OK |
| unet_modern | 60 | 4 | 1.603 | 10.700 | 12.839 | 0.15x | 0.12x | 0.83x | OK |

### dit

| Model | steps | batch | CUDA (s) | nntile₁ (s) | nntile₂ (s) | Accel@1 | Accel@2 | Accel(1→2) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dit | 12 | 8 | 1.220 | 12.006 | 12.720 | 0.10x | 0.10x | 0.94x | OK |

## Takeaways

1. **Correctness:** tiny losses match exactly; middle keeps the CPU
   pattern (exact match on most models; small multi-step FP drift on
   BERT / RoBERTa / LeNet / MobileNet).
2. **Tiny GPU walls:** Accel@1 is often **≥1×** (overhead-dominated);
   Accel(1→2) is usually **≤1** — a second CUDA worker does not help at
   this scale (and can hurt).
3. **Middle GPU walls:** torch CUDA stays ~1 s while nntile stretches to
   tens of seconds on several HF models (per-step graph compile / sync
   still dominate vs fused CUDA). Accel(1→2) stays near **0.7–1.0×** —
   unlike several CPU middle HF models where `ncpu=2` beats `ncpu=1`.
   Untiled PrivateUse1 graphs do not yet benefit from a second GPU at
   these sizes.

## Related

- [reproducibility.md](reproducibility.md)
- [torch_native_middle_cpu_vs_nntile.md](torch_native_middle_cpu_vs_nntile.md)
- [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md)
- [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md)
- [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md)
