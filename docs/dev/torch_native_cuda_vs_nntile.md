# Torch-native train: CUDA vs `device=nntile` (single GPU)

Same tiny and middle recipes as the CPU showcases, run on one GPU with
`CUDA_VISIBLE_DEVICES=1`. Compares plain PyTorch `--device cuda` to
`device=nntile` with one StarPU CUDA worker
(`--ncpu 0 --ncuda 1 --restrict-cuda`). TF32 is disabled on both sides
for full FP32 parity.

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
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1

python torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py \
  --suite tiny --families hf,cnn,dit \
  --markdown-out /tmp/torch_native_tiny_cuda.md

python torch_nntile/examples/bench_torch_native_cuda_vs_nntile.py \
  --suite middle --families hf,cnn,dit \
  --markdown-out /tmp/torch_native_middle_cuda.md
```

HF / CNN / DiT train commons accept `--device cuda` the same way as GPT-2
HF. Walls are **train-loop only** (printed `wall=…s` / GPT-2
`timing … train wall`).

## Host

Measured 2026-07-21 on NVIDIA A40 (driver 550.54.15, 46 GiB),
`torch==2.9.1+cu129`, `USE_CUDA=ON` install prefix `install/lib`,
`seed=0`, `host_threads=1`, `CUDA_VISIBLE_DEVICES=1`.

**Acceleration** (higher is better for nntile):
`Accel = CUDA_wall / nntile_wall` (>1 ⇒ nntile faster than torch CUDA).

## Tiny suite

`steps=1` (HF `seq-len=16`, `batch=1`; CNN/DiT `batch=2`).

### hf

| Model | CUDA loss | nntile loss | CUDA (s) | nntile (s) | Accel | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 5.552704 | 5.552704 | 0.546 | 0.222 | 2.46x | 0 | OK |
| gpt-neo | 4.827158 | 4.827158 | 0.250 | 0.213 | 1.17x | 0 | OK |
| gpt-neox | 4.835960 | 4.835960 | 0.223 | 0.262 | 0.85x | 0 | OK |
| llama | 4.915325 | 4.915325 | 0.265 | 0.274 | 0.97x | 0 | OK |
| bert | 4.833462 | 4.833462 | 0.216 | 0.215 | 1.00x | 0 | OK |
| roberta | 4.735972 | 4.735972 | 0.213 | 0.217 | 0.98x | 0 | OK |
| t5 | 5.692081 | 5.692081 | 0.317 | 0.344 | 0.92x | 0 | OK |

### cnn

| Model | CUDA loss | nntile loss | CUDA (s) | nntile (s) | Accel | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| lenet | 2.230573 | 2.230573 | 0.557 | 0.297 | 1.87x | 0 | OK |
| resnet | 1.905450 | 1.905450 | 0.700 | 0.283 | 2.47x | 0 | OK |
| vgg | 2.474704 | 2.474704 | 0.624 | 0.295 | 2.12x | 0 | OK |
| mobilenet | 2.080944 | 2.080944 | 0.340 | 0.286 | 1.19x | 0 | OK |
| unet | 1.160912 | 1.160912 | 0.613 | 0.362 | 1.69x | 0 | OK |
| unet_modern | 1.164545 | 1.164545 | 0.561 | 0.403 | 1.39x | 0 | OK |

### dit

| Model | CUDA loss | nntile loss | CUDA (s) | nntile (s) | Accel | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| dit | 1.603470 | 1.603470 | 0.559 | 0.395 | 1.42x | 0 | OK |

## Middle suite

Same configs / steps / batches as
[`torch_native_middle_recipes.json`](../../torch_nntile/examples/torch_native_middle_recipes.json)
(~1 min on one CPU core; much shorter on A40).

### hf

| Model | steps | batch | seq | CUDA loss | nntile loss | CUDA (s) | nntile (s) | Accel | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gpt2 | 32 | 8 | 256 | 7.709977 | 7.709977 | 1.238 | 2.708 | 0.46x | 0 | OK |
| gpt-neo | 28 | 8 | 256 | 7.620250 | 7.620250 | 1.106 | 30.854 | 0.04x | 0 | OK |
| gpt-neox | 30 | 8 | 256 | 7.678890 | 7.678890 | 0.880 | 54.297 | 0.02x | 0 | OK |
| llama | 24 | 8 | 256 | 7.612030 | 7.612030 | 1.250 | 47.716 | 0.03x | 0 | OK |
| bert | 32 | 8 | 256 | 7.593865 | 7.578586 | 1.046 | 11.486 | 0.09x | 1.5e-2 | OK |
| roberta | 32 | 8 | 256 | 7.595626 | 7.595359 | 0.995 | 10.821 | 0.09x | 2.7e-4 | OK |
| t5 | 12 | 4 | 192 | 8.274583 | 8.274585 | 0.731 | 37.118 | 0.02x | 2e-6 | OK |

### cnn

| Model | steps | batch | CUDA loss | nntile loss | CUDA (s) | nntile (s) | Accel | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lenet | 30 | 32 | 0.481605 | 0.489254 | 0.946 | 1.636 | 0.58x | 7.6e-3 | OK |
| resnet | 12 | 16 | 2.240705 | 2.240705 | 1.085 | 2.329 | 0.47x | 0 | OK |
| vgg | 64 | 8 | 2.151951 | 2.151951 | 0.829 | 2.939 | 0.28x | 0 | OK |
| mobilenet | 40 | 16 | 0.258925 | 0.267127 | 1.135 | 6.049 | 0.19x | 8.2e-3 | OK |
| unet | 64 | 4 | 1.104982 | 1.104981 | 1.488 | 11.391 | 0.13x | 1e-6 | OK |
| unet_modern | 60 | 4 | 1.107598 | 1.107598 | 1.603 | 10.700 | 0.15x | 0 | OK |

### dit

| Model | steps | batch | CUDA loss | nntile loss | CUDA (s) | nntile (s) | Accel | Δ loss | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dit | 12 | 8 | 1.305994 | 1.305994 | 1.220 | 12.006 | 0.10x | 0 | OK |

## Takeaways

1. **Correctness:** tiny losses match to printing precision on every
   model. Middle keeps the same pattern as CPU: most match exactly; a
   few multi-step runs (BERT / RoBERTa / LeNet / MobileNet) show small
   FP drift after tens of steps.
2. **Tiny GPU walls:** Accel is often **≥1×** (nntile competitive or
   faster) because both paths are overhead-dominated; absolute walls
   are sub-second.
3. **Middle GPU walls:** torch CUDA stays ~1 s while nntile stretches to
   tens of seconds on several HF models (per-step graph compile / sync
   still dominate vs fused CUDA). CNN / DiT land Accel **~0.1–0.6×**.
   This is the GPU follow-up called out in
   [torch_native_middle_cpu_vs_nntile.md](torch_native_middle_cpu_vs_nntile.md).

## Related

- [reproducibility.md](reproducibility.md)
- [torch_native_middle_cpu_vs_nntile.md](torch_native_middle_cpu_vs_nntile.md)
- [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md)
- [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md)
- [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md)
