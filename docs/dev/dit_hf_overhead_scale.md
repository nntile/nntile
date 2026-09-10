# DiT HF: graph overhead vs width / patch count

**Notation.** Each label is **implementation(backend)**. The word
*outside* the brackets is the implementation; the word *inside* is the
backend.

- **HF** — HuggingFace Diffusers `DiTTransformer2DModel`
  (`diffusers==0.32.2`).
- **nntile** (as implementation) — `torch_nntile.models.dit.DiT`, based on
  `torch_nntile.nn` operations and backed by hand-written nntile kernels.
- **cuda** — PyTorch CUDA (`device=cuda`).
- **nntile** (as backend) — StarPU / nntile (`device=nntile`).

**HF(cuda)** is Diffusers on CUDA. **HF(nntile)** is the same Diffusers
graph on `device=nntile`. **nntile(nntile)** is the
`torch_nntile.models` rewrite on `device=nntile`.

Three setups, same configs / patch counts / 10 steps:

1. **HF(cuda)** — stock Diffusers `DiTTransformer2DModel`, `device=cuda`.
2. **HF(nntile)** — same HF model on `device=nntile` (aten / torch-native
   StarPU codelets).
3. **nntile(nntile)** —
   `torch_nntile.models.dit.DiT` (hand-written nntile kernels). Host patchify
   + integer timesteps; HF is used only to init weights.

Three-setup loss and wall: [Three setups](#three-setups). HF(cuda) /
HF(nntile) 10-repeat detail is below that. nntile(nntile) is in
[nntile(nntile) vs HF(cuda)](#nntilenntile-vs-hfcuda).

**VRAM ladder (matched to Llama HF(cuda) peaks).** Hidden size follows the Llama
overhead rungs (1536 … 5760). `sample_size` is set so patch count
`(sample_size / patch_size)²` is close to Llama `seq_len` at each rung.
**L is 11 layers** and **XL is 5 layers** so **nntile(nntile) stays on-GPU**
(D2H **0**).

> **VRAM / nntile.** nntile allocates extra graph buffers. If that footprint
> no longer fits, StarPU **pages CPU↔GPU** and those transfers dominate the
> wall. This study used one **NVIDIA A40** per job (`CUDA_VISIBLE_DEVICES`);
> do not overlap processes on one GPU. nntile(nntile) L **42.7 GiB**, XL
> **43.6 GiB**, D2H **0** on every size. See
> [Peak VRAM and bus](#peak-vram-and-bus).

Configs: [`torch_nntile/examples/overhead_dit/`](../../torch_nntile/examples/overhead_dit/).
HF(cuda) / HF(nntile): [`train_dit_hf_overhead.py`](../../torch_nntile/examples/train_dit_hf_overhead.py),
[`run_dit_overhead_benchmark.py`](../../torch_nntile/tools/run_dit_overhead_benchmark.py).
nntile(nntile): [`train_nntile_native_overhead.py`](../../torch_nntile/examples/train_nntile_native_overhead.py)
(`--family dit`),
[`run_nntile_native_overhead_benchmark.py`](../../torch_nntile/tools/run_nntile_native_overhead_benchmark.py).

## Model and data

- **HF:** Diffusers `DiTTransformer2DModel` (AdaLN-Zero, `patch_size=2`,
  `in_channels=3`). Class/timestep conditioning; label dropout disabled
  (`disable_dit_label_dropout`) for deterministic runs.
- **nntile(nntile):** `torch_nntile.models.dit.DiT`. Patchify NCHW and
  `nchw_to_unpatchify_tokens` on the host; timesteps are integer table
  indices. AdaLN-Zero uses six `Linear(H, H)` (classic `narrow` is wrong
  for `start ≠ 0`).
- **Batch:** `make_synthetic_diffusion_batch()` — random `noisy` / `noise`
  tensors, timesteps, class labels; seed `42 + step`.
- **Optimizer:** SGD, lr `1e-3`, B=1, 10 steps (100 for long S), `--no-shuffle`.
- **CUDA / nntile:** `--disable-tf32 --disable-cudnn` (IEEE FP32 GEMM; no cuDNN, including patch-embed `Conv2d`). **nntile also:** `--ncpu 0 --ncuda 1 --restrict-cuda`.

## Loss

HF: NCHW MSE `model(noisy, timestep, class_labels)` vs ground-truth `noise`.
nntile(nntile): same mean SSE in token layout (`p, p, C`) after host
`nchw_to_unpatchify_tokens`. All three setups match to printed 1e-6.

## Train wall

Same protocol as
[`gpt2_hf_overhead_scale.md`](gpt2_hf_overhead_scale.md): nntile
`record → compile → wait(prev) → run`, wall from first record through final
`wait()`; HF(cuda) synchronized per iter. Prefetch outside the wall. Iter 1 nntile
`wait=0`; iter 10 `wait` includes the final join.

## Recipe

| | XS | S | M | L | XL |
|--|--:|--:|--:|--:|--:|
| Config | `dit_xs.json` | `dit_s.json` | `dit_m.json` | `dit_l.json` | `dit_xl.json` |
| `num_layers` | 11 | 10 | 11 | **11** | **5** |
| hidden (`heads×head_dim`) | 1536 (24×64) | 2048 (16×128) | 3072 (24×128) | 4096 (32×128) | 5760 (45×128) |
| `sample_size` | 56 | 64 | 78 | 90 | 108 |
| patches `T` (`(size/2)²`) | **784** | **1024** | **1521** | **2025** | **2916** |
| HF(cuda) VRAM (smi, 10-step) | ~4.8 GiB | ~7.6 GiB | ~17.9 GiB | **31.4 GiB** | **29.2 GiB** |

NVIDIA A40, one GPU per job, **10 repeats** per configuration.
Includes **S HF(nntile) 100-step** steady-state run.
HF(cuda) and HF(nntile) both use `--disable-tf32 --disable-cudnn`. Requires
`diffusers==0.32.2` (see
[`reproducibility.md`](reproducibility.md)).
nntile(nntile): **10 repeats** (mean ± stdev), `STARPU_LIMIT_CUDA_MEM=46000`.

## Three setups

Same recipe. Walls are **10-repeat** means. nntile(nntile) record
breakdown is in
[nntile(nntile) vs HF(cuda)](#nntilenntile-vs-hfcuda).

### Loss

| Setup | HF(cuda) | HF(nntile) | nntile(nntile) |
|-------|-----:|----------------:|----------------:|
| XS T=784 | 1.209802 | 1.209802 | 1.209802 |
| S T=1024 | 1.192550 | 1.192550 | 1.192550 |
| M T=1521 | 1.141145 | 1.141145 | 1.141145 |
| L T=2025 | 1.221610 | 1.221610 | 1.221610 |
| XL T=2916 | 1.034324 | 1.034324 | 1.034324 |

All three setups match to printed 1e-6.

### 10-step train wall

**10 repeats** (mean ± stdev), `STARPU_LIMIT_CUDA_MEM=46000`.

| Setup | HF(cuda) | HF(nntile) | nntile(nntile) | HF(nntile) / HF(cuda) | nntile(nntile) / HF(cuda) |
|-------|-----:|---------:|----------------:|--------------:|-------------:|
| XS T=784 | 1.393 ± 0.006 s | 1.898 ± 0.044 s | 1.976 ± 0.031 s | **1.36×** | **1.42×** |
| S T=1024 | 2.577 ± 0.008 s | 2.834 ± 0.011 s | 2.871 ± 0.008 s | **1.10×** | **1.11×** |
| M T=1521 | 7.720 ± 0.022 s | 7.756 ± 0.038 s | 8.225 ± 0.013 s | **1.00×** | **1.07×** |
| L T=2025 | 17.346 ± 0.028 s | 17.007 ± 0.034 s | 17.880 ± 0.028 s | **0.98×** | **1.03×** |
| XL T=2916 | 21.796 ± 0.074 s | 21.015 ± 0.061 s | 22.421 ± 0.114 s | **0.96×** | **1.03×** |

nntile(nntile) walls are the published classic-kernel 10-repeat means (FP32; not rerun). HF(cuda) / HF(nntile) are this `--disable-tf32 --disable-cudnn` campaign.

nntile(nntile) / HF(cuda): XS **1.42×**, S **1.11×**, M **1.07×**, L **1.03×**, XL **1.03×**. Isolated GPU time: L **1.770** vs HF(cuda) **1.725 ± 0.001** s; XL **2.242** vs **2.171 ± 0.011** s.
**XL train wall: HF(nntile) 0.96×, nntile(nntile) 1.03×.**

### Peak VRAM and bus

Peak VRAM is `nvidia-smi memory.used`. H2D/D2H are StarPU bus stats at
shutdown. HF(cuda) has no StarPU bus.

| Setup | HF(cuda) VRAM | HF(nntile) VRAM | HF(nntile) H2D | HF(nntile) D2H | nntile(nntile) VRAM | nntile(nntile) H2D | nntile(nntile) D2H |
|-------|----------:|--------------:|-------------:|-------------:|---------------------:|--------------------:|--------------------:|
| XS T=784 | 4.8 GiB | 4.9 GiB | 1.94 GB | **0** | 6.8 GiB | 1.95 GB | **0** |
| S T=1024 | 7.6 GiB | 6.5 GiB | 3.11 GB | **0** | 9.4 GiB | 3.12 GB | **0** |
| M T=1521 | 17.9 GiB | 16.4 GiB | 7.60 GB | **0** | 23.5 GiB | 7.61 GB | **0** |
| L T=2025 | 31.4 GiB | 30.4 GiB | 13.43 GB | **0** | **42.7 GiB** | 13.45 GB | **0** |
| XL T=2916 | 29.2 GiB | 35.7 GiB | 12.19 GB | **0** | **43.6 GiB** | 12.20 GB | **0** |

No D2H on any nntile setup. H2D is the initial prefetch. nntile(nntile) L
(11 layers) and XL (5 layers) both fit under `STARPU_LIMIT_CUDA_MEM=46000`.

## HF(nntile) vs HF(cuda) (10-step train wall)

This section is **HF(nntile) only** (stock Diffusers on `device=nntile`).
XL is **0.96×** HF(cuda) here. nntile(nntile) is not in this table. **nntile(nntile) XL is 1.03×** — see [nntile(nntile) vs HF(cuda)](#nntilenntile-vs-hfcuda).
VRAM for HF(cuda) / HF(nntile) / nntile(nntile) is in
[Peak VRAM and bus](#peak-vram-and-bus) (`nvidia-smi`).

| Setup | HF(cuda) wall | HF(nntile) wall | HF(nntile) / HF(cuda) | record(nntile) | record(torch) | compile | run | wait | host/wall | HF(cuda) loss | HF(nntile) loss |
|-------|----------:|------------:|------------:|---------------:|--------------:|--------:|----:|-----:|----------:|----------:|------------:|
| XS T=784 | 1.393 ± 0.006 s | 1.898 ± 0.044 s | **1.36×** | 0.131 ± 0.010 s | 0.392 ± 0.024 s | 0.177 ± 0.030 s | 0.193 ± 0.026 s | 1.004 ± 0.035 s | **36.8%** | 1.209802 | **1.209802** |
| S T=1024 | 2.577 ± 0.008 s | 2.834 ± 0.011 s | **1.10×** | 0.117 ± 0.011 s | 0.344 ± 0.017 s | 0.138 ± 0.005 s | 0.164 ± 0.012 s | 2.070 ± 0.035 s | **21.1%** | 1.192550 | **1.192550** |
| M T=1521 | 7.720 ± 0.022 s | 7.756 ± 0.038 s | **1.00×** | 0.122 ± 0.007 s | 0.370 ± 0.011 s | 0.145 ± 0.007 s | 0.175 ± 0.010 s | 6.942 ± 0.037 s | **8.2%** | 1.141145 | **1.141145** |
| L T=2025 | 17.346 ± 0.028 s | 17.007 ± 0.034 s | **0.98×** | 0.122 ± 0.005 s | 0.372 ± 0.006 s | 0.144 ± 0.004 s | 0.179 ± 0.008 s | 16.189 ± 0.033 s | **3.7%** | 1.221610 | **1.221610** |
| XL T=2916 | 21.796 ± 0.074 s | 21.015 ± 0.061 s | **0.96×** | 0.087 ± 0.003 s | 0.244 ± 0.005 s | 0.100 ± 0.003 s | 0.113 ± 0.003 s | 20.469 ± 0.059 s | **2.1%** | 1.034324 | **1.034324** |

Host = `record(nntile)+record(torch)+compile` (~0.29–0.51 s for 10 steps,
**flat**). Host **share** drops **36.8% → 21.1% → 8.2% → 3.7% → 2.1%**
as GPU work grows.

MSE noise-prediction loss matches HF(cuda) vs HF(nntile) to printed 1e-4 at all ladder sizes (XS 1.209802 both).

HF(nntile) isolated GPU `run+wait` vs HF(cuda) isolated wall:
XS 0.157 ± 0.003 vs 0.126 ± 0.001 s, S 0.256 ± 0.002 vs 0.244 ± 0.001 s, M 0.747 ± 0.003 vs 0.756 ± 0.002 s, L 1.676 ± 0.002 vs 1.725 ± 0.001 s, XL 2.088 ± 0.007 vs 2.171 ± 0.011 s.

## nntile(nntile) vs HF(cuda)

nntile(nntile) only, overlap, 10 steps, **10 repeats** (mean ± stdev).
`STARPU_LIMIT_CUDA_MEM=46000`. Host patchify is outside the train wall.
HF(cuda) walls are this `--disable-tf32 --disable-cudnn` 10-repeat campaign. nntile(nntile) walls are the published classic-kernel means (FP32; not rerun). Peak VRAM / H2D / D2H
below are **nntile(nntile)**. HF(cuda) VRAM and HF(nntile) bus stats are in
[Peak VRAM and bus](#peak-vram-and-bus).

| Setup | HF(cuda) wall | nntile(nntile) wall | nntile(nntile) / HF(cuda) | isolated | peak VRAM | H2D | D2H | host/wall | nntile(nntile) loss |
|-------|----------:|-------------:|-------------:|---------:|----------:|----:|----:|----------:|-------------:|
| XS T=784 | 1.393 ± 0.006 s | 1.976 ± 0.031 s | **1.42×** | 0.176 ± 0.001 s | 6.8 GiB | 1.95 GB | **0** | **32.7%** | 1.209802 |
| S T=1024 | 2.577 ± 0.008 s | 2.871 ± 0.008 s | **1.11×** | 0.268 ± 0.001 s | 9.4 GiB | 3.12 GB | **0** | **19.5%** | 1.192550 |
| M T=1521 | 7.720 ± 0.022 s | 8.225 ± 0.013 s | **1.07×** | 0.805 ± 0.002 s | 23.5 GiB | 7.61 GB | **0** | **7.3%** | 1.141145 |
| L T=2025 | 17.346 ± 0.028 s | 17.880 ± 0.028 s | **1.03×** | 1.770 ± 0.004 s | **42.7 GiB** | 13.45 GB | **0** | **3.3%** | 1.221610 |
| XL T=2916 | 21.796 ± 0.074 s | 22.421 ± 0.114 s | **1.03×** | 2.242 ± 0.010 s | **43.6 GiB** | 12.20 GB | **0** | **1.8%** | 1.034324 |

Host = `record(nntile)+record(torch)+compile`. Host **share** drops
**32.7% → 19.5% → 7.3% → 3.3% → 1.8%**.

| Setup | record(nntile) | record(torch) | compile | run | wait |
|-------|---------------:|--------------:|--------:|----:|-----:|
| XS T=784 | 0.061 ± 0.003 s | 0.352 ± 0.020 s | 0.234 ± 0.011 s | 0.202 ± 0.016 s | 1.126 ± 0.033 s |
| S T=1024 | 0.051 ± 0.003 s | 0.307 ± 0.010 s | 0.200 ± 0.010 s | 0.174 ± 0.011 s | 2.137 ± 0.026 s |
| M T=1521 | 0.055 ± 0.002 s | 0.334 ± 0.011 s | 0.209 ± 0.005 s | 0.186 ± 0.009 s | 7.440 ± 0.025 s |
| L T=2025 | 0.054 ± 0.002 s | 0.337 ± 0.007 s | 0.204 ± 0.007 s | 0.189 ± 0.007 s | 17.094 ± 0.034 s |
| XL T=2916 | 0.039 ± 0.003 s | 0.222 ± 0.008 s | 0.141 ± 0.006 s | 0.136 ± 0.005 s | 21.881 ± 0.116 s |

nntile(nntile) MSE matches HF(cuda) / HF(nntile) to printed 1e-6.

No StarPU reclaim. D2H is **0** on every size. XL bus at shutdown
(prefetch + 10 steps + isolated):

| Direction | Volume | Transfers | avg size |
|--|--:|--:|--:|
| NUMA 0 → CUDA 0 | **12.20 GB** | 221 | 57 MB |
| CUDA 0 → NUMA 0 | **0** | 1 | 0 |
| **Total** | **12.20 GB** | 222 | |

H2D is the initial prefetch. Isolated `run+wait` is slightly above
HF(cuda) because AdaLN-Zero is six `H→H` GEMMs (no fused `Linear(H, 6H)`
on classic kernels).

## 100-step S (HF(nntile) steady state, mean ± stdev over 10 runs)

Same **S** config (hidden 2048, `sample_size=64`, **1024 patches**), B=1,
**100 optimizer steps**, **HF(nntile)** overlap only (not nntile(nntile)).
Complements the 10-step HF ladder above.

Loss **1.226684** (MSE noise; matches 10-step S).

| | Total | mean / step |
|--|--:|--:|
| record(nntile) | 1.407 ± 0.056 s | 14.1 ms |
| record(torch) | 5.131 ± 0.449 s | 51 ms |
| compile | 1.727 ± 0.067 s | 17 ms |
| run | 1.947 ± 0.123 s | 19 ms |
| wait | 16.290 ± 0.521 s | 163 ms |
| **train wall** | **26.513 ± 0.184 s** | 265 ms |

Host (record + compile) is **31%** of the wall (~83 ms/step).

![Host overhead per iteration](dit_hf_overhead_s_100.svg)

CSV: [`dit_hf_overhead_s_100.csv`](dit_hf_overhead_s_100.csv) (median of 10 runs).

## Comparison to GPT-2 (wall time only)

GPT-2 uses the same **hidden / token-count ladder** but a different task
(causal LM cross-entropy vs DiT MSE noise prediction). Compare **HF(nntile) / HF(cuda)
wall ratios** only; loss values are not comparable.
nntile(nntile) is not in this table (DiT XL nntile(nntile) is **1.03×**).

See [`gpt2_hf_overhead_scale.md`](gpt2_hf_overhead_scale.md).

| Size | GPT-2 HF(nntile)/HF(cuda) | DiT HF(nntile)/HF(cuda) |
|------|------------------:|-----------------:|
| XS | 0.99× | **1.36×** |
| S | 0.96× | **1.10×** |
| M | 0.94× | **1.00×** |
| L | 0.94× | **0.98×** |
| XL | 0.96× | **0.96×** |

### 100-step S (HF(nntile))

| | GPT-2 | DiT | Notes |
|--|------:|-----:|-------|
| train wall | 27.5 s | **26.5 s** | same ballpark |
| final loss | 7.734033 (LM CE) | **1.226684** (MSE noise) | different task |
| host share | 22% | **31%** | flat host, GPU-bound |

## Per iteration (HF(nntile), mean ± stdev over 10 runs)

### XS (hidden 1536, T=784 patches)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.256 ± 0.004 | 0.008 ± 0.001 | 0.027 ± 0.002 | 0.010 ± 0.001 | 0.011 ± 0.002 | 0.000 |
| 2 | 0.127 ± 0.001 | 0.007 ± 0.001 | 0.022 ± 0.001 | 0.011 ± 0.002 | 0.018 ± 0.005 | 0.325 ± 0.018 |
| 3 | 0.126 ± 0.001 | 0.014 ± 0.003 | 0.032 ± 0.003 | 0.018 ± 0.003 | 0.020 ± 0.004 | 0.080 ± 0.009 |
| 4 | 0.126 ± 0.001 | 0.015 ± 0.002 | 0.035 ± 0.002 | 0.018 ± 0.004 | 0.020 ± 0.004 | 0.074 ± 0.006 |
| 5 | 0.126 ± 0.001 | 0.014 ± 0.002 | 0.037 ± 0.002 | 0.020 ± 0.003 | 0.021 ± 0.003 | 0.073 ± 0.005 |
| 6 | 0.126 ± 0.001 | 0.014 ± 0.001 | 0.041 ± 0.004 | 0.018 ± 0.003 | 0.020 ± 0.004 | 0.071 ± 0.007 |
| 7 | 0.127 ± 0.001 | 0.014 ± 0.002 | 0.046 ± 0.006 | 0.020 ± 0.005 | 0.021 ± 0.003 | 0.064 ± 0.009 |
| 8 | 0.126 ± 0.001 | 0.015 ± 0.001 | 0.049 ± 0.007 | 0.020 ± 0.005 | 0.020 ± 0.003 | 0.062 ± 0.007 |
| 9 | 0.126 ± 0.001 | 0.014 ± 0.001 | 0.050 ± 0.008 | 0.020 ± 0.005 | 0.022 ± 0.004 | 0.061 ± 0.011 |
| 10 | 0.127 ± 0.001 | 0.015 ± 0.001 | 0.054 ± 0.005 | 0.022 ± 0.004 | 0.020 ± 0.002 | 0.195 ± 0.006 |

### S (hidden 2048, T=1024 patches)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.374 ± 0.005 | 0.007 ± 0.001 | 0.026 ± 0.002 | 0.010 ± 0.001 | 0.011 ± 0.001 | 0.000 |
| 2 | 0.244 ± 0.001 | 0.007 ± 0.001 | 0.021 ± 0.001 | 0.010 ± 0.001 | 0.015 ± 0.003 | 0.413 ± 0.006 |
| 3 | 0.245 ± 0.001 | 0.012 ± 0.002 | 0.028 ± 0.003 | 0.013 ± 0.001 | 0.017 ± 0.003 | 0.188 ± 0.008 |
| 4 | 0.245 ± 0.001 | 0.012 ± 0.002 | 0.031 ± 0.002 | 0.014 ± 0.001 | 0.017 ± 0.003 | 0.184 ± 0.007 |
| 5 | 0.245 ± 0.001 | 0.013 ± 0.002 | 0.035 ± 0.003 | 0.015 ± 0.001 | 0.019 ± 0.003 | 0.178 ± 0.007 |
| 6 | 0.245 ± 0.001 | 0.014 ± 0.002 | 0.036 ± 0.004 | 0.016 ± 0.001 | 0.016 ± 0.001 | 0.175 ± 0.009 |
| 7 | 0.245 ± 0.001 | 0.013 ± 0.001 | 0.038 ± 0.003 | 0.015 ± 0.001 | 0.017 ± 0.003 | 0.177 ± 0.005 |
| 8 | 0.244 ± 0.001 | 0.013 ± 0.002 | 0.041 ± 0.004 | 0.016 ± 0.002 | 0.017 ± 0.002 | 0.173 ± 0.007 |
| 9 | 0.244 ± 0.001 | 0.013 ± 0.002 | 0.043 ± 0.004 | 0.015 ± 0.001 | 0.019 ± 0.004 | 0.171 ± 0.006 |
| 10 | 0.244 ± 0.001 | 0.013 ± 0.003 | 0.046 ± 0.003 | 0.015 ± 0.001 | 0.017 ± 0.002 | 0.410 ± 0.009 |

### M (hidden 3072, T=1521 patches)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.914 ± 0.004 | 0.007 ± 0.000 | 0.026 ± 0.002 | 0.010 ± 0.001 | 0.012 ± 0.001 | 0.000 |
| 2 | 0.757 ± 0.003 | 0.007 ± 0.000 | 0.021 ± 0.001 | 0.011 ± 0.001 | 0.016 ± 0.005 | 0.894 ± 0.006 |
| 3 | 0.757 ± 0.003 | 0.011 ± 0.003 | 0.029 ± 0.004 | 0.013 ± 0.001 | 0.018 ± 0.004 | 0.681 ± 0.011 |
| 4 | 0.757 ± 0.004 | 0.013 ± 0.002 | 0.033 ± 0.003 | 0.014 ± 0.001 | 0.018 ± 0.003 | 0.671 ± 0.010 |
| 5 | 0.757 ± 0.004 | 0.014 ± 0.002 | 0.037 ± 0.002 | 0.016 ± 0.001 | 0.019 ± 0.003 | 0.666 ± 0.008 |
| 6 | 0.757 ± 0.003 | 0.014 ± 0.002 | 0.040 ± 0.002 | 0.015 ± 0.000 | 0.018 ± 0.003 | 0.664 ± 0.006 |
| 7 | 0.755 ± 0.002 | 0.014 ± 0.001 | 0.042 ± 0.001 | 0.015 ± 0.001 | 0.018 ± 0.002 | 0.663 ± 0.007 |
| 8 | 0.755 ± 0.002 | 0.014 ± 0.002 | 0.045 ± 0.002 | 0.015 ± 0.001 | 0.019 ± 0.003 | 0.660 ± 0.007 |
| 9 | 0.756 ± 0.002 | 0.014 ± 0.001 | 0.048 ± 0.002 | 0.017 ± 0.004 | 0.019 ± 0.002 | 0.655 ± 0.005 |
| 10 | 0.755 ± 0.002 | 0.014 ± 0.001 | 0.049 ± 0.002 | 0.018 ± 0.001 | 0.018 ± 0.001 | 1.388 ± 0.007 |

### L (hidden 4096, T=2025 patches)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 1.853 ± 0.003 | 0.007 ± 0.000 | 0.027 ± 0.001 | 0.010 ± 0.001 | 0.012 ± 0.000 | 0.000 |
| 2 | 1.724 ± 0.006 | 0.007 ± 0.000 | 0.022 ± 0.001 | 0.011 ± 0.001 | 0.016 ± 0.003 | 1.815 ± 0.003 |
| 3 | 1.723 ± 0.007 | 0.012 ± 0.003 | 0.030 ± 0.004 | 0.013 ± 0.001 | 0.016 ± 0.002 | 1.603 ± 0.015 |
| 4 | 1.721 ± 0.009 | 0.012 ± 0.002 | 0.033 ± 0.003 | 0.014 ± 0.000 | 0.018 ± 0.002 | 1.600 ± 0.011 |
| 5 | 1.721 ± 0.008 | 0.014 ± 0.002 | 0.037 ± 0.002 | 0.017 ± 0.004 | 0.019 ± 0.003 | 1.589 ± 0.010 |
| 6 | 1.719 ± 0.003 | 0.014 ± 0.002 | 0.040 ± 0.002 | 0.015 ± 0.000 | 0.021 ± 0.004 | 1.589 ± 0.009 |
| 7 | 1.719 ± 0.001 | 0.014 ± 0.002 | 0.044 ± 0.004 | 0.016 ± 0.001 | 0.018 ± 0.003 | 1.580 ± 0.009 |
| 8 | 1.720 ± 0.001 | 0.013 ± 0.002 | 0.046 ± 0.003 | 0.015 ± 0.001 | 0.019 ± 0.003 | 1.585 ± 0.006 |
| 9 | 1.722 ± 0.001 | 0.013 ± 0.002 | 0.046 ± 0.004 | 0.015 ± 0.001 | 0.020 ± 0.003 | 1.585 ± 0.009 |
| 10 | 1.724 ± 0.001 | 0.014 ± 0.002 | 0.049 ± 0.003 | 0.018 ± 0.000 | 0.020 ± 0.003 | 3.243 ± 0.010 |

### XL (hidden 5760, T=2916 patches, 5 layers)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 2.342 ± 0.011 | 0.004 | 0.015 ± 0.000 | 0.005 | 0.006 ± 0.000 | 0.000 |
| 2 | 2.150 ± 0.008 | 0.003 ± 0.000 | 0.011 ± 0.000 | 0.005 | 0.007 ± 0.001 | 2.228 ± 0.012 |
| 3 | 2.156 ± 0.007 | 0.005 ± 0.001 | 0.014 ± 0.001 | 0.007 ± 0.001 | 0.009 ± 0.001 | 2.034 ± 0.007 |
| 4 | 2.157 ± 0.007 | 0.008 ± 0.001 | 0.020 ± 0.002 | 0.011 ± 0.000 | 0.012 ± 0.001 | 2.027 ± 0.008 |
| 5 | 2.162 ± 0.007 | 0.011 ± 0.001 | 0.025 ± 0.001 | 0.013 ± 0.001 | 0.013 ± 0.001 | 2.017 ± 0.010 |
| 6 | 2.163 ± 0.005 | 0.011 ± 0.001 | 0.028 ± 0.002 | 0.012 ± 0.001 | 0.013 ± 0.001 | 2.016 ± 0.007 |
| 7 | 2.166 ± 0.005 | 0.011 ± 0.001 | 0.030 ± 0.001 | 0.012 ± 0.001 | 0.013 ± 0.001 | 2.018 ± 0.008 |
| 8 | 2.165 ± 0.008 | 0.012 ± 0.001 | 0.033 ± 0.002 | 0.012 ± 0.000 | 0.013 ± 0.001 | 2.018 ± 0.005 |
| 9 | 2.166 ± 0.011 | 0.011 ± 0.000 | 0.034 ± 0.000 | 0.012 ± 0.001 | 0.013 ± 0.001 | 2.017 ± 0.007 |
| 10 | 2.168 ± 0.009 | 0.011 ± 0.001 | 0.034 ± 0.001 | 0.012 ± 0.001 | 0.013 ± 0.001 | 4.095 ± 0.014 |

## Isolated extra step (HF(nntile), mean ± stdev over 10 runs)

| Setup | record(nntile) | record(torch) | compile | run | wait | run+wait | HF(cuda) isolated |
|-------|---------------:|--------------:|--------:|----:|-----:|---------:|--------------:|
| XS | 0.016 ± 0.001 | 0.057 ± 0.006 | 0.019 ± 0.002 | 0.020 ± 0.003 | 0.137 ± 0.002 | **0.157 ± 0.003** | 0.126 ± 0.001 |
| S | 0.017 ± 0.004 | 0.052 ± 0.007 | 0.018 ± 0.003 | 0.017 ± 0.003 | 0.239 ± 0.002 | **0.256 ± 0.002** | 0.244 ± 0.001 |
| M | 0.019 ± 0.001 | 0.054 ± 0.007 | 0.019 ± 0.001 | 0.018 ± 0.001 | 0.729 ± 0.003 | **0.747 ± 0.003** | 0.756 ± 0.002 |
| L | 0.018 ± 0.003 | 0.057 ± 0.005 | 0.018 ± 0.003 | 0.018 ± 0.003 | 1.659 ± 0.003 | **1.676 ± 0.002** | 1.725 ± 0.001 |
| XL | 0.013 ± 0.001 | 0.036 ± 0.001 | 0.016 ± 0.001 | 0.013 ± 0.001 | 2.076 ± 0.007 | **2.088 ± 0.007** | 2.171 ± 0.011 |

| Setup | Full isolated (record+compile+run+wait) | Hidden host (`run+wait`) | Saved |
|-------|----------------------------------------:|-------------------------:|------:|
| XS | 0.249 s | 0.157 s | 0.092 s (**37%**) |
| S | 0.344 s | 0.256 s | 0.088 s (**26%**) |
| M | 0.840 s | 0.747 s | 0.092 s (**11%**) |
| L | 1.769 s | 1.676 s | 0.093 s (**5%**) |
| XL | 2.153 s | 2.088 s | 0.065 s (**3%**) |

## Sequential prep vs compute (`--wait-after-run`, HF(nntile))

| Setup | HF(cuda) wall | sequential wall | prep | compute | compute / HF(cuda) | prep/wall |
|-------|----------:|----------------:|-----:|--------:|-------------:|----------:|
| XS T=784 | 1.393 ± 0.006 s | 2.453 ± 0.028 s | 0.700 ± 0.017 s | **1.752 ± 0.014 s** | **1.26×** | 28.5% |
| S T=1024 | 2.577 ± 0.008 s | 3.405 ± 0.053 s | 0.659 ± 0.046 s | **2.744 ± 0.018 s** | **1.06×** | 19.4% |
| M T=1521 | 7.720 ± 0.022 s | 8.357 ± 0.046 s | 0.700 ± 0.053 s | **7.655 ± 0.036 s** | **0.99×** | 8.4% |
| L T=2025 | 17.346 ± 0.028 s | 17.617 ± 0.040 s | 0.705 ± 0.022 s | **16.910 ± 0.041 s** | **0.97×** | 4.0% |
| XL T=2916 | 21.796 ± 0.074 s | 21.415 ± 0.076 s | 0.472 ± 0.007 s | **20.942 ± 0.068 s** | **0.96×** | 2.2% |

Sequential HF(nntile) loss: XS 1.209802, S 1.192550, M 1.141145, L 1.221610, XL 1.034324.

### Per iteration (prep / compute, mean ± stdev)

#### XS (T=784)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.045 ± 0.003 | 0.365 ± 0.007 | 0.007 ± 0.001 | 0.027 ± 0.001 | 0.010 ± 0.002 | 0.010 ± 0.001 | 0.355 ± 0.006 |
| 2 | 0.042 ± 0.002 | 0.152 ± 0.002 | 0.008 ± 0.001 | 0.023 ± 0.001 | 0.011 ± 0.001 | 0.013 ± 0.001 | 0.140 ± 0.001 |
| 3 | 0.056 ± 0.003 | 0.153 ± 0.002 | 0.013 ± 0.001 | 0.030 ± 0.001 | 0.014 ± 0.001 | 0.015 ± 0.001 | 0.138 ± 0.001 |
| 4 | 0.064 ± 0.002 | 0.154 ± 0.001 | 0.014 ± 0.001 | 0.034 ± 0.001 | 0.016 ± 0.001 | 0.017 ± 0.002 | 0.137 ± 0.002 |
| 5 | 0.073 ± 0.005 | 0.155 ± 0.003 | 0.016 ± 0.002 | 0.040 ± 0.003 | 0.018 ± 0.001 | 0.018 ± 0.001 | 0.138 ± 0.003 |
| 6 | 0.079 ± 0.003 | 0.154 ± 0.001 | 0.017 ± 0.001 | 0.044 ± 0.003 | 0.018 ± 0.001 | 0.018 ± 0.001 | 0.136 ± 0.001 |
| 7 | 0.081 ± 0.003 | 0.154 ± 0.002 | 0.018 ± 0.002 | 0.045 ± 0.001 | 0.018 ± 0.001 | 0.018 ± 0.001 | 0.136 ± 0.001 |
| 8 | 0.085 ± 0.003 | 0.155 ± 0.002 | 0.018 ± 0.001 | 0.048 ± 0.001 | 0.019 ± 0.001 | 0.018 ± 0.001 | 0.137 ± 0.002 |
| 9 | 0.084 ± 0.006 | 0.154 ± 0.002 | 0.017 ± 0.002 | 0.049 ± 0.002 | 0.018 ± 0.002 | 0.019 ± 0.002 | 0.135 ± 0.002 |
| 10 | 0.091 ± 0.005 | 0.155 ± 0.002 | 0.018 ± 0.002 | 0.053 ± 0.003 | 0.020 ± 0.001 | 0.019 ± 0.002 | 0.136 ± 0.002 |

#### S (T=1024)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.043 ± 0.005 | 0.459 ± 0.009 | 0.007 ± 0.001 | 0.025 ± 0.002 | 0.010 ± 0.003 | 0.011 ± 0.002 | 0.448 ± 0.008 |
| 2 | 0.041 ± 0.003 | 0.253 ± 0.003 | 0.008 ± 0.001 | 0.022 ± 0.002 | 0.011 ± 0.001 | 0.012 ± 0.002 | 0.241 ± 0.003 |
| 3 | 0.053 ± 0.004 | 0.253 ± 0.002 | 0.012 ± 0.001 | 0.027 ± 0.002 | 0.014 ± 0.003 | 0.015 ± 0.002 | 0.239 ± 0.001 |
| 4 | 0.062 ± 0.004 | 0.254 ± 0.003 | 0.014 ± 0.002 | 0.033 ± 0.001 | 0.015 ± 0.002 | 0.015 ± 0.003 | 0.239 ± 0.003 |
| 5 | 0.068 ± 0.011 | 0.254 ± 0.002 | 0.016 ± 0.005 | 0.037 ± 0.003 | 0.015 ± 0.004 | 0.015 ± 0.003 | 0.238 ± 0.003 |
| 6 | 0.072 ± 0.008 | 0.254 ± 0.001 | 0.016 ± 0.003 | 0.039 ± 0.003 | 0.017 ± 0.004 | 0.017 ± 0.003 | 0.237 ± 0.002 |
| 7 | 0.074 ± 0.009 | 0.254 ± 0.002 | 0.016 ± 0.003 | 0.041 ± 0.005 | 0.017 ± 0.002 | 0.017 ± 0.002 | 0.237 ± 0.002 |
| 8 | 0.079 ± 0.009 | 0.254 ± 0.002 | 0.017 ± 0.004 | 0.043 ± 0.006 | 0.018 ± 0.002 | 0.017 ± 0.003 | 0.237 ± 0.002 |
| 9 | 0.084 ± 0.008 | 0.254 ± 0.002 | 0.018 ± 0.001 | 0.047 ± 0.008 | 0.018 ± 0.002 | 0.018 ± 0.001 | 0.237 ± 0.002 |
| 10 | 0.085 ± 0.010 | 0.255 ± 0.001 | 0.018 ± 0.002 | 0.049 ± 0.008 | 0.018 ± 0.001 | 0.018 ± 0.001 | 0.237 ± 0.002 |

#### M (T=1521)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.045 ± 0.002 | 0.943 ± 0.005 | 0.007 ± 0.000 | 0.027 ± 0.001 | 0.011 ± 0.001 | 0.012 ± 0.001 | 0.931 ± 0.005 |
| 2 | 0.047 ± 0.002 | 0.744 ± 0.002 | 0.010 ± 0.001 | 0.024 ± 0.001 | 0.012 ± 0.001 | 0.013 ± 0.001 | 0.731 ± 0.002 |
| 3 | 0.057 ± 0.004 | 0.746 ± 0.003 | 0.013 ± 0.001 | 0.030 ± 0.002 | 0.014 ± 0.001 | 0.014 ± 0.001 | 0.732 ± 0.003 |
| 4 | 0.066 ± 0.005 | 0.746 ± 0.003 | 0.015 ± 0.002 | 0.035 ± 0.002 | 0.016 ± 0.002 | 0.016 ± 0.002 | 0.730 ± 0.005 |
| 5 | 0.071 ± 0.006 | 0.746 ± 0.003 | 0.015 ± 0.002 | 0.039 ± 0.002 | 0.017 ± 0.002 | 0.017 ± 0.002 | 0.729 ± 0.004 |
| 6 | 0.076 ± 0.007 | 0.745 ± 0.004 | 0.017 ± 0.003 | 0.043 ± 0.003 | 0.017 ± 0.002 | 0.017 ± 0.002 | 0.728 ± 0.004 |
| 7 | 0.079 ± 0.007 | 0.746 ± 0.003 | 0.016 ± 0.003 | 0.045 ± 0.003 | 0.017 ± 0.002 | 0.017 ± 0.002 | 0.729 ± 0.004 |
| 8 | 0.083 ± 0.009 | 0.745 ± 0.004 | 0.017 ± 0.003 | 0.049 ± 0.004 | 0.017 ± 0.003 | 0.018 ± 0.002 | 0.728 ± 0.004 |
| 9 | 0.087 ± 0.006 | 0.746 ± 0.004 | 0.017 ± 0.002 | 0.051 ± 0.003 | 0.018 ± 0.002 | 0.019 ± 0.002 | 0.728 ± 0.005 |
| 10 | 0.090 ± 0.010 | 0.747 ± 0.004 | 0.017 ± 0.002 | 0.051 ± 0.007 | 0.021 ± 0.002 | 0.018 ± 0.001 | 0.728 ± 0.005 |

#### L (T=2025)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.044 ± 0.001 | 1.865 ± 0.004 | 0.007 | 0.026 ± 0.001 | 0.010 ± 0.000 | 0.012 ± 0.001 | 1.853 ± 0.004 |
| 2 | 0.046 ± 0.001 | 1.673 ± 0.007 | 0.010 ± 0.001 | 0.024 ± 0.001 | 0.012 ± 0.001 | 0.013 ± 0.001 | 1.660 ± 0.007 |
| 3 | 0.057 ± 0.002 | 1.672 ± 0.009 | 0.013 ± 0.001 | 0.030 ± 0.001 | 0.014 ± 0.001 | 0.014 ± 0.001 | 1.658 ± 0.009 |
| 4 | 0.066 ± 0.003 | 1.672 ± 0.011 | 0.015 ± 0.001 | 0.036 ± 0.001 | 0.016 ± 0.001 | 0.015 ± 0.001 | 1.657 ± 0.011 |
| 5 | 0.074 ± 0.004 | 1.672 ± 0.008 | 0.016 ± 0.002 | 0.040 ± 0.002 | 0.018 ± 0.002 | 0.017 ± 0.001 | 1.655 ± 0.008 |
| 6 | 0.079 ± 0.006 | 1.668 ± 0.003 | 0.017 ± 0.001 | 0.042 ± 0.003 | 0.020 ± 0.003 | 0.019 ± 0.002 | 1.649 ± 0.005 |
| 7 | 0.078 ± 0.007 | 1.670 ± 0.003 | 0.015 ± 0.003 | 0.045 ± 0.004 | 0.018 ± 0.001 | 0.017 ± 0.001 | 1.652 ± 0.003 |
| 8 | 0.085 ± 0.004 | 1.670 ± 0.002 | 0.017 ± 0.002 | 0.049 ± 0.003 | 0.019 ± 0.001 | 0.018 ± 0.002 | 1.652 ± 0.002 |
| 9 | 0.088 ± 0.008 | 1.672 ± 0.003 | 0.018 ± 0.003 | 0.051 ± 0.004 | 0.019 ± 0.002 | 0.018 ± 0.002 | 1.654 ± 0.005 |
| 10 | 0.089 ± 0.009 | 1.673 ± 0.003 | 0.017 ± 0.003 | 0.051 ± 0.005 | 0.021 ± 0.003 | 0.018 ± 0.002 | 1.656 ± 0.003 |

#### XL (T=2916)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.024 ± 0.001 | 2.247 ± 0.006 | 0.004 | 0.015 ± 0.001 | 0.005 | 0.006 | 2.242 ± 0.006 |
| 2 | 0.024 ± 0.001 | 2.065 ± 0.009 | 0.005 ± 0.000 | 0.013 ± 0.000 | 0.006 ± 0.000 | 0.006 ± 0.001 | 2.059 ± 0.008 |
| 3 | 0.032 ± 0.001 | 2.069 ± 0.009 | 0.007 ± 0.000 | 0.017 ± 0.001 | 0.008 ± 0.001 | 0.008 ± 0.001 | 2.061 ± 0.009 |
| 4 | 0.040 ± 0.004 | 2.074 ± 0.009 | 0.009 ± 0.001 | 0.021 ± 0.001 | 0.010 ± 0.001 | 0.009 ± 0.001 | 2.065 ± 0.008 |
| 5 | 0.054 ± 0.002 | 2.077 ± 0.010 | 0.012 ± 0.001 | 0.028 ± 0.001 | 0.014 ± 0.001 | 0.013 ± 0.001 | 2.064 ± 0.010 |
| 6 | 0.054 ± 0.001 | 2.078 ± 0.008 | 0.013 ± 0.001 | 0.029 ± 0.001 | 0.012 ± 0.001 | 0.011 ± 0.001 | 2.067 ± 0.008 |
| 7 | 0.060 ± 0.002 | 2.082 ± 0.005 | 0.013 ± 0.001 | 0.035 ± 0.001 | 0.012 ± 0.001 | 0.011 ± 0.001 | 2.070 ± 0.005 |
| 8 | 0.061 ± 0.001 | 2.082 ± 0.005 | 0.013 ± 0.001 | 0.034 ± 0.001 | 0.014 ± 0.000 | 0.012 ± 0.001 | 2.070 ± 0.005 |
| 9 | 0.060 ± 0.002 | 2.085 ± 0.007 | 0.013 ± 0.001 | 0.035 ± 0.001 | 0.012 | 0.011 | 2.074 ± 0.007 |
| 10 | 0.062 ± 0.001 | 2.083 ± 0.007 | 0.013 ± 0.001 | 0.035 ± 0.001 | 0.013 ± 0.001 | 0.013 ± 0.000 | 2.071 ± 0.007 |

Steady compute after iter 1 (mean over repeats): ~0.152 s (XS), ~0.253 s (S), ~0.744 s (M), ~1.673 s (L), ~2.065 s (XL).

## Takeaways

1. **Diffusers DiT**, synthetic diffusion batches, MSE noise loss; ladder
   geometry aligned to Llama via hidden size + patch count + HF(cuda) VRAM match.
2. **HF(nntile) graph host overhead is flat** (~0.3–0.5 s / 10 steps); share
   falls as GPU work grows (36.8% → 2.1%).
3. **HF(nntile)** is within ~5–40% of HF(cuda) on wall time
   (XS 1.36×, S 1.10×, M 1.00×, L 0.98×, XL 0.96×). XS is host-bound;
   **HF(nntile)** M/L/XL are near parity. That XL **0.96× is not
   nntile(nntile)** — nntile(nntile) XL is **1.03×** (takeaway 9).
4. **L=11 / XL=5** are the published json so nntile(nntile) has D2H **0**.
   Isolated GPU time is still a bit above HF(cuda) because AdaLN-Zero is
   six `H→H` GEMMs, not a fused `H→6H`.
5. **HF(nntile) sequential GPU time** (`run+wait`): **1.26× → 1.06× → 0.99× → 0.97× → 0.96×** vs HF(cuda).
6. Timings are **mean ± stdev** over 10 runs.
7. **MSE loss** matches all three setups to printed 1e-6 — see
   [Three setups](#three-setups).
8. **100-step S HF(nntile)** wall **26.513 ± 0.184 s** — see section above.
9. nntile(nntile): **1.03–1.11×** HF(cuda) on S–XL (D2H **0**); XS **1.42×**
   (host-bound; HF(nntile) XS is **1.36×**). Peak VRAM L **42.7 GiB**,
   XL **43.6 GiB**. Peak VRAM / H2D / D2H are in
   [Peak VRAM and bus](#peak-vram-and-bus).

## How to reproduce

```bash
# diffusers in a venv with system-site-packages (inherits conda torch):
python3 -m venv --system-site-packages .venv
.venv/bin/pip install 'diffusers==0.32.2'

export TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export NNTILE_BUILD_DIR=$PWD/build TORCH_NNTILE_BUILD_DIR=$PWD/build
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:$PWD/build/nntile:$PWD/build/torch_nntile:/opt/starpu/lib"
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1

# Full ladder, 10 repeats, one GPU (HF(cuda) / HF(nntile)).
# Runner passes --disable-tf32 --disable-cudnn on both backends.
.venv/bin/python torch_nntile/tools/run_dit_overhead_benchmark.py \
  --logdir /tmp/dit_overhead --gpu 0 --repeats 10

# nntile(nntile) (host patchify, then DiT on device=nntile):
.venv/bin/python torch_nntile/tools/run_nntile_native_overhead_benchmark.py \
  --family dit --logdir /tmp/dit_native --gpu 0 --repeats 10

# Regenerate HF(cuda)/HF(nntile) sections from parsed HF logs.
# Three-setup / nntile(nntile) / Peak VRAM tables in the markdown are
# hand-maintained (do not drop them).
.venv/bin/python torch_nntile/tools/update_dit_overhead_doc.py \
  --summary /tmp/dit_overhead/results_summary.json \
  --results /tmp/dit_overhead/results.json
```
