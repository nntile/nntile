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
VRAM search: [`match_dit_vram_to_llama.py`](../../torch_nntile/tools/match_dit_vram_to_llama.py).

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
- **CUDA:** `--disable-tf32`. **nntile:** `--ncpu 0 --ncuda 1 --restrict-cuda`.

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
Includes **S HF(nntile) 100-step** steady-state run. Requires
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
| XS T=784 | 1.480 ± 0.059 s | 2.080 ± 0.159 s | 1.976 ± 0.031 s | **1.41×** | **1.33×** |
| S T=1024 | 2.841 ± 0.178 s | 3.015 ± 0.171 s | 2.871 ± 0.008 s | **1.06×** | **1.01×** |
| M T=1521 | 8.070 ± 0.152 s | 8.144 ± 0.167 s | 8.225 ± 0.013 s | **1.01×** | **1.02×** |
| L T=2025 | 17.579 ± 0.032 s | 17.307 ± 0.033 s | 17.880 ± 0.028 s | **0.98×** | **1.02×** |
| XL T=2916 | 22.342 ± 0.045 s | 21.635 ± 0.026 s | 22.421 ± 0.114 s | **0.97×** | **1.00×** |

nntile(nntile) tracks HF(cuda) on S–XL (**1.00–1.02×**). XS is host-bound
but faster than HF(nntile). Isolated GPU time: L **1.770** vs HF(cuda)
**1.741** s; XL **2.242** vs **2.220** s.
**XL train wall: HF(nntile) 0.97×, nntile(nntile) 1.00×.**

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
XL is **0.97×** HF(cuda) here. **nntile(nntile) XL is 1.00×** — see
[nntile(nntile) vs HF(cuda)](#nntilenntile-vs-hfcuda).
VRAM for HF(cuda) / HF(nntile) / nntile(nntile) is in
[Peak VRAM and bus](#peak-vram-and-bus) (`nvidia-smi`).

| Setup | HF(cuda) wall | HF(nntile) wall | HF(nntile) / HF(cuda) | record(nntile) | record(torch) | compile | run | wait | host/wall | HF(cuda) loss | HF(nntile) loss |
|-------|----------:|------------:|------------:|---------------:|--------------:|--------:|----:|-----:|----------:|----------:|------------:|
| XS T=784 | 1.480 ± 0.059 s | 2.080 ± 0.159 s | **1.41×** | 0.125 ± 0.004 s | 0.428 ± 0.028 s | 0.171 ± 0.014 s | 0.181 ± 0.011 s | 1.174 ± 0.163 s | **35.0%** | 1.209802 | **1.209802** |
| S T=1024 | 2.841 ± 0.178 s | 3.015 ± 0.171 s | **1.06×** | 0.119 ± 0.010 s | 0.381 ± 0.018 s | 0.142 ± 0.006 s | 0.166 ± 0.013 s | 2.206 ± 0.161 s | **21.3%** | 1.192550 | **1.192550** |
| M T=1521 | 8.070 ± 0.152 s | 8.144 ± 0.167 s | **1.01×** | 0.123 ± 0.009 s | 0.392 ± 0.014 s | 0.142 ± 0.004 s | 0.180 ± 0.012 s | 7.306 ± 0.162 s | **8.1%** | 1.141145 | **1.141145** |
| L T=2025 | 17.579 ± 0.032 s | 17.307 ± 0.033 s | **0.98×** | 0.124 ± 0.009 s | 0.398 ± 0.026 s | 0.147 ± 0.004 s | 0.189 ± 0.011 s | 16.448 ± 0.056 s | **3.9%** | 1.221610 | **1.221610** |
| XL T=2916 | 22.342 ± 0.045 s | 21.635 ± 0.026 s | **0.97×** | 0.080 ± 0.016 s | 0.249 ± 0.025 s | 0.091 ± 0.013 s | 0.104 ± 0.019 s | 21.110 ± 0.068 s | **1.9%** | 1.034324 | **1.034324** |

Host = `record(nntile)+record(torch)+compile` (~0.29–0.51 s for 10 steps,
**flat**). Host **share** drops **35.0% → 21.3% → 8.1% → 3.9% → 1.9%**
as GPU work grows.

MSE noise-prediction loss matches HF(cuda) vs HF(nntile) to printed 1e-4 at all ladder sizes (XS 1.209802 both).

HF(nntile) isolated GPU `run+wait` vs HF(cuda) isolated wall:
XS 0.158 ± 0.005 vs 0.127 ± 0.001 s, S 0.256 ± 0.002 vs 0.244 ± 0.001 s, M 0.762 ± 0.002 vs 0.766 ± 0.002 s, L 1.700 ± 0.004 vs 1.741 ± 0.005 s, XL 2.143 ± 0.003 vs 2.220 ± 0.004 s.

## nntile(nntile) vs HF(cuda)

nntile(nntile) only, overlap, 10 steps, **10 repeats** (mean ± stdev).
`STARPU_LIMIT_CUDA_MEM=46000`. Host patchify is outside the train wall.
HF(cuda) walls are the published 10-repeat means. Peak VRAM / H2D / D2H
below are **nntile(nntile)**. HF(cuda) VRAM and HF(nntile) bus stats are in
[Peak VRAM and bus](#peak-vram-and-bus).

| Setup | HF(cuda) wall | nntile(nntile) wall | nntile(nntile) / HF(cuda) | isolated | peak VRAM | H2D | D2H | host/wall | nntile(nntile) loss |
|-------|----------:|-------------:|-------------:|---------:|----------:|----:|----:|----------:|-------------:|
| XS T=784 | 1.480 ± 0.059 s | 1.976 ± 0.031 s | **1.33×** | 0.176 ± 0.001 s | 6.8 GiB | 1.95 GB | **0** | **32.7%** | 1.209802 |
| S T=1024 | 2.841 ± 0.178 s | 2.871 ± 0.008 s | **1.01×** | 0.268 ± 0.001 s | 9.4 GiB | 3.12 GB | **0** | **19.5%** | 1.192550 |
| M T=1521 | 8.070 ± 0.152 s | 8.225 ± 0.013 s | **1.02×** | 0.805 ± 0.002 s | 23.5 GiB | 7.61 GB | **0** | **7.3%** | 1.141145 |
| L T=2025 | 17.579 ± 0.032 s | 17.880 ± 0.028 s | **1.02×** | 1.770 ± 0.004 s | **42.7 GiB** | 13.45 GB | **0** | **3.3%** | 1.221610 |
| XL T=2916 | 22.342 ± 0.045 s | 22.421 ± 0.114 s | **1.00×** | 2.242 ± 0.010 s | **43.6 GiB** | 12.20 GB | **0** | **1.8%** | 1.034324 |

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
| record(nntile) | 1.337 ± 0.055 s | 13.4 ms |
| record(torch) | 5.251 ± 0.353 s | 53 ms |
| compile | 1.656 ± 0.062 s | 17 ms |
| run | 1.797 ± 0.062 s | 18 ms |
| wait | 16.670 ± 0.635 s | 167 ms |
| **train wall** | **26.721 ± 0.274 s** | 267 ms |

Host (record + compile) is **31%** of the wall (~82 ms/step).

![Host overhead per iteration](dit_hf_overhead_s_100.svg)

CSV: [`dit_hf_overhead_s_100.csv`](dit_hf_overhead_s_100.csv) (median of 10 runs).

## Comparison to GPT-2 (wall time only)

GPT-2 uses the same **hidden / token-count ladder** but a different task
(causal LM cross-entropy vs DiT MSE noise prediction). Compare **HF(nntile) / HF(cuda)
wall ratios** only; loss values are not comparable.
nntile(nntile) is not in this table (DiT XL nntile(nntile) is **1.00×**).

See [`gpt2_hf_overhead_scale.md`](gpt2_hf_overhead_scale.md).

| Size | GPT-2 HF(nntile)/HF(cuda) | DiT HF(nntile)/HF(cuda) |
|------|------------------:|-----------------:|
| XS | 0.99× | **1.41×** |
| S | 0.96× | **1.06×** |
| M | 0.94× | **1.01×** |
| L | 0.94× | **0.98×** |
| XL | 0.96× | **0.97×** |

### 100-step S (HF(nntile))

| | GPT-2 | DiT | Notes |
|--|------:|-----:|-------|
| train wall | 27.5 s | **26.7 s** | same ballpark |
| final loss | 7.734033 (LM CE) | **1.226684** (MSE noise) | different task |
| host share | 22% | **31%** | flat host, GPU-bound |

## Per iteration (HF(nntile), mean ± stdev over 10 runs)

### XS (hidden 1536, T=784 patches)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.342 ± 0.058 | 0.008 ± 0.001 | 0.027 ± 0.002 | 0.011 ± 0.001 | 0.013 ± 0.001 | 0.000 |
| 2 | 0.126 ± 0.001 | 0.007 ± 0.000 | 0.023 ± 0.002 | 0.011 ± 0.001 | 0.017 ± 0.003 | 0.507 ± 0.161 |
| 3 | 0.126 ± 0.001 | 0.013 ± 0.002 | 0.035 ± 0.004 | 0.017 ± 0.002 | 0.016 ± 0.002 | 0.079 ± 0.009 |
| 4 | 0.127 ± 0.001 | 0.013 ± 0.001 | 0.036 ± 0.003 | 0.017 ± 0.002 | 0.019 ± 0.002 | 0.078 ± 0.007 |
| 5 | 0.127 ± 0.001 | 0.014 ± 0.002 | 0.040 ± 0.002 | 0.019 ± 0.003 | 0.018 ± 0.002 | 0.071 ± 0.009 |
| 6 | 0.127 ± 0.001 | 0.013 ± 0.001 | 0.045 ± 0.005 | 0.018 ± 0.003 | 0.019 ± 0.003 | 0.070 ± 0.008 |
| 7 | 0.127 ± 0.001 | 0.014 ± 0.001 | 0.050 ± 0.007 | 0.019 ± 0.003 | 0.020 ± 0.003 | 0.064 ± 0.013 |
| 8 | 0.127 ± 0.001 | 0.014 ± 0.001 | 0.055 ± 0.008 | 0.018 ± 0.002 | 0.020 ± 0.002 | 0.057 ± 0.013 |
| 9 | 0.127 ± 0.001 | 0.014 ± 0.002 | 0.058 ± 0.009 | 0.019 ± 0.003 | 0.020 ± 0.003 | 0.055 ± 0.017 |
| 10 | 0.127 ± 0.001 | 0.014 ± 0.001 | 0.059 ± 0.007 | 0.022 ± 0.004 | 0.019 ± 0.002 | 0.192 ± 0.017 |

### S (hidden 2048, T=1024 patches)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.638 ± 0.179 | 0.006 ± 0.000 | 0.024 ± 0.001 | 0.011 ± 0.002 | 0.013 ± 0.002 | 0.000 |
| 2 | 0.245 ± 0.001 | 0.007 ± 0.001 | 0.022 ± 0.001 | 0.011 ± 0.002 | 0.016 ± 0.004 | 0.580 ± 0.164 |
| 3 | 0.245 ± 0.001 | 0.012 ± 0.003 | 0.031 ± 0.003 | 0.013 ± 0.001 | 0.015 ± 0.003 | 0.186 ± 0.009 |
| 4 | 0.245 ± 0.001 | 0.012 ± 0.002 | 0.034 ± 0.004 | 0.014 ± 0.002 | 0.018 ± 0.003 | 0.182 ± 0.010 |
| 5 | 0.245 ± 0.001 | 0.014 ± 0.002 | 0.039 ± 0.003 | 0.015 ± 0.001 | 0.017 ± 0.002 | 0.174 ± 0.007 |
| 6 | 0.245 ± 0.001 | 0.014 ± 0.002 | 0.041 ± 0.004 | 0.016 ± 0.001 | 0.017 ± 0.003 | 0.173 ± 0.006 |
| 7 | 0.245 ± 0.001 | 0.013 ± 0.002 | 0.044 ± 0.004 | 0.015 ± 0.002 | 0.017 ± 0.003 | 0.170 ± 0.008 |
| 8 | 0.245 ± 0.001 | 0.013 ± 0.001 | 0.046 ± 0.003 | 0.016 ± 0.002 | 0.018 ± 0.003 | 0.169 ± 0.006 |
| 9 | 0.245 ± 0.001 | 0.014 ± 0.002 | 0.049 ± 0.003 | 0.016 ± 0.001 | 0.018 ± 0.002 | 0.164 ± 0.008 |
| 10 | 0.244 ± 0.001 | 0.014 ± 0.002 | 0.050 ± 0.002 | 0.016 ± 0.001 | 0.018 ± 0.003 | 0.408 ± 0.006 |

### M (hidden 3072, T=1521 patches)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 1.179 ± 0.137 | 0.007 | 0.027 ± 0.002 | 0.010 ± 0.000 | 0.013 ± 0.001 | 0.000 |
| 2 | 0.766 ± 0.002 | 0.007 ± 0.000 | 0.023 ± 0.001 | 0.011 ± 0.001 | 0.015 ± 0.003 | 1.168 ± 0.164 |
| 3 | 0.765 ± 0.002 | 0.012 ± 0.003 | 0.032 ± 0.004 | 0.013 ± 0.000 | 0.018 ± 0.004 | 0.690 ± 0.010 |
| 4 | 0.765 ± 0.003 | 0.014 ± 0.002 | 0.037 ± 0.003 | 0.015 ± 0.002 | 0.019 ± 0.003 | 0.681 ± 0.010 |
| 5 | 0.766 ± 0.003 | 0.013 ± 0.002 | 0.039 ± 0.003 | 0.015 ± 0.001 | 0.019 ± 0.003 | 0.677 ± 0.009 |
| 6 | 0.766 ± 0.003 | 0.014 ± 0.001 | 0.042 ± 0.002 | 0.015 ± 0.001 | 0.019 ± 0.003 | 0.674 ± 0.006 |
| 7 | 0.766 ± 0.003 | 0.014 ± 0.002 | 0.046 ± 0.003 | 0.015 ± 0.001 | 0.018 ± 0.002 | 0.668 ± 0.007 |
| 8 | 0.765 ± 0.003 | 0.013 ± 0.002 | 0.047 ± 0.003 | 0.015 ± 0.001 | 0.021 ± 0.003 | 0.669 ± 0.007 |
| 9 | 0.766 ± 0.002 | 0.014 ± 0.001 | 0.049 ± 0.005 | 0.015 ± 0.000 | 0.019 ± 0.002 | 0.665 ± 0.010 |
| 10 | 0.765 ± 0.002 | 0.014 ± 0.003 | 0.050 ± 0.003 | 0.017 ± 0.001 | 0.018 ± 0.002 | 1.413 ± 0.009 |

### L (hidden 4096, T=2025 patches)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 1.953 ± 0.011 | 0.007 | 0.027 ± 0.001 | 0.011 ± 0.001 | 0.013 ± 0.001 | 0.000 |
| 2 | 1.733 ± 0.003 | 0.007 ± 0.000 | 0.024 ± 0.001 | 0.011 ± 0.001 | 0.017 ± 0.003 | 1.910 ± 0.009 |
| 3 | 1.736 ± 0.003 | 0.012 ± 0.003 | 0.034 ± 0.006 | 0.014 ± 0.001 | 0.017 ± 0.003 | 1.616 ± 0.011 |
| 4 | 1.737 ± 0.003 | 0.012 ± 0.002 | 0.036 ± 0.003 | 0.015 ± 0.001 | 0.019 ± 0.003 | 1.617 ± 0.008 |
| 5 | 1.736 ± 0.004 | 0.015 ± 0.002 | 0.041 ± 0.003 | 0.015 ± 0.000 | 0.019 ± 0.003 | 1.609 ± 0.005 |
| 6 | 1.736 ± 0.002 | 0.013 ± 0.002 | 0.043 ± 0.004 | 0.015 ± 0.001 | 0.021 ± 0.004 | 1.609 ± 0.011 |
| 7 | 1.736 ± 0.004 | 0.014 ± 0.001 | 0.045 ± 0.004 | 0.016 ± 0.001 | 0.022 ± 0.003 | 1.603 ± 0.009 |
| 8 | 1.736 ± 0.005 | 0.015 ± 0.002 | 0.048 ± 0.005 | 0.016 ± 0.001 | 0.020 ± 0.003 | 1.598 ± 0.012 |
| 9 | 1.737 ± 0.005 | 0.014 ± 0.002 | 0.049 ± 0.005 | 0.015 ± 0.001 | 0.020 ± 0.003 | 1.601 ± 0.008 |
| 10 | 1.739 ± 0.005 | 0.015 ± 0.002 | 0.051 ± 0.008 | 0.019 ± 0.003 | 0.020 ± 0.003 | 3.285 ± 0.016 |

### XL (hidden 5760, T=2916 patches, 5 layers)

| Iter | HF(cuda) wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 2.465 ± 0.008 | 0.006 ± 0.001 | 0.015 ± 0.001 | 0.005 ± 0.001 | 0.007 ± 0.002 | 0.000 |
| 2 | 2.194 ± 0.008 | 0.004 ± 0.001 | 0.012 ± 0.001 | 0.005 ± 0.001 | 0.008 ± 0.001 | 2.348 ± 0.017 |
| 3 | 2.200 ± 0.007 | 0.006 ± 0.000 | 0.017 ± 0.001 | 0.008 ± 0.001 | 0.009 ± 0.002 | 2.082 ± 0.005 |
| 4 | 2.203 ± 0.007 | 0.007 ± 0.002 | 0.020 ± 0.002 | 0.009 ± 0.001 | 0.011 ± 0.002 | 2.081 ± 0.006 |
| 5 | 2.208 ± 0.005 | 0.009 ± 0.003 | 0.025 ± 0.004 | 0.011 ± 0.002 | 0.013 ± 0.003 | 2.078 ± 0.007 |
| 6 | 2.210 ± 0.004 | 0.011 ± 0.002 | 0.029 ± 0.003 | 0.011 ± 0.002 | 0.012 ± 0.003 | 2.071 ± 0.009 |
| 7 | 2.212 ± 0.005 | 0.009 ± 0.003 | 0.030 ± 0.004 | 0.011 ± 0.002 | 0.011 ± 0.003 | 2.077 ± 0.011 |
| 8 | 2.215 ± 0.005 | 0.010 ± 0.003 | 0.033 ± 0.004 | 0.010 ± 0.002 | 0.011 ± 0.004 | 2.078 ± 0.012 |
| 9 | 2.216 ± 0.004 | 0.008 ± 0.003 | 0.033 ± 0.004 | 0.010 ± 0.002 | 0.011 ± 0.003 | 2.080 ± 0.013 |
| 10 | 2.218 ± 0.004 | 0.009 ± 0.003 | 0.034 ± 0.006 | 0.010 ± 0.002 | 0.011 ± 0.004 | 4.215 ± 0.011 |

## Isolated extra step (HF(nntile), mean ± stdev over 10 runs)

| Setup | record(nntile) | record(torch) | compile | run | wait | run+wait | HF(cuda) isolated |
|-------|---------------:|--------------:|--------:|----:|-----:|---------:|--------------:|
| XS | 0.016 ± 0.002 | 0.063 ± 0.009 | 0.018 ± 0.001 | 0.019 ± 0.002 | 0.139 ± 0.005 | **0.158 ± 0.005** | 0.127 ± 0.001 |
| S | 0.018 ± 0.004 | 0.058 ± 0.006 | 0.020 ± 0.003 | 0.018 ± 0.002 | 0.239 ± 0.001 | **0.256 ± 0.002** | 0.244 ± 0.001 |
| M | 0.018 ± 0.002 | 0.059 ± 0.003 | 0.019 ± 0.002 | 0.019 ± 0.002 | 0.744 ± 0.003 | **0.762 ± 0.002** | 0.766 ± 0.002 |
| L | 0.018 ± 0.003 | 0.056 ± 0.009 | 0.019 ± 0.002 | 0.018 ± 0.001 | 1.681 ± 0.005 | **1.700 ± 0.004** | 1.741 ± 0.005 |
| XL | 0.010 ± 0.004 | 0.037 ± 0.006 | 0.013 ± 0.004 | 0.011 ± 0.003 | 2.133 ± 0.003 | **2.143 ± 0.003** | 2.220 ± 0.004 |

| Setup | Full isolated (record+compile+run+wait) | Hidden host (`run+wait`) | Saved |
|-------|----------------------------------------:|-------------------------:|------:|
| XS | 0.255 s | 0.158 s | 0.097 s (**38%**) |
| S | 0.352 s | 0.256 s | 0.096 s (**27%**) |
| M | 0.858 s | 0.762 s | 0.096 s (**11%**) |
| L | 1.792 s | 1.700 s | 0.093 s (**5%**) |
| XL | 2.203 s | 2.143 s | 0.060 s (**3%**) |

## Sequential prep vs compute (`--wait-after-run`, HF(nntile))

| Setup | HF(cuda) wall | sequential wall | prep | compute | compute / HF(cuda) | prep/wall |
|-------|----------:|----------------:|-----:|--------:|-------------:|----------:|
| XS T=784 | 1.480 ± 0.059 s | 2.709 ± 0.180 s | 0.734 ± 0.009 s | **1.974 ± 0.177 s** | **1.33×** | 27.1% |
| S T=1024 | 2.841 ± 0.178 s | 3.618 ± 0.146 s | 0.720 ± 0.010 s | **2.896 ± 0.153 s** | **1.02×** | 19.9% |
| M T=1521 | 8.070 ± 0.152 s | 8.760 ± 0.155 s | 0.728 ± 0.020 s | **8.031 ± 0.153 s** | **1.00×** | 8.3% |
| L T=2025 | 17.579 ± 0.032 s | 17.951 ± 0.024 s | 0.748 ± 0.020 s | **17.202 ± 0.024 s** | **0.98×** | 4.2% |
| XL T=2916 | 22.342 ± 0.045 s | 21.998 ± 0.119 s | 0.458 ± 0.067 s | **21.536 ± 0.075 s** | **0.96×** | 2.1% |

Sequential HF(nntile) loss: XS 1.209802, S 1.192550, M 1.141145, L 1.221610, XL 1.034324.

### Per iteration (prep / compute, mean ± stdev)

#### XS (T=784)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.044 ± 0.002 | 0.594 ± 0.175 | 0.007 ± 0.000 | 0.027 ± 0.002 | 0.010 ± 0.001 | 0.012 ± 0.001 | 0.582 ± 0.175 |
| 2 | 0.044 ± 0.003 | 0.152 ± 0.001 | 0.009 ± 0.001 | 0.025 ± 0.001 | 0.010 ± 0.001 | 0.012 ± 0.001 | 0.140 ± 0.001 |
| 3 | 0.058 ± 0.004 | 0.153 ± 0.001 | 0.012 ± 0.001 | 0.032 ± 0.002 | 0.014 ± 0.002 | 0.015 ± 0.002 | 0.138 ± 0.002 |
| 4 | 0.068 ± 0.004 | 0.152 ± 0.001 | 0.014 ± 0.001 | 0.038 ± 0.002 | 0.016 ± 0.001 | 0.017 ± 0.001 | 0.136 ± 0.001 |
| 5 | 0.079 ± 0.002 | 0.153 ± 0.001 | 0.017 ± 0.001 | 0.043 ± 0.002 | 0.019 ± 0.001 | 0.018 ± 0.001 | 0.135 ± 0.002 |
| 6 | 0.082 ± 0.003 | 0.153 ± 0.001 | 0.017 ± 0.001 | 0.046 ± 0.002 | 0.018 ± 0.001 | 0.018 ± 0.001 | 0.135 ± 0.001 |
| 7 | 0.085 ± 0.003 | 0.154 ± 0.001 | 0.018 ± 0.002 | 0.048 ± 0.001 | 0.018 ± 0.001 | 0.019 ± 0.001 | 0.135 ± 0.002 |
| 8 | 0.088 ± 0.002 | 0.153 ± 0.002 | 0.018 ± 0.001 | 0.052 ± 0.001 | 0.018 ± 0.001 | 0.018 ± 0.001 | 0.136 ± 0.002 |
| 9 | 0.091 ± 0.003 | 0.154 ± 0.001 | 0.019 ± 0.003 | 0.054 ± 0.001 | 0.018 ± 0.001 | 0.019 ± 0.001 | 0.136 ± 0.002 |
| 10 | 0.094 ± 0.001 | 0.155 ± 0.001 | 0.018 ± 0.001 | 0.056 ± 0.001 | 0.020 ± 0.001 | 0.019 ± 0.001 | 0.136 ± 0.001 |

#### S (T=1024)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.040 ± 0.001 | 0.604 ± 0.149 | 0.006 ± 0.000 | 0.024 ± 0.001 | 0.009 ± 0.000 | 0.012 ± 0.001 | 0.593 ± 0.149 |
| 2 | 0.043 ± 0.002 | 0.253 ± 0.001 | 0.009 ± 0.001 | 0.024 ± 0.001 | 0.010 ± 0.001 | 0.012 ± 0.002 | 0.241 ± 0.002 |
| 3 | 0.056 ± 0.002 | 0.254 ± 0.001 | 0.012 ± 0.001 | 0.030 ± 0.001 | 0.013 ± 0.001 | 0.015 ± 0.002 | 0.239 ± 0.003 |
| 4 | 0.066 ± 0.002 | 0.254 ± 0.001 | 0.015 ± 0.001 | 0.036 ± 0.001 | 0.016 ± 0.001 | 0.016 ± 0.001 | 0.237 ± 0.001 |
| 5 | 0.076 ± 0.002 | 0.255 ± 0.001 | 0.017 ± 0.001 | 0.041 ± 0.001 | 0.018 ± 0.001 | 0.017 ± 0.000 | 0.237 ± 0.001 |
| 6 | 0.082 ± 0.003 | 0.255 ± 0.001 | 0.018 ± 0.001 | 0.045 ± 0.001 | 0.019 ± 0.001 | 0.018 ± 0.001 | 0.237 ± 0.002 |
| 7 | 0.085 ± 0.002 | 0.255 ± 0.001 | 0.018 ± 0.001 | 0.048 ± 0.001 | 0.019 | 0.018 ± 0.001 | 0.237 ± 0.001 |
| 8 | 0.088 ± 0.002 | 0.256 ± 0.001 | 0.019 ± 0.001 | 0.051 ± 0.001 | 0.019 ± 0.000 | 0.019 ± 0.001 | 0.237 ± 0.001 |
| 9 | 0.090 ± 0.003 | 0.256 ± 0.002 | 0.018 ± 0.001 | 0.053 ± 0.001 | 0.019 ± 0.001 | 0.018 ± 0.001 | 0.237 ± 0.002 |
| 10 | 0.095 ± 0.002 | 0.256 ± 0.002 | 0.019 ± 0.001 | 0.057 ± 0.002 | 0.019 | 0.019 ± 0.001 | 0.237 ± 0.001 |

#### M (T=1521)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.045 ± 0.002 | 1.195 ± 0.148 | 0.007 ± 0.000 | 0.027 ± 0.002 | 0.010 ± 0.001 | 0.013 ± 0.001 | 1.182 ± 0.147 |
| 2 | 0.045 ± 0.004 | 0.760 ± 0.001 | 0.009 ± 0.001 | 0.025 ± 0.002 | 0.010 ± 0.001 | 0.012 ± 0.001 | 0.748 ± 0.001 |
| 3 | 0.058 ± 0.004 | 0.760 ± 0.002 | 0.012 ± 0.002 | 0.032 ± 0.002 | 0.014 ± 0.002 | 0.014 ± 0.001 | 0.746 ± 0.003 |
| 4 | 0.069 ± 0.003 | 0.758 ± 0.004 | 0.015 ± 0.001 | 0.038 ± 0.001 | 0.016 ± 0.001 | 0.016 ± 0.001 | 0.742 ± 0.004 |
| 5 | 0.077 ± 0.003 | 0.760 ± 0.004 | 0.017 ± 0.001 | 0.042 ± 0.002 | 0.018 ± 0.001 | 0.018 ± 0.001 | 0.742 ± 0.004 |
| 6 | 0.079 ± 0.006 | 0.759 ± 0.004 | 0.016 ± 0.002 | 0.045 ± 0.002 | 0.017 ± 0.002 | 0.017 ± 0.002 | 0.742 ± 0.004 |
| 7 | 0.083 ± 0.005 | 0.759 ± 0.003 | 0.017 ± 0.002 | 0.049 ± 0.002 | 0.018 ± 0.003 | 0.018 ± 0.003 | 0.740 ± 0.003 |
| 8 | 0.086 ± 0.004 | 0.759 ± 0.003 | 0.017 ± 0.002 | 0.051 ± 0.001 | 0.018 ± 0.001 | 0.019 ± 0.002 | 0.740 ± 0.003 |
| 9 | 0.091 ± 0.002 | 0.759 ± 0.002 | 0.018 ± 0.001 | 0.054 ± 0.001 | 0.019 ± 0.001 | 0.019 ± 0.001 | 0.740 ± 0.003 |
| 10 | 0.096 ± 0.003 | 0.761 ± 0.003 | 0.019 ± 0.001 | 0.056 ± 0.001 | 0.021 ± 0.001 | 0.019 ± 0.001 | 0.742 ± 0.003 |

#### L (T=2025)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.044 ± 0.003 | 1.962 ± 0.011 | 0.007 ± 0.001 | 0.026 ± 0.002 | 0.011 ± 0.001 | 0.013 ± 0.001 | 1.949 ± 0.011 |
| 2 | 0.050 ± 0.002 | 1.689 ± 0.003 | 0.010 ± 0.001 | 0.027 ± 0.001 | 0.013 ± 0.001 | 0.014 ± 0.002 | 1.676 ± 0.003 |
| 3 | 0.062 ± 0.002 | 1.692 ± 0.002 | 0.013 ± 0.001 | 0.033 ± 0.001 | 0.015 ± 0.001 | 0.015 ± 0.001 | 1.677 ± 0.003 |
| 4 | 0.070 ± 0.006 | 1.692 ± 0.003 | 0.015 ± 0.002 | 0.038 ± 0.002 | 0.017 ± 0.002 | 0.016 ± 0.001 | 1.676 ± 0.003 |
| 5 | 0.077 ± 0.008 | 1.695 ± 0.003 | 0.016 ± 0.003 | 0.043 ± 0.004 | 0.018 ± 0.002 | 0.017 ± 0.002 | 1.678 ± 0.004 |
| 6 | 0.083 ± 0.006 | 1.694 ± 0.004 | 0.017 ± 0.002 | 0.047 ± 0.002 | 0.019 ± 0.002 | 0.019 ± 0.002 | 1.674 ± 0.006 |
| 7 | 0.085 ± 0.004 | 1.693 ± 0.003 | 0.017 ± 0.002 | 0.049 ± 0.001 | 0.019 ± 0.001 | 0.018 ± 0.002 | 1.675 ± 0.004 |
| 8 | 0.088 ± 0.004 | 1.695 ± 0.003 | 0.018 ± 0.002 | 0.052 ± 0.002 | 0.018 ± 0.001 | 0.017 ± 0.002 | 1.678 ± 0.004 |
| 9 | 0.091 ± 0.003 | 1.695 ± 0.003 | 0.018 ± 0.001 | 0.054 ± 0.001 | 0.019 ± 0.001 | 0.018 ± 0.001 | 1.677 ± 0.003 |
| 10 | 0.098 ± 0.003 | 1.696 ± 0.002 | 0.018 ± 0.001 | 0.057 ± 0.002 | 0.022 ± 0.002 | 0.018 ± 0.001 | 1.677 ± 0.003 |

#### XL (T=2916)

| Iter | prep | compute | record(nntile) | record(torch) | compile | run | wait |
|-----:|-----:|--------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.025 ± 0.001 | 2.372 ± 0.018 | 0.005 ± 0.001 | 0.015 ± 0.001 | 0.005 ± 0.000 | 0.006 ± 0.000 | 2.366 ± 0.018 |
| 2 | 0.030 ± 0.001 | 2.115 ± 0.006 | 0.007 ± 0.001 | 0.016 ± 0.001 | 0.008 ± 0.001 | 0.007 ± 0.000 | 2.108 ± 0.007 |
| 3 | 0.036 ± 0.003 | 2.120 ± 0.009 | 0.008 ± 0.001 | 0.020 ± 0.001 | 0.009 ± 0.001 | 0.008 ± 0.001 | 2.112 ± 0.009 |
| 4 | 0.042 ± 0.008 | 2.126 ± 0.009 | 0.009 ± 0.002 | 0.023 ± 0.004 | 0.010 ± 0.002 | 0.010 ± 0.002 | 2.116 ± 0.007 |
| 5 | 0.050 ± 0.009 | 2.127 ± 0.009 | 0.010 ± 0.003 | 0.027 ± 0.003 | 0.012 ± 0.003 | 0.011 ± 0.003 | 2.116 ± 0.007 |
| 6 | 0.052 ± 0.009 | 2.132 ± 0.008 | 0.011 ± 0.003 | 0.030 ± 0.005 | 0.011 ± 0.002 | 0.011 ± 0.002 | 2.121 ± 0.007 |
| 7 | 0.056 ± 0.009 | 2.132 ± 0.008 | 0.011 ± 0.003 | 0.032 ± 0.005 | 0.013 ± 0.002 | 0.012 ± 0.003 | 2.120 ± 0.008 |
| 8 | 0.057 ± 0.010 | 2.136 ± 0.008 | 0.011 ± 0.003 | 0.034 ± 0.006 | 0.012 ± 0.002 | 0.011 ± 0.002 | 2.124 ± 0.008 |
| 9 | 0.056 ± 0.015 | 2.137 ± 0.008 | 0.011 ± 0.004 | 0.034 ± 0.008 | 0.011 ± 0.004 | 0.011 ± 0.004 | 2.127 ± 0.009 |
| 10 | 0.054 ± 0.016 | 2.139 ± 0.006 | 0.010 ± 0.004 | 0.033 ± 0.009 | 0.011 ± 0.004 | 0.010 ± 0.003 | 2.128 ± 0.007 |

Steady compute after iter 1 (mean over repeats): ~0.152 s (XS), ~0.253 s (S), ~0.760 s (M), ~1.693 s (L), ~2.129 s (XL).

## Takeaways

1. **Diffusers DiT**, synthetic diffusion batches, MSE noise loss; ladder
   geometry aligned to Llama via hidden size + patch count + HF(cuda) VRAM match.
2. **HF(nntile) graph host overhead is flat** (~0.3–0.5 s / 10 steps); share
   falls as GPU work grows (35.0% → 1.9%).
3. **HF(nntile)** is within ~5–40% of HF(cuda) on wall time
   (XS 1.41×, S 1.06×, M 1.01×, L **0.98×**, XL **0.97×**). XS is host-bound;
   **HF(nntile)** M/L/XL are near parity. That XL **0.97× is not
   nntile(nntile)** — nntile(nntile) XL is **1.00×** (takeaway 9).
4. **L=11 / XL=5** are the published json so nntile(nntile) has D2H **0**.
   Isolated GPU time is still a bit above HF(cuda) because AdaLN-Zero is
   six `H→H` GEMMs, not a fused `H→6H`.
5. **HF(nntile) sequential GPU time** (`run+wait`): **1.33× → 1.02× → 1.00× → 0.98× → 0.96×** vs HF(cuda).
6. Timings are **mean ± stdev** over 10 runs.
7. **MSE loss** matches all three setups to printed 1e-6 — see
   [Three setups](#three-setups).
8. **100-step S HF(nntile)** wall **26.721 ± 0.274 s** — see section above.
9. nntile(nntile): **1.00–1.02×** HF(cuda) on S–XL (D2H **0**); XS **1.33×**
   (host-bound, faster than HF(nntile) **1.41×**). Peak VRAM L **42.7 GiB**,
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

# VRAM-match configs (optional --apply writes overhead_dit/*.json):
.venv/bin/python torch_nntile/tools/match_dit_vram_to_llama.py 1 --apply

# Full ladder, 10 repeats, one GPU (HF(cuda) / HF(nntile)):
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
