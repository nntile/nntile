# VGG HF: graph overhead vs width / spatial size

**Notation.** Each label is **implementation(backend)**.

- **HF** — stock PyTorch `torch.nn` CNN (`TinyVGG` in
  [`train_vgg_tiny.py`](../../torch_nntile/examples/train_vgg_tiny.py)).
  Not HuggingFace Transformers; the name matches the transformer overhead
  docs (implementation outside the brackets).
- **cuda** — PyTorch CUDA (`device=cuda`).
- **nntile** (as backend) — StarPU / nntile (`device=nntile`).

**There is no nntile(nntile) VGG.** `torch_nntile.models` has no CNN
ports; this study is only **HF(cuda)** vs **HF(nntile)**.

Two setups, same configs / 10 steps:

1. **HF(cuda)** — stock `TinyVGG`, `device=cuda`, no `torch_nntile` import.
2. **HF(nntile)** — same graph on `device=nntile` (aten / torch-native
   StarPU codelets: stacked `convolution_overrideable`, MaxPool,
   AdaptiveAvgPool2d, Linear).

> **VRAM warning.** Nntile keeps extra graph buffers. Keep HF(cuda) well
> under the card limit so `device=nntile` stays on-device (no StarPU
> CPU↔GPU paging). Four MaxPool stages need `height` / `width` divisible
> by 16 (XL has three stages → divisible by 8). GPUs are in exclusive
> mode — one process per GPU.

Configs: [`torch_nntile/examples/overhead_vgg/`](../../torch_nntile/examples/overhead_vgg/).
HF(cuda) / HF(nntile):
[`train_cnn_hf_overhead.py`](../../torch_nntile/examples/train_cnn_hf_overhead.py)
(`--model vgg`),
[`run_vgg_overhead_benchmark.py`](../../torch_nntile/tools/run_vgg_overhead_benchmark.py).

## Loss

Classification CE on synthetic RGB images vs class labels (new batch
seed per step: `42 + step`). Same `F.cross_entropy` on HF(cuda) and
HF(nntile).

## Train wall

Same recipe as
[`gpt2_hf_overhead_scale.md`](gpt2_hf_overhead_scale.md): nntile
`record → compile → wait(prev) → run`, wall from first record through
final `wait()`; HF(cuda) synced per iter. Prefetch outside the wall.
Iter 1 nntile `wait=0`; iter 10 `wait` includes the final join.

## Recipe

Spatial size grows with stage width. **XL** drops to **3 stages** (depth
analog) at 144² so HF(nntile) fills an A40.

| | XS | S | M | L | XL |
|--|--:|--:|--:|--:|--:|
| Config | `vgg_xs.json` | `vgg_s.json` | `vgg_m.json` | `vgg_l.json` | `vgg_xl.json` |
| stages (`channels`) | 4 | 4 | 4 | 4 | **3** |
| `channels` | 1024…4096 | 1280…5120 | 1792…7168 | 2304…9216 | 3072…9216 |
| `height` × `width` | 32² | 48² | 64² | 80² | **144²** |
| `fc_hidden` | 4096 | 2048 | 2048 | 2048 | **1024** |
| Params (FP32) | 611 M (2.28 GiB) | 940 M (3.50 GiB) | 1.84 B (6.84 GiB) | 3.03 B (11.28 GiB) | 1.88 B (7.00 GiB) |

B=1, 10 steps, seed 42, `--no-shuffle`, HF(cuda) and HF(nntile) `--disable-tf32`,
`device=nntile` `--ncpu 0 --ncuda 1 --restrict-cuda`. NVIDIA A40, one GPU
per job. Separate processes (`PYTHONNOUSERSITE=1`; never import
`torch_nntile` in the HF(cuda) process). **Do not overlap jobs on one GPU.**

HF(cuda) / HF(nntile): **10 repeats** (mean ± stdev), including **S HF(nntile) 100-step**.
`STARPU_LIMIT_CUDA_MEM=46000`.

## Two setups

### Loss

| Setup | HF(cuda) | HF(nntile) |
|-------|-----:|----------------:|
| XS 32² | 2.308218 | 2.308218 |
| S 48² | 2.323977 | 2.323977 |
| M 64² | 2.319142 | 2.319142 |
| L 80² | 2.281812 | 2.281812 |
| XL 144² | 2.354988 | 2.354988 |

### 10-step train wall

**10 repeats** (mean ± stdev), `STARPU_LIMIT_CUDA_MEM=46000`.

| Setup | HF(cuda) | HF(nntile) | HF(nntile) / HF(cuda) |
|-------|-----:|---------:|--------------:|
| XS 32² | 0.645 ± 0.080 s | 0.698 ± 0.130 s | **1.08×** |
| S 48² | 1.064 ± 0.099 s | 1.133 ± 0.150 s | **1.07×** |
| M 64² | 2.477 ± 0.155 s | 2.521 ± 0.193 s | **1.02×** |
| L 80² | 5.285 ± 0.147 s | 5.369 ± 0.148 s | **1.02×** |
| XL 144² | 18.047 ± 0.171 s | 18.284 ± 0.172 s | **1.01×** |

### Peak VRAM and bus

10-step overlap, NVIDIA A40, `STARPU_LIMIT_CUDA_MEM=46000`, B=1. Peak VRAM is `nvidia-smi memory.used` polled by a sidecar process (`watch_gpu_peak.py`) while the train child runs (`peak_vram_gib=`). H2D/D2H are StarPU bus stats at shutdown from the VRAM-fit probe. **D2H is 0 on every size**.

| Setup | HF(cuda) VRAM | HF(nntile) VRAM | H2D | D2H |
|-------|----------:|----------------:|----:|----:|
| XS 32² | 5.0 GiB | 5.6 GiB | 2.28 GB | **0** |
| S 48² | 7.6 GiB | 8.4 GiB | 3.50 GB | **0** |
| M 64² | 14.5 GiB | 16.3 GiB | 6.84 GB | **0** |
| L 80² | 24.0 GiB | 26.8 GiB | 11.28 GB | **0** |
| XL 144² | 18.1 GiB | 25.7 GiB | 7.00 GB | **0** |

## HF(nntile) vs HF(cuda) (10 repeats)

Overlap mode. Host = `record(nntile)+record(torch)+compile`.

| Setup | HF(cuda) wall | HF(nntile) wall | HF(nntile) / HF(cuda) | record(nntile) | record(torch) | compile | run | wait | host/wall | isolated |
|-------|----------:|------------:|------------:|---------------:|--------------:|--------:|----:|-----:|----------:|---------:|
| XS 32² | 0.645 ± 0.080 s | 0.698 ± 0.130 s | **1.08×** | 0.012 ± 0.001 s | 0.037 ± 0.002 s | 0.016 ± 0.001 s | 0.017 ± 0.001 s | 0.615 ± 0.132 s | **9.6%** | 0.046 s |
| S 48² | 1.064 ± 0.099 s | 1.133 ± 0.150 s | **1.07×** | 0.012 ± 0.001 s | 0.036 ± 0.003 s | 0.015 ± 0.001 s | 0.018 ± 0.001 s | 1.051 ± 0.149 s | **5.7%** | 0.090 ± 0.000 s |
| M 64² | 2.477 ± 0.155 s | 2.521 ± 0.193 s | **1.02×** | 0.012 ± 0.002 s | 0.038 ± 0.004 s | 0.015 ± 0.002 s | 0.018 ± 0.002 s | 2.436 ± 0.188 s | **2.6%** | 0.209 ± 0.001 s |
| L 80² | 5.285 ± 0.147 s | 5.369 ± 0.148 s | **1.02×** | 0.011 ± 0.002 s | 0.038 ± 0.004 s | 0.014 ± 0.002 s | 0.018 ± 0.002 s | 5.287 ± 0.141 s | **1.2%** | 0.502 ± 0.002 s |
| XL 144² | 18.047 ± 0.171 s | 18.284 ± 0.172 s | **1.01×** | 0.011 ± 0.001 s | 0.037 ± 0.003 s | 0.013 ± 0.001 s | 0.016 ± 0.002 s | 18.205 ± 0.169 s | **0.3%** | 1.791 ± 0.010 s |

## Sequential HF(nntile)

`--wait-after-run`: record → compile → run → wait (no overlap).

| Setup | HF(cuda) | HF(nntile) overlap | HF(nntile) sequential | seq / cuda |
|-------|-----:|---------:|----------:|----------:|
| XS 32² | 0.645 ± 0.080 s | 0.698 ± 0.130 s | 0.727 ± 0.109 s | **1.13×** |
| S 48² | 1.064 ± 0.099 s | 1.133 ± 0.150 s | 1.221 ± 0.174 s | **1.15×** |
| M 64² | 2.477 ± 0.155 s | 2.521 ± 0.193 s | 2.480 ± 0.187 s | **1.00×** |
| L 80² | 5.285 ± 0.147 s | 5.369 ± 0.148 s | 5.529 ± 0.112 s | **1.05×** |
| XL 144² | 18.047 ± 0.171 s | 18.284 ± 0.172 s | 18.454 ± 0.202 s | **1.02×** |

## S HF(nntile) 100-step

Overlap, size S, 100 steps, 10 repeats: wall 9.419 ± 0.184 s, loss 2.210866, host/wall 7.9%.

## How to reproduce

Probe **HF(nntile) VRAM** (all sizes, D2H must stay 0) before the
10-repeat ladder:

```bash
python3 torch_nntile/tools/probe_cnn_nntile_vram.py \
  --family vgg --logdir /tmp/vgg_vram --gpu 0 --steps 10
```

```bash
export TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export NNTILE_BUILD_DIR=$PWD/build TORCH_NNTILE_BUILD_DIR=$PWD/build
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:$PWD/build/nntile:$PWD/build/torch_nntile:${CONDA_PREFIX}/lib"
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
export STARPU_LIMIT_CUDA_MEM=46000

# Probe
python3 torch_nntile/tools/run_vgg_overhead_benchmark.py \
  --logdir /tmp/vgg_overhead_probe --gpu 0 --repeats 1 --sizes xs --skip-long

# Full ladder, 10 repeats, one idle GPU
python3 torch_nntile/tools/run_vgg_overhead_benchmark.py \
  --logdir /tmp/vgg_overhead --gpu 0 --repeats 10 --long-steps 100
```

Equivalent with the shared runner:
`python3 torch_nntile/tools/run_cnn_overhead_benchmark.py --family vgg ...`.
