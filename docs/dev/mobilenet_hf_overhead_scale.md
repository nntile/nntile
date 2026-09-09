# MobileNet HF: graph overhead vs width / spatial size

**Notation.** Each label is **implementation(backend)**.

- **HF** — stock PyTorch `torch.nn` CNN (`TinyMobileNet` in
  [`train_mobilenet_tiny.py`](../../torch_nntile/examples/train_mobilenet_tiny.py)).
  Not HuggingFace Transformers; the name matches the transformer overhead
  docs (implementation outside the brackets).
- **cuda** — PyTorch CUDA (`device=cuda`).
- **nntile** (as backend) — StarPU / nntile (`device=nntile`).

**There is no nntile(nntile) MobileNet.** `torch_nntile.models` has no CNN
ports; this study is only **HF(cuda)** vs **HF(nntile)**.

Two setups, same configs / 10 steps:

1. **HF(cuda)** — stock `TinyMobileNet`, `device=cuda`, no `torch_nntile`
   import.
2. **HF(nntile)** — same graph on `device=nntile` (aten / torch-native
   StarPU codelets: depthwise `groups=C` + pointwise 1×1
   `convolution_overrideable`, `native_batch_norm`, AdaptiveAvgPool2d).

> **VRAM warning.** Nntile keeps extra graph buffers. Keep HF(cuda) well
> under the card limit so `device=nntile` stays on-device (no StarPU
> CPU↔GPU paging). If logs show D2H volume, shrink `base_channels` or
> spatial size before collecting 10-repeat walls. GPUs are in exclusive
> mode — one process per GPU.

Configs: [`torch_nntile/examples/overhead_mobilenet/`](../../torch_nntile/examples/overhead_mobilenet/).
HF(cuda) / HF(nntile):
[`train_cnn_hf_overhead.py`](../../torch_nntile/examples/train_cnn_hf_overhead.py)
(`--model mobilenet`),
[`run_mobilenet_overhead_benchmark.py`](../../torch_nntile/tools/run_mobilenet_overhead_benchmark.py).

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

Spatial size grows with `base_channels`. Param count stays lower than
ResNet/VGG (depthwise). **XL** uses **3 blocks** and 128² (depth analog)
so HF(nntile) fills an A40; XS–L stay at 4 blocks.

| | XS | S | M | L | XL |
|--|--:|--:|--:|--:|--:|
| Config | `mobilenet_xs.json` | `mobilenet_s.json` | `mobilenet_m.json` | `mobilenet_l.json` | `mobilenet_xl.json` |
| `blocks` | 4 | 4 | 4 | 4 | **3** |
| `base_channels` | 6400 | 9216 | 12000 | 14400 | 17280 |
| `height` × `width` | 32² | 48² | 64² | 80² | **128²** |
| Params (FP32) | 451 M (1.68 GiB) | 936 M (3.48 GiB) | 1.59 B (5.91 GiB) | 2.28 B (8.50 GiB) | 2.09 B (7.79 GiB) |

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
| XS 32² | 21.285589 | 21.285589 |
| S 48² | 30.524199 | 30.524199 |
| M 64² | 37.306808 | 37.306808 |
| L 80² | 45.432007 | 45.432007 |
| XL 128² | 55.193851 | 55.193851 |

### 10-step train wall

**10 repeats** (mean ± stdev), `STARPU_LIMIT_CUDA_MEM=46000`.

| Setup | HF(cuda) | HF(nntile) | HF(nntile) / HF(cuda) |
|-------|-----:|---------:|--------------:|
| XS 32² | 0.818 ± 0.158 s | 0.783 ± 0.250 s | **0.96×** |
| S 48² | 1.788 ± 0.093 s | 1.907 ± 0.161 s | **1.07×** |
| M 64² | 4.033 ± 0.143 s | 4.162 ± 0.186 s | **1.03×** |
| L 80² | 8.575 ± 0.157 s | 8.779 ± 0.114 s | **1.02×** |
| XL 128² | 26.792 ± 0.280 s | 27.578 ± 0.295 s | **1.03×** |

### Peak VRAM and bus

10-step overlap, NVIDIA A40, `STARPU_LIMIT_CUDA_MEM=46000`, B=1. Peak VRAM is `nvidia-smi memory.used` polled by a sidecar process (`watch_gpu_peak.py`) while the train child runs (`peak_vram_gib=`). H2D/D2H are StarPU bus stats at shutdown from the VRAM-fit probe. **D2H is 0 on every size**.

| Setup | HF(cuda) VRAM | HF(nntile) VRAM | H2D | D2H |
|-------|----------:|----------------:|----:|----:|
| XS 32² | 4.0 GiB | 4.6 GiB | 1.68 GB | **0** |
| S 48² | 8.2 GiB | 9.3 GiB | 3.49 GB | **0** |
| M 64² | 14.1 GiB | 16.2 GiB | 5.91 GB | **0** |
| L 80² | 20.8 GiB | 24.2 GiB | 8.51 GB | **0** |
| XL 128² | 26.5 GiB | 37.2 GiB | 7.80 GB | **0** |

## HF(nntile) vs HF(cuda) (10 repeats)

Overlap mode. Host = `record(nntile)+record(torch)+compile`.

| Setup | HF(cuda) wall | HF(nntile) wall | HF(nntile) / HF(cuda) | record(nntile) | record(torch) | compile | run | wait | host/wall | isolated |
|-------|----------:|------------:|------------:|---------------:|--------------:|--------:|----:|-----:|----------:|---------:|
| XS 32² | 0.818 ± 0.158 s | 0.783 ± 0.250 s | **0.96×** | 0.010 ± 0.001 s | 0.045 ± 0.002 s | 0.023 ± 0.002 s | 0.025 ± 0.002 s | 0.679 ± 0.250 s | **10.6%** | 0.050 ± 0.000 s |
| S 48² | 1.788 ± 0.093 s | 1.907 ± 0.161 s | **1.07×** | 0.010 ± 0.002 s | 0.046 ± 0.005 s | 0.022 ± 0.004 s | 0.026 ± 0.003 s | 1.802 ± 0.160 s | **4.1%** | 0.164 ± 0.001 s |
| M 64² | 4.033 ± 0.143 s | 4.162 ± 0.186 s | **1.03×** | 0.010 ± 0.001 s | 0.051 ± 0.005 s | 0.024 ± 0.002 s | 0.028 ± 0.003 s | 4.047 ± 0.191 s | **2.1%** | 0.386 ± 0.003 s |
| L 80² | 8.575 ± 0.157 s | 8.779 ± 0.114 s | **1.02×** | 0.011 ± 0.000 s | 0.051 ± 0.005 s | 0.024 ± 0.002 s | 0.028 ± 0.002 s | 8.664 ± 0.113 s | **1.0%** | 0.841 ± 0.005 s |
| XL 128² | 26.792 ± 0.280 s | 27.578 ± 0.295 s | **1.03×** | 0.008 ± 0.001 s | 0.049 ± 0.004 s | 0.020 ± 0.001 s | 0.024 ± 0.002 s | 27.475 ± 0.297 s | **0.3%** | 2.721 ± 0.018 s |

## Sequential HF(nntile)

`--wait-after-run`: record → compile → run → wait (no overlap).

| Setup | HF(cuda) | HF(nntile) overlap | HF(nntile) sequential | seq / cuda |
|-------|-----:|---------:|----------:|----------:|
| XS 32² | 0.818 ± 0.158 s | 0.783 ± 0.250 s | 0.816 ± 0.168 s | **1.00×** |
| S 48² | 1.788 ± 0.093 s | 1.907 ± 0.161 s | 1.928 ± 0.119 s | **1.08×** |
| M 64² | 4.033 ± 0.143 s | 4.162 ± 0.186 s | 4.329 ± 0.214 s | **1.07×** |
| L 80² | 8.575 ± 0.157 s | 8.779 ± 0.114 s | 8.776 ± 0.114 s | **1.02×** |
| XL 128² | 26.792 ± 0.280 s | 27.578 ± 0.295 s | 27.621 ± 0.277 s | **1.03×** |

## S HF(nntile) 100-step

Overlap, size S, 100 steps, 10 repeats: wall 17.027 ± 0.155 s, loss 28.558474, host/wall 5.5%.

## How to reproduce

Probe **HF(nntile) VRAM** (all sizes, D2H must stay 0) before the
10-repeat ladder:

```bash
python3 torch_nntile/tools/probe_cnn_nntile_vram.py \
  --family mobilenet --logdir /tmp/mobilenet_vram --gpu 0 --steps 10
```

```bash
export TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export NNTILE_BUILD_DIR=$PWD/build TORCH_NNTILE_BUILD_DIR=$PWD/build
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:$PWD/build/nntile:$PWD/build/torch_nntile:${CONDA_PREFIX}/lib"
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
export STARPU_LIMIT_CUDA_MEM=46000

# Probe
python3 torch_nntile/tools/run_mobilenet_overhead_benchmark.py \
  --logdir /tmp/mobilenet_overhead_probe --gpu 0 --repeats 1 --sizes xs --skip-long

# Full ladder, 10 repeats, one idle GPU
python3 torch_nntile/tools/run_mobilenet_overhead_benchmark.py \
  --logdir /tmp/mobilenet_overhead --gpu 0 --repeats 10 --long-steps 100
```

Equivalent with the shared runner:
`python3 torch_nntile/tools/run_cnn_overhead_benchmark.py --family mobilenet ...`.
