# U-Net HF: graph overhead vs width / spatial size

**Notation.** Each label is **implementation(backend)**.

- **HF** — stock PyTorch `torch.nn` CNN (`TinyUNet` in
  [`train_unet_tiny.py`](../../torch_nntile/examples/train_unet_tiny.py)).
  Not HuggingFace Transformers; the name matches the transformer overhead
  docs (implementation outside the brackets).
- **cuda** — PyTorch CUDA (`device=cuda`).
- **nntile** (as backend) — StarPU / nntile (`device=nntile`).

**There is no nntile(nntile) U-Net.** `torch_nntile.models` has no CNN
ports; this study is only **HF(cuda)** vs **HF(nntile)**.

Two setups, same configs / 10 steps:

1. **HF(cuda)** — stock `TinyUNet`, `device=cuda`, no `torch_nntile` import.
2. **HF(nntile)** — same graph on `device=nntile` (aten / torch-native
   StarPU codelets: encoder / decoder, skip `cat`, `ConvTranspose2d` via
   `convolution_overrideable`, pixel CE flattened to 1D `nll_loss`).

> **VRAM warning.** Skip concatenations plus nntile graph buffers grow
> faster than classification CNNs. Keep HF(cuda) well under the card
> limit so `device=nntile` stays on-device. Depth *d* needs
> `height` / `width` divisible by `2^d`. If logs show D2H volume, shrink
> `base_channels` or spatial size. GPUs are in exclusive mode — one
> process per GPU.

Configs: [`torch_nntile/examples/overhead_unet/`](../../torch_nntile/examples/overhead_unet/).
HF(cuda) / HF(nntile):
[`train_cnn_hf_overhead.py`](../../torch_nntile/examples/train_cnn_hf_overhead.py)
(`--model unet`),
[`run_unet_overhead_benchmark.py`](../../torch_nntile/tools/run_unet_overhead_benchmark.py).

## Loss

Pixel-wise CE: NCHW logits vs NHW labels, flattened to 1D (same as the
tiny U-Net smoke; `nll_loss2d` is not registered on PrivateUse1). New
batch seed per step: `42 + step`.

## Train wall

Same recipe as
[`gpt2_hf_overhead_scale.md`](gpt2_hf_overhead_scale.md): nntile
`record → compile → wait(prev) → run`, wall from first record through
final `wait()`; HF(cuda) synced per iter. Prefetch outside the wall.
Iter 1 nntile `wait=0`; iter 10 `wait` includes the final join.

## Recipe

Spatial size grows with `base_channels`. **XL** uses **depth=3** and
192² (depth analog) so HF(nntile) fills an A40; XS–L stay at depth 4.

| | XS | S | M | L | XL |
|--|--:|--:|--:|--:|--:|
| Config | `unet_xs.json` | `unet_s.json` | `unet_m.json` | `unet_l.json` | `unet_xl.json` |
| `depth` | 4 | 4 | 4 | 4 | **3** |
| `base_channels` | 256 | 384 | 480 | 576 | 1280 |
| `height` × `width` | 32² | 48² | 64² | 80² | **192²** |
| Params (FP32) | 496 M (1.85 GiB) | 1.12 B (4.16 GiB) | 1.75 B (6.50 GiB) | 2.51 B (9.36 GiB) | 3.08 B (11.46 GiB) |

B=1, 10 steps, seed 42, `--no-shuffle`, HF(cuda) and HF(nntile)
`--disable-tf32 --disable-cudnn`, `device=nntile` `--ncpu 0 --ncuda 1
--restrict-cuda`. NVIDIA A40, one GPU per job. Separate processes
(`PYTHONNOUSERSITE=1`; never import `torch_nntile` in the HF(cuda)
process). **Do not overlap jobs on one GPU.**

HF(cuda) / HF(nntile): **10 repeats** (mean ± stdev), including **S HF(nntile) 100-step**.
`STARPU_LIMIT_CUDA_MEM=46000`.

## Two setups

### Loss

| Setup | HF(cuda) | HF(nntile) |
|-------|-----:|----------------:|
| XS 32² | 1.141706 | 1.141706 |
| S 48² | 1.132266 | 1.132266 |
| M 64² | 1.128510 | 1.128510 |
| L 80² | 1.127552 | 1.127552 |
| XL 192² | 1.123813 | 1.123813 |

### 10-step train wall

**10 repeats** (mean ± stdev), `STARPU_LIMIT_CUDA_MEM=46000`.
XS is a dedicated `--sizes xs --skip-long` 10-repeat (same flags); S–XL
are from the full ladder. The 10-step wall includes first-kernel launch
(HF(cuda) iter 1 ~0.20–0.61 s vs iters 2–10 **0.034 s**).

| Setup | HF(cuda) | HF(nntile) | HF(nntile) / HF(cuda) |
|-------|-----:|---------:|--------------:|
| XS 32² | 0.658 ± 0.166 s | 0.709 ± 0.172 s | **1.08×** |
| S 48² | 1.251 ± 0.173 s | 1.185 ± 0.167 s | **0.95×** |
| M 64² | 1.713 ± 0.152 s | 1.744 ± 0.144 s | **1.02×** |
| L 80² | 2.640 ± 0.132 s | 2.642 ± 0.024 s | **1.00×** |
| XL 192² | 25.924 ± 0.092 s | 26.512 ± 0.084 s | **1.02×** |

### Peak VRAM and bus

10-step overlap, NVIDIA A40, `STARPU_LIMIT_CUDA_MEM=46000`, B=1. Peak VRAM is `nvidia-smi memory.used` polled by a sidecar process (`watch_gpu_peak.py`) while the train child runs (`peak_vram_gib=`). H2D/D2H are StarPU bus stats at shutdown from the VRAM-fit probe. **D2H is 0 on every size**.

| Setup | HF(cuda) VRAM | HF(nntile) VRAM | H2D | D2H |
|-------|----------:|----------------:|----:|----:|
| XS 32² | 4.1 GiB | 5.0 GiB | 1.85 GB | **0** |
| S 48² | 8.8 GiB | 10.9 GiB | 4.16 GB | **0** |
| M 64² | 13.7 GiB | 16.8 GiB | 6.50 GB | **0** |
| L 80² | 19.5 GiB | 24.3 GiB | 9.36 GB | **0** |
| XL 192² | 28.9 GiB | 37.3 GiB | 11.47 GB | **0** |

## HF(nntile) vs HF(cuda) (10 repeats)

Overlap mode. Host = `record(nntile)+record(torch)+compile`.

| Setup | HF(cuda) wall | HF(nntile) wall | HF(nntile) / HF(cuda) | record(nntile) | record(torch) | compile | run | wait | host/wall | isolated |
|-------|----------:|------------:|------------:|---------------:|--------------:|--------:|----:|-----:|----------:|---------:|
| XS 32² | 0.658 ± 0.166 s | 0.709 ± 0.172 s | **1.08×** | 0.031 ± 0.002 s | 0.092 ± 0.003 s | 0.057 ± 0.004 s | 0.056 ± 0.002 s | 0.473 ± 0.172 s | **26.5%** | 0.044 ± 0.001 s |
| S 48² | 1.251 ± 0.173 s | 1.185 ± 0.167 s | **0.95×** | 0.029 ± 0.002 s | 0.093 ± 0.006 s | 0.061 ± 0.003 s | 0.060 ± 0.004 s | 0.940 ± 0.167 s | **15.7%** | 0.091 ± 0.001 s |
| M 64² | 1.713 ± 0.152 s | 1.744 ± 0.144 s | **1.02×** | 0.028 ± 0.002 s | 0.090 ± 0.004 s | 0.056 ± 0.005 s | 0.058 ± 0.004 s | 1.511 ± 0.143 s | **10.0%** | 0.150 ± 0.001 s |
| L 80² | 2.640 ± 0.132 s | 2.642 ± 0.024 s | **1.00×** | 0.029 ± 0.002 s | 0.091 ± 0.004 s | 0.055 ± 0.004 s | 0.060 ± 0.004 s | 2.406 ± 0.027 s | **6.6%** | 0.247 ± 0.000 s |
| XL 192² | 25.924 ± 0.092 s | 26.512 ± 0.084 s | **1.02×** | 0.025 ± 0.001 s | 0.089 ± 0.003 s | 0.044 ± 0.004 s | 0.051 ± 0.003 s | 26.301 ± 0.084 s | **0.6%** | 2.591 ± 0.010 s |

## Sequential HF(nntile)

`--wait-after-run`: record → compile → run → wait (no overlap).
XS sequential was remeasured with the dedicated 10-repeat. The old
**1.48×** was first-kernel launch plus serialized host prep (~0.18 s
record+compile that overlap mode hides under `wait`). Isolated extra
step (GPU idle): HF(cuda) **0.034 s**, HF(nntile) sequential run+wait
**0.042 s**.

| Setup | HF(cuda) | HF(nntile) overlap | HF(nntile) sequential | seq / cuda |
|-------|-----:|---------:|----------:|----------:|
| XS 32² | 0.658 ± 0.166 s | 0.709 ± 0.172 s | 0.789 ± 0.124 s | **1.20×** |
| S 48² | 1.251 ± 0.173 s | 1.185 ± 0.167 s | 1.233 ± 0.009 s | **0.99×** |
| M 64² | 1.713 ± 0.152 s | 1.744 ± 0.144 s | 1.831 ± 0.038 s | **1.07×** |
| L 80² | 2.640 ± 0.132 s | 2.642 ± 0.024 s | 2.889 ± 0.174 s | **1.09×** |
| XL 192² | 25.924 ± 0.092 s | 26.512 ± 0.084 s | 26.719 ± 0.096 s | **1.03×** |

## S HF(nntile) 100-step

Overlap, size S, 100 steps, 10 repeats: wall 9.676 ± 0.195 s, loss 1.123617, host/wall 23.0%.

## How to reproduce

Probe **HF(nntile) VRAM** (all sizes, D2H must stay 0) before the
10-repeat ladder:

```bash
python3 torch_nntile/tools/probe_cnn_nntile_vram.py \
  --family unet --logdir /tmp/unet_vram --gpu 0 --steps 10
```

```bash
export TORCH_LIB_DIR="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export NNTILE_BUILD_DIR=$PWD/build TORCH_NNTILE_BUILD_DIR=$PWD/build
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB_DIR}:$PWD/build/nntile:$PWD/build/torch_nntile:${CONDA_PREFIX}/lib"
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
export STARPU_LIMIT_CUDA_MEM=46000

# Probe
python3 torch_nntile/tools/run_unet_overhead_benchmark.py \
  --logdir /tmp/unet_overhead_probe --gpu 0 --repeats 1 --sizes xs --skip-long

# Full ladder, 10 repeats, one idle GPU
python3 torch_nntile/tools/run_unet_overhead_benchmark.py \
  --logdir /tmp/unet_overhead --gpu 0 --repeats 10 --long-steps 100

# XS-only 10-repeat (first-launch vs isolated-step / sequential check)
python3 torch_nntile/tools/run_unet_overhead_benchmark.py \
  --logdir /tmp/unet_xs --gpu 0 --repeats 10 --sizes xs --skip-long
```

Equivalent with the shared runner:
`python3 torch_nntile/tools/run_cnn_overhead_benchmark.py --family unet ...`.
