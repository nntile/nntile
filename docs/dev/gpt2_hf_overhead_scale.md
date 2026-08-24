# GPT-2 HF: graph overhead vs width / seqlen

Ten-step stock HuggingFace GPT-2 on **CUDA** vs **`device=nntile`**.
Depth is **12 layers** everywhere. Width and sequence length grow
together with **`seq_len = n_embd / 2`**.

> **VRAM warning.** Nntile keeps extra graph buffers, so it uses
> **more GPU memory than CUDA** on the same model. If that footprint
> no longer fits in device memory, StarPU **moves data between CPU and
> GPU**. Those transfers dominate step time and make nntile look much
> slower than CUDA. Keep CUDA well under the card limit (this ladder
> peaks at ~28 GiB CUDA / ~40 GiB nntile on a 46 GiB A40) so nntile
> stays on-device.

Configs: [`torch_nntile/examples/overhead_gpt2/`](../../torch_nntile/examples/overhead_gpt2/).  
Script: [`train_gpt2_hf.py`](../../torch_nntile/examples/train_gpt2_hf.py).

## Train wall

Nntile:

1. Drain leftover work (`wait()`). GPU idle.
2. **Start timer** — *before* the first `record`.
3. Each step: `record → compile → wait(previous run) → run(submit)`.
4. After the last `run()`, **`wait()`**. **Stop timer.**
5. Loss `.item()` is **after** that join.

Logs print `elapsed after first record` (~20 ms). That time is inside
the wall.

CUDA: `synchronize`, start timer, 10 synced steps, stop after the last
synchronize. Prefetch is outside both walls.

Iter 1 nntile `wait=0` (no previous `run()`). Iter 10 `wait` includes
the final join (~2× a steady `wait`).

## Recipe

| | S | M | L |
|--|--:|--:|--:|
| Config | `gpt2_s.json` | `gpt2_m.json` | `gpt2_l.json` |
| `n_layer` | 12 | 12 | 12 |
| `n_embd` / `n_head` | 2048 / 16 | 3072 / 24 | 4096 / 32 |
| `--seq-len` (`= n_embd/2`) | **1024** | **1536** | **2048** |
| Params (FP32) | 611 M (2.44 GiB) | 1.37 B (5.49 GiB) | 2.44 B (9.74 GiB) |

B=1, 10 steps, seed 42, `--no-shuffle`, MATH SDPA, CUDA `--disable-tf32`,
nntile `--ncpu 0 --ncuda 1 --restrict-cuda`. NVIDIA A40, GPU 3. Separate
processes (never import `torch_nntile` in the CUDA child).

## Overall (10-step train wall)

Loss matches CUDA vs nntile to printed 1e-6 (L last digit 8.127417 vs
8.127419).

| Setup | CUDA wall | nntile wall | nntile/CUDA | record(nntile) | record(torch) | compile | run | wait | host/wall | peak VRAM CUDA / nntile |
|-------|----------:|------------:|------------:|---------------:|--------------:|--------:|----:|-----:|----------:|------------------------:|
| S T=1024 | 3.018 s | 3.127 s | **1.04×** | 0.062 s | 0.285 s | 0.161 s | 0.150 s | 2.468 s | **16.2%** | 7.2 / 8.7 GiB |
| M T=1536 | 8.517 s | 8.428 s | **0.99×** | 0.054 s | 0.271 s | 0.128 s | 0.139 s | 7.834 s | **5.4%** | 16.3 / 21.1 GiB |
| L T=2048 | 19.114 s | 18.598 s | **0.97×** | 0.052 s | 0.288 s | 0.129 s | 0.145 s | 17.983 s | **2.5%** | 28.2 / 40.4 GiB |

Host = `record(nntile)+record(torch)+compile` (~0.45–0.51 s for 10
steps, **flat**). Host **share** drops **16.2% → 5.4% → 2.5%** as
GPU work grows.

On this ladder CUDA stays ≤28 GiB, so nntile’s extra ~5–12 GiB still
fits on the 46 GiB card. Isolated GPU `wait` is then **slightly faster**
than CUDA (S 0.281 vs 0.283 s, M 0.810 vs 0.838 s, L 1.825 vs 1.904 s).
S’s 4% wall gap is leftover host on a short step (especially iter 1,
where `wait=0`). M and L hide that host behind `wait`.

## Per iteration

### S (`n_embd=2048`, `T=1024`)

| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.461 | 0.002 | 0.019 | 0.007 | 0.007 | 0.000 |
| 2 | 0.283 | 0.002 | 0.013 | 0.006 | 0.011 | 0.402 |
| 3 | 0.284 | 0.005 | 0.019 | 0.031 | 0.011 | 0.243 |
| 4 | 0.284 | 0.005 | 0.022 | 0.014 | 0.017 | 0.241 |
| 5 | 0.284 | 0.008 | 0.030 | 0.018 | 0.017 | 0.221 |
| 6 | 0.284 | 0.008 | 0.031 | 0.018 | 0.016 | 0.223 |
| 7 | 0.284 | 0.008 | 0.036 | 0.016 | 0.015 | 0.218 |
| 8 | 0.285 | 0.008 | 0.034 | 0.017 | 0.019 | 0.218 |
| 9 | 0.284 | 0.008 | 0.039 | 0.018 | 0.017 | 0.213 |
| 10 | 0.284 | 0.008 | 0.041 | 0.016 | 0.019 | 0.489 |

### M (`n_embd=3072`, `T=1536`)

| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 0.995 | 0.002 | 0.019 | 0.007 | 0.007 | 0.000 |
| 2 | 0.837 | 0.002 | 0.013 | 0.007 | 0.010 | 0.931 |
| 3 | 0.837 | 0.004 | 0.017 | 0.010 | 0.011 | 0.778 |
| 4 | 0.839 | 0.005 | 0.021 | 0.012 | 0.015 | 0.774 |
| 5 | 0.837 | 0.006 | 0.025 | 0.015 | 0.016 | 0.765 |
| 6 | 0.834 | 0.007 | 0.030 | 0.015 | 0.016 | 0.756 |
| 7 | 0.834 | 0.006 | 0.032 | 0.015 | 0.017 | 0.755 |
| 8 | 0.833 | 0.008 | 0.036 | 0.018 | 0.015 | 0.752 |
| 9 | 0.835 | 0.007 | 0.038 | 0.016 | 0.018 | 0.757 |
| 10 | 0.836 | 0.007 | 0.041 | 0.014 | 0.014 | 1.566 |

### L (`n_embd=4096`, `T=2048`)

| Iter | CUDA wall | record(nntile) | record(torch) | compile | run | wait |
|-----:|----------:|---------------:|--------------:|--------:|----:|-----:|
| 1 | 2.018 | 0.003 | 0.019 | 0.007 | 0.008 | 0.000 |
| 2 | 1.886 | 0.002 | 0.013 | 0.007 | 0.011 | 1.931 |
| 3 | 1.898 | 0.004 | 0.020 | 0.011 | 0.014 | 1.795 |
| 4 | 1.899 | 0.005 | 0.024 | 0.013 | 0.017 | 1.786 |
| 5 | 1.902 | 0.006 | 0.029 | 0.013 | 0.015 | 1.783 |
| 6 | 1.902 | 0.006 | 0.031 | 0.013 | 0.014 | 1.774 |
| 7 | 1.901 | 0.006 | 0.033 | 0.015 | 0.019 | 1.777 |
| 8 | 1.899 | 0.008 | 0.037 | 0.018 | 0.014 | 1.762 |
| 9 | 1.903 | 0.007 | 0.041 | 0.016 | 0.018 | 1.775 |
| 10 | 1.905 | 0.007 | 0.040 | 0.016 | 0.014 | 3.600 |

## Isolated extra step (after final loss, GPU idle)

Sequential nntile phases; no overlap with a previous `run()`.

| Setup | record(nntile) | record(torch) | compile | run | wait | run+wait | CUDA isolated |
|-------|---------------:|--------------:|--------:|----:|-----:|---------:|--------------:|
| S | 0.003 | 0.032 | 0.007 | 0.007 | 0.281 | **0.288** | 0.283 |
| M | 0.004 | 0.035 | 0.007 | 0.007 | 0.810 | **0.817** | 0.838 |
| L | 0.003 | 0.034 | 0.007 | 0.007 | 1.825 | **1.832** | 1.904 |

If record+compile run on the host while another job owns the GPU, the
remaining critical path is `run+wait`:

| Setup | Full isolated (record+compile+run+wait) | Hidden host (`run+wait`) | Saved |
|-------|----------------------------------------:|-------------------------:|------:|
| S | 0.337 s | 0.288 s | 0.049 s (**15%**) |
| M | 0.863 s | 0.817 s | 0.046 s (**5%**) |
| L | 1.876 s | 1.832 s | 0.044 s (**2%**) |

In the timed 10-step loop, record+compile of step \(k\) already overlap
GPU work of step \(k-1\).

## Takeaways

1. **`seq_len = n_embd / 2`**: S 1024, M 1536, L 2048.
2. **First record is in the wall** (~20 ms after `t0`).
3. **Host overhead is flat** (~50 ms/step). Share **16% → 5% → 2.5%**.
4. **With VRAM headroom, nntile matches or beats CUDA** on the train
   wall (S 1.04×, M 0.99×, L 0.97×).
5. **If nntile VRAM is too high**, StarPU pages GPU↔CPU and performance
   collapses. Size the job so CUDA is comfortably below the device
   limit (here L CUDA is 28 GiB on a 46 GiB A40).

## How to reproduce

```bash
# CUDA: do not put torch_nntile on PYTHONPATH
python3 -u torch_nntile/examples/train_gpt2_hf.py train \
  --device cuda --disable-tf32 --seed 42 --no-shuffle \
  --config torch_nntile/examples/overhead_gpt2/gpt2_s.json \
  --seq-len 1024 --batch-size 1 --max-sequences 10 --epochs 1 \
  --output-dir /tmp/gpt2_overhead_s_cuda

# nntile: PYTHONPATH=torch_nntile package, libnntile on LD_LIBRARY_PATH
python3 -u torch_nntile/examples/train_gpt2_hf.py train \
  --device nntile --restrict-cuda --ncpu 0 --ncuda 1 --seed 42 --no-shuffle \
  --config torch_nntile/examples/overhead_gpt2/gpt2_s.json \
  --seq-len 1024 --batch-size 1 --max-sequences 10 --epochs 1 \
  --output-dir /tmp/gpt2_overhead_s_nntile
```

M: `gpt2_m.json --seq-len 1536`. L: `gpt2_l.json --seq-len 2048`.
The nntile log must show `elapsed after first record` ~0.02 s.
