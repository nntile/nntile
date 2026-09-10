# CUDA vs `device=nntile`: ≥2 GiB GPU comparison

Recipe for comparing **stock** HuggingFace / CNN / Diffusers DiT training
on plain PyTorch CUDA versus `device=nntile` (StarPU CUDA worker).
This is **not** a `torch_nntile.models.*` benchmark.

Configs: [`torch_nntile/examples/2gb/`](../../torch_nntile/examples/2gb/)
(≥2 GiB FP32 weights each).  
CUDA helper: [`train_cuda_only.py`](../../torch_nntile/examples/train_cuda_only.py)  
Orchestrator: [`bench_cuda_vs_nntile_2gb.py`](../../torch_nntile/examples/bench_cuda_vs_nntile_2gb.py)

Related: GPT-2-only shell
[`run_gpt2_hf_cuda_vs_nntile.sh`](../../torch_nntile/examples/run_gpt2_hf_cuda_vs_nntile.sh),
CPU vs nntile
[hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md),
protocol [reproducibility.md](reproducibility.md).

## Hard constraints (read before running)

1. **Two processes.** PyTorch cannot use CUDA autograd and PrivateUse1
   `nntile` in one process (PyTorch ≥ 2.8). Never import
   `torch_nntile` in the CUDA child. Do **not** add `--device cuda` to
   `hf_tiny_train_common.py` / `cnn_tiny_train_common.py` /
   `dit_hf_tiny_train_common.py`.
2. **Dedicated GPU.** Shared-server jobs on the same device distort
   wall time and `nvidia-smi` VRAM. Run `nvidia-smi`, pick a GPU with
   ~idle memory and 0% util, then pin it.
3. **Stock models only.** Do not touch `torch_nntile.models.*` or the
   C++ native model bindings. `cpu_fallback` stays **False**.
4. **No implicit host copy on nntile ops.** ATen `from_blob` inside a
   StarPU codelet is OK. Do not implement a `device=nntile` op as
   “run CPU Torch then copy onto nntile”.
5. **Run Python from `/tmp` (the bench already does this)** so a
   checkout of `torch_nntile/` does not shadow the package.

## What is measured

| Field | Meaning |
|-------|---------|
| **Loss** | Last printed `loss=` (synthetic batch, seed 42, SGD). |
| **Wall** | Train loop only (`wall=…s` or GPT-2 `timing … train wall`). Excludes import, HF construct, StarPU `init_context`, prefetch. CUDA is the training loop plus device synchronize. On nntile it is a single clock from **before the first record** through the **final `wait()`** (every record, compile, wait, and run). Loss readout is after that join. |
| **record / compile / run / wait** | Cumulative nntile phases: record each step, `compile_graph` each step, **`wait()` for the previous `run()`**, then `run()` the compiled step, plus a final `wait()`. |
| **VRAM** | `nvidia-smi` peak during the child minus idle-before. Includes leftover CUDA context (~300–500 MiB). |

On nntile each step is recorded and compiled while the previous
``run()`` is in flight; ``wait()`` then joins that submit before
``run()`` of the compiled step. A final ``wait()`` joins the last
submit. CUDA uses `--disable-tf32` (full
FP32) for a fair numeric compare. Attention on both CUDA and nntile
is MATH SDPA (debt D8).

The same synthetic batch is reused every step, so CNN losses can
collapse toward 0 by step 50. That is expected; compare CUDA vs
nntile, not absolute CNN loss.

## How to run (Cursor agent / another server)

### 0. Pick a free GPU

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv
```

Use a device with memory ≈ idle and util 0%. Example pin:

```bash
export CUDA_VISIBLE_DEVICES=0   # replace with the idle index
```

### 1. CUDA build of NNTile + torch_nntile

Need `torch==2.9.1` (not 2.12), StarPU, and `-DUSE_CUDA=ON`.
Adjust compilers / StarPU prefix to the host.

```bash
# from repo root
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig   # or your StarPU
TORCH_PREFIX=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON \
  -DBUILD_TESTING=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -GNinja
cmake --build build --target nntile torch_nntile -j$(nproc)

export NNTILE_BUILD_DIR=$PWD/build
export TORCH_NNTILE_BUILD_DIR=$PWD/build
export STARPU_LIB=/opt/starpu/lib   # if StarPU is elsewhere, set this
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
```

Python deps used by the smokes: `transformers<4.53`, `diffusers`
(0.32.x is fine), `datasets`.

### 2. Launch the 2 GiB suite

Defaults match the table below: **50 steps**, HF **batch 16 / seq 32**,
CNN/DiT **batch 4**, seed **42**, nntile `--ncpu 0 --ncuda 1
--restrict-cuda`.

```bash
# still from repo root; GPU from CUDA_VISIBLE_DEVICES or --gpu
python3 -u torch_nntile/examples/bench_cuda_vs_nntile_2gb.py --gpu "${CUDA_VISIBLE_DEVICES}" \
  2>&1 | tee /tmp/cuda_vs_nntile_2gb.log
```

Optional knobs:

```bash
python3 -u torch_nntile/examples/bench_cuda_vs_nntile_2gb.py \
  --gpu 0 \
  --steps 50 \
  --hf-batch-size 16 --seq-len 32 --cnn-batch-size 4 \
  --build-dir "$PWD/build" \
  --output-root /tmp/cuda_vs_nntile_2gb_ckpts
```

Wall clock on one A40 was ~12 minutes for the full 15-model suite
(idle GPU, per-iter compile). Expect longer if the GPU is slower or
contended.

### 3. Single-model debug (optional)

CUDA child (no libnntile on `LD_LIBRARY_PATH`, `PYTHONPATH` = examples):

```bash
cd /tmp
PYTHONPATH=/path/to/nntile/torch_nntile/examples \
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python3 /path/to/nntile/torch_nntile/examples/train_cuda_only.py \
  --model gpt-neox --config /path/to/nntile/torch_nntile/examples/2gb/gpt_neox.json \
  --steps 2 --batch-size 16 --seq-len 32 --seed 42
```

Nntile child:

```bash
cd /tmp
export PYTHONPATH=/path/to/nntile/torch_nntile
export LD_LIBRARY_PATH=/path/to/nntile/build/nntile:/path/to/nntile/build/torch_nntile:${STARPU_LIB}:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python3 /path/to/nntile/torch_nntile/examples/train_gpt_neox_hf.py train \
  --device nntile --seed 42 --steps 2 --seq-len 32 --batch-size 16 \
  --config /path/to/nntile/torch_nntile/examples/2gb/gpt_neox.json \
  --ncpu 0 --ncuda 1 --restrict-cuda
```

GPT-2 CUDA uses `train_gpt2_hf.py train --device cuda --disable-tf32`
(it imports `torch_nntile` only on the nntile branch). GPT-2 step count
is `--max-sequences = steps * batch-size` with `--epochs 1 --no-shuffle`.

## Configs (`examples/2gb/`)

Typical transformer: hidden 1536, ~18–20 layers, vocab 2048, max pos 128.
T5 is 10+10 layers. CNNs are small spatial size with fat channels
(MobileNet `base_channels=7200` at 64² is the VRAM outlier).

| File | Train script (nntile) | CUDA `--model` |
|------|------------------------|----------------|
| `gpt2.json` | `train_gpt2_hf.py` | (GPT-2 script, both devices) |
| `gpt_neo.json` | `train_gpt_neo_hf.py` | `gpt-neo` |
| `gpt_neox.json` | `train_gpt_neox_hf.py` | `gpt-neox` |
| `llama.json` | `train_llama_hf.py` | `llama` |
| `llama_gqa.json` | `train_llama_hf.py` | `llama-gqa` |
| `bert.json` | `train_bert_hf.py` | `bert` |
| `roberta.json` | `train_roberta_hf.py` | `roberta` |
| `t5.json` | `train_t5_hf.py` | `t5` |
| `lenet.json` | `train_lenet_tiny.py` | `lenet` |
| `resnet.json` | `train_resnet_tiny.py` | `resnet` |
| `vgg.json` | `train_vgg_tiny.py` | `vgg` |
| `mobilenet.json` | `train_mobilenet_tiny.py` | `mobilenet` |
| `unet.json` | `train_unet_tiny.py` | `unet` |
| `unet_modern.json` | `train_unet_modern_tiny.py` | `unet-modern` |
| `dit.json` | `train_dit_hf.py` | `dit` |

## Recorded results

NVIDIA A40 (46 GiB), branch `nntile-no-implicit-host-copy`,
`NNTILE_TORCH_NATIVE_OPS` CUDA build, MATH SDPA (debt D8).

| Id | When | GPU | Nntile loop |
|----|------|-----|-------------|
| **D** | 2026-08-21 16:35 UTC | 1 (idle) | Per-iter record / compile / `run`; **one `wait()` at the end** |
| A / B / C | earlier 2026-08-21 | see below | Older loop (compile-once or wait-per-step); walls not comparable to D |

### 2 GiB, 50 steps (run D, this recipe)

HF batch 16, seq 32; CNN/DiT batch 4; seed 42. Idle GPU 1.

**Nntile wall** = sum of `run()` + the final `wait()` (GPU submit +
join). **Nntile total** = record + compile + run + wait (host graph
work plus GPU). Compare **CUDA wall** to **nntile total**, not to
nntile wall alone.

| Model | CUDA loss | nntile loss | CUDA VRAM | nntile VRAM | CUDA wall | nntile record | nntile compile | nntile run | nntile wait | nntile wall | nntile total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 HF | 7.760632 | 7.760633 | 6450 MiB | 6666 MiB | 7.196 s | 2.606 s | 2.341 s | 0.745 s | 4.051 s | 4.797 s | 9.743 s |
| GPT-Neo HF | 2.635335 | 2.631112 | 5920 MiB | 6302 MiB | 7.251 s | 3.350 s | 2.837 s | 0.800 s | 2.299 s | 3.099 s | 9.286 s |
| GPT-NeoX HF | 6.450442 | 6.450443 | 5582 MiB | 12854 MiB | 6.436 s | 3.772 s | 4.717 s | 1.122 s | 1.788 s | 2.910 s | 11.399 s |
| Llama HF | 2.314239 | 2.314239 | 6522 MiB | 15144 MiB | 7.909 s | 4.415 s | 6.005 s | 1.496 s | 2.164 s | 3.660 s | 14.080 s |
| Llama HF GQA | 2.167546 | 2.167546 | 6288 MiB | 12818 MiB | 7.320 s | 3.745 s | 4.970 s | 1.154 s | 2.789 s | 3.942 s | 12.658 s |
| BERT HF | 5.385140 | 5.635214 | 5198 MiB | 6408 MiB | 6.330 s | 3.058 s | 2.526 s | 0.803 s | 2.874 s | 3.678 s | 9.261 s |
| RoBERTa HF | 5.267773 | 5.260685 | 5198 MiB | 6632 MiB | 6.336 s | 2.564 s | 2.465 s | 0.767 s | 4.500 s | 5.267 s | 10.296 s |
| T5 HF | 7.660880 | 7.660872 | 5862 MiB | 7740 MiB | 8.136 s | 4.252 s | 4.919 s | 1.121 s | 2.235 s | 3.357 s | 12.527 s |
| LeNet | 0.015324 | 0.015325 | 4750 MiB | 6694 MiB | 1.585 s | 0.084 s | 0.038 s | 0.024 s | 2.632 s | 2.656 s | 2.778 s |
| ResNet | 0.000001 | 0.000099 | 6506 MiB | 5336 MiB | 30.652 s | 0.393 s | 0.266 s | 0.145 s | 17.713 s | 17.859 s | 18.517 s |
| VGG | 2.202861 | 2.202862 | 8138 MiB | 7598 MiB | 6.492 s | 0.163 s | 0.087 s | 0.050 s | 4.108 s | 4.158 s | 4.408 s |
| MobileNet | 0.005540 | 0.005539 | 9682 MiB | 13568 MiB | 42.687 s | 0.193 s | 0.121 s | 0.069 s | 23.149 s | 23.218 s | 23.532 s |
| UNet | 0.432281 | 0.432325 | 6822 MiB | 7782 MiB | 4.742 s | 0.512 s | 0.345 s | 0.175 s | 3.333 s | 3.508 s | 4.365 s |
| UNet modern | 0.465618 | 0.465730 | 8136 MiB | 9760 MiB | 5.757 s | 0.480 s | 0.324 s | 0.172 s | 3.822 s | 3.994 s | 4.798 s |
| DiT HF | 0.602640 | 0.602640 | 7136 MiB | 8506 MiB | 4.636 s | 4.362 s | 4.619 s | 1.267 s | 0.782 s | 2.049 s | 11.030 s |

**Loss:** GPT-2, NeoX, Llama, Llama GQA, T5, DiT match CUDA vs nntile
to printed ~1e-6. VGG / LeNet / MobileNet / UNet are within ~1e-4.
**BERT is a known outlier** (eager CE / `ignore_index` path). GPT-Neo
and RoBERTa are slightly off (~4e-3 and ~7e-3). CNN losses near zero
are repeated-batch overfit; ResNet last-step loss is not stable across
repeats (still ≪1).

**Speed:** transformer **nntile total** is ~1.3–1.8× CUDA wall; host
record+compile is a large share (Llama compile 6.0 s). CNN **nntile
total** beats CUDA on ResNet / VGG / MobileNet (same as earlier runs).
Older tables that labeled nntile wall as run+wait only omitted that
host time; the current train wall includes record and compile.

**Memory (run D, wait only at the end):** that loop lets StarPU submit
many `STARPU_W` destination clears ahead of gemms (debt D7). Llama /
NeoX / Llama GQA nntile VRAM is ~2× CUDA (~12–15 GiB vs ~6 GiB).
GPT-2 stays close (6666 vs 6450 MiB). MobileNet remains the CNN VRAM
outlier (~13.6 GiB nntile vs ~9.7 GiB CUDA).

**Why Llama (and NeoX / GQA), not GPT-2:** CUDA Llama is **flat**
in step count (6522 MiB at 1 and 50 steps). Nntile Llama is already
heavier at **one** step (8140 vs 6522 MiB) because the HF graph
materializes extras CUDA’s allocator reuses: SwiGLU
(`gate` / `silu` / `up` / product vs GPT-2’s two GEMMs), split
Q/K/V, and RoPE (`rotate_half`, mul/add) as TensorNodes.

With **wait only after all `run()`s**, VRAM then **grows with
submitted steps** (D7):

| Steps | Llama CUDA | Llama nntile (wait at end) | GPT-2 nntile |
|------:|-----------:|---------------------------:|-------------:|
| 1 | 6522 | 8140 | 6250 |
| 10 | 6522 | 11020 | — |
| 50 | 6522 | 12644 | 6570 |
| 50 + `STARPU_LIMIT_{MAX=100,MIN=50}` | — | 7824 | — |

The current loop is **record → compile → wait → run** (wait joins
the previous `run()`; record/compile of step \(N\) overlap that
GPU work). Llama VRAM is then **flat** in step count; loss still
matches CUDA (2.314239 at 50 steps):

| Steps | Llama CUDA | Llama nntile (wait before run) | GPT-2 nntile |
|------:|-----------:|-------------------------------:|-------------:|
| 1 | 6522 | 8140 | — |
| 10 | 6522 | 8152 | — |
| 50 | 6522 | 8152 | 6282 |

The leftover ~1.6 GiB vs CUDA is the fatter per-step graph, not
in-flight steps. GPT-2’s fused `c_attn` / two-GEMM MLP emit far
fewer `STARPU_W` clears, so even wait-at-end added only ~0.3 GiB.

### Earlier 50-step runs (A / B / C)

Same 2 GiB recipe, older nntile train loop (not per-iter compile with
a single final wait). Transformer losses matched across A/B/C to
printed 1e-6. Treat walls on **A** as possibly contended; **B** and
**C** are idle GPU 0. VRAM is B/C (identical); A is ~4 MiB lower.

| Id | GPU | Notes |
|----|-----|-------|
| **A** | 2 | Original recording, shared box |
| **B** | 0 | Idle; `NNTILE_TORCH_NATIVE_OPS` CUDA build |
| **C** | 0 | Idle repeat of B (same binary, same GPU) |

| Model | CUDA loss | nntile loss | CUDA VRAM | nntile VRAM | CUDA wall A / B / C | nntile wall A / B / C |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 HF | 7.795432 | 7.795432 | 6420 MiB | 6220 MiB | 7.883 / 7.095 / 7.091 | 13.606 / 12.721 / 12.331 |
| GPT-Neo HF | 2.667529 | 2.670587 | 5906 MiB | 6012 MiB | 8.309 / 7.218 / 7.236 | 13.770 / 13.514 / 13.474 |
| GPT-NeoX HF | 6.450442 | 6.450442 | 5582 MiB | 6664 MiB | 6.911 / 6.396 / 6.396 | 15.241 / 14.724 / 15.176 |
| Llama HF | 2.314239 | 2.314239 | 6522 MiB | 8096 MiB | 8.497 / 8.222 / 7.867 | 18.115 / 16.900 / 16.643 |
| Llama HF GQA | 2.167546 | 2.167546 | 6288 MiB | 7344 MiB | 7.848 / 7.319 / 7.399 | 17.935 / 16.334 / 16.280 |
| BERT HF | 5.879711 | 5.695960 | 5190 MiB | 6064 MiB | 7.168 / 6.354 / 6.345 | 14.739 / 13.699 / 13.644 |
| RoBERTa HF | 5.261613 | 5.267606 | 5190 MiB | 6064 MiB | 6.857 / 6.354 / 6.359 | 15.011 / 13.707 / 13.862 |
| T5 HF | 7.660874 | 7.660884 | 5862 MiB | 7480 MiB | 8.364 / 7.902 / 8.226 | 18.334 / 17.324 / 17.345 |
| LeNet | 0.015324 | 0.015325 | 4750 MiB | 6694 MiB | 1.616 / 1.976 / 1.592 | 3.507 / 3.117 / 3.352 |
| ResNet | 0.000155 / 0.000049 / 0.000106 | 0.000164 / 0.000932 / 0.000052 | 6506 MiB | 7496 MiB | 31.105 / 30.912 / 30.995 | 19.523 / 19.056 / 19.160 |
| VGG | 2.202861 | 2.202862 | 8138 MiB | 8174 MiB | 6.131 / 6.104 / 6.104 | 5.052 / 4.772 / 4.927 |
| MobileNet | 0.005540 | 0.005539 / 0.005539 / 0.005538 | 9682 MiB | 14360 MiB | 42.656 / 42.415 / 42.724 | 23.751 / 23.809 / 23.411 |
| UNet | 0.432288 / 0.432502 / 0.432420 | 0.432160 / 0.432230 / 0.432464 | 6822 MiB | 8512 MiB | 5.082 / 4.739 / 4.729 | 5.816 / 5.712 / 5.734 |
| UNet modern | 0.465648 / 0.465643 / 0.465861 | 0.465919 / 0.465683 / 0.465909 | 8136 MiB | 10662 MiB | 5.823 / 5.752 / 5.750 | 6.875 / 6.132 / 6.168 |
| DiT HF | 0.602640 | 0.602640 | 7136 MiB | 7856 MiB | 4.814 / 4.915 / 4.895 | 15.787 / 15.021 / 15.418 |

**Speed / memory (idle GPU 0, B/C):** transformers ~1.7–2.2× slower on
nntile and a bit more VRAM. ResNet / VGG / MobileNet were faster on
nntile; MobileNet is the VRAM outlier (~9.7 GiB CUDA vs ~14.4 GiB
nntile). Walls B vs C agree to a few tenths of a second on most
models; run A is often ~0.5–1.5 s slower (shared GPU).

### Tiny configs, 10 steps (same box, earlier)

Default `*_tiny_config.json` / `*_hf_tiny_config.json`, 10 steps, seed
42, HF batch 1 seq 16, CNN/DiT batch 2. Tiny models do not fill an A40;
VRAM is mostly context. Kept as a small-correctness baseline.

| Model | CUDA loss | nntile loss | CUDA VRAM | nntile VRAM | CUDA wall | nntile wall |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 HF | 5.560533 | 5.560533 | 346 MiB | 396 MiB | 0.303 s | 0.308 s |
| GPT-Neo HF | 4.673506 | 4.673506 | 346 MiB | 396 MiB | 0.334 s | 0.678 s |
| GPT-NeoX HF | 4.703880 | 4.703880 | 346 MiB | 396 MiB | 0.275 s | 0.478 s |
| Llama HF | 4.752017 | 4.752017 | 346 MiB | 396 MiB | 0.367 s | 0.482 s |
| Llama HF GQA | 4.784087 | 4.784087 | 346 MiB | 396 MiB | 0.330 s | 0.621 s |
| BERT HF | 4.410489 | 4.364455 | 346 MiB | 396 MiB | 0.532 s | 0.461 s |
| RoBERTa HF | 4.439625 | 4.439625 | 346 MiB | 396 MiB | 0.254 s | 0.393 s |
| T5 HF | 6.075338 | 6.075390 | 346 MiB | 396 MiB | 0.401 s | 0.856 s |
| LeNet | 1.758321 | 1.758291 | 352 MiB | 398 MiB | 0.300 s | 0.335 s |
| ResNet | 1.850353 | 1.850353 | 352 MiB | 398 MiB | 0.381 s | 0.626 s |
| VGG | 2.329700 | 2.329700 | 352 MiB | 398 MiB | 0.310 s | 0.386 s |
| MobileNet | 2.301882 | 2.301884 | 352 MiB | 398 MiB | 0.371 s | 0.394 s |
| UNet | 1.094769 | 1.094716 | 316 MiB | 372 MiB | 0.386 s | 0.559 s |
| UNet modern | 1.102412 | 1.102413 | 316 MiB | 372 MiB | 0.611 s | 0.553 s |
| DiT HF | 1.348700 | 1.348700 | 352 MiB | 400 MiB | 0.433 s | 0.727 s |

BERT already mismatches on the tiny recipe.

## Agent checklist

- [ ] `nvidia-smi` shows the chosen GPU idle; `CUDA_VISIBLE_DEVICES` set
- [ ] CUDA build (`USE_CUDA=ON`), `NNTILE_BUILD_DIR` / `STARPU_LIB` set
- [ ] Did not import `torch_nntile` in CUDA processes
- [ ] Did not edit `torch_nntile.models.*` or add cuda to the commons
- [ ] Ran `bench_cuda_vs_nntile_2gb.py` and saved the printed table
- [ ] Compared losses to the 50-step table (BERT outlier is known)
- [ ] If OOM: drop `--cnn-batch-size` or `--hf-batch-size`, do not shrink
      the 2 GiB JSON (weights would no longer be ≥2 GiB)
