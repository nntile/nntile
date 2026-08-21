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
| **Wall** | Train loop only (`wall=…s` or GPT-2 `timing … train wall`). Excludes import, HF construct, StarPU `init_context`. |
| **VRAM** | `nvidia-smi` peak during the child minus idle-before. Includes leftover CUDA context (~300–500 MiB). |

On nntile each step `compile_graph` / `run`s and syncs loss with
`.to("cpu")` so StarPU reclaim stays in that step (debt D7). That
sync is inside the reported wall. CUDA uses `--disable-tf32` (full
FP32) for a fair numeric compare.

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

Wall clock on one A40 was ~13 minutes for the full 15-model suite.
Expect longer if the GPU is slower or contended.

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

Measured 2026-08-21 on a **shared** box of NVIDIA A40 (46 GiB), GPU
index 2, branch `nntile-no-implicit-host-copy`. Another user may have
been on other GPUs; **re-run on a quiet GPU** before treating walls as
gospel. Losses should still match.

### 2 GiB, 50 steps (this recipe)

HF batch 16, seq 32; CNN/DiT batch 4; seed 42.

| Model | CUDA loss | nntile loss | CUDA VRAM | nntile VRAM | CUDA wall | nntile wall |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 HF | 7.795432 | 7.795432 | 6416 MiB | 6204 MiB | 7.883 s | 13.606 s |
| GPT-Neo HF | 2.667529 | 2.670587 | 5902 MiB | 6008 MiB | 8.309 s | 13.770 s |
| GPT-NeoX HF | 6.450442 | 6.450442 | 5578 MiB | 6660 MiB | 6.911 s | 15.241 s |
| Llama HF | 2.314239 | 2.314239 | 6518 MiB | 8092 MiB | 8.497 s | 18.115 s |
| Llama HF GQA | 2.167546 | 2.167546 | 6284 MiB | 7340 MiB | 7.848 s | 17.935 s |
| BERT HF | 5.879711 | 5.695960 | 5186 MiB | 6060 MiB | 7.168 s | 14.739 s |
| RoBERTa HF | 5.261613 | 5.267606 | 5186 MiB | 6060 MiB | 6.857 s | 15.011 s |
| T5 HF | 7.660874 | 7.660884 | 5858 MiB | 7476 MiB | 8.364 s | 18.334 s |
| LeNet | 0.015324 | 0.015325 | 4746 MiB | 6690 MiB | 1.616 s | 3.507 s |
| ResNet | 0.000155 | 0.000164 | 6502 MiB | 7492 MiB | 31.105 s | 19.523 s |
| VGG | 2.202861 | 2.202862 | 8134 MiB | 8170 MiB | 6.131 s | 5.052 s |
| MobileNet | 0.005540 | 0.005539 | 9678 MiB | 14356 MiB | 42.656 s | 23.751 s |
| UNet | 0.432288 | 0.432160 | 6818 MiB | 8508 MiB | 5.082 s | 5.816 s |
| UNet modern | 0.465648 | 0.465919 | 8132 MiB | 10660 MiB | 5.823 s | 6.875 s |
| DiT HF | 0.602640 | 0.602640 | 7132 MiB | 7852 MiB | 4.814 s | 15.787 s |

**Loss:** GPT-2, NeoX, Llama, Llama GQA, DiT match to printed 1e-6.
Most others are within ~1e-5. **BERT is a known outlier** (eager CE /
`ignore_index` path). GPT-Neo and RoBERTa are slightly off (~3e-3 and
~6e-3). CNN losses near zero are repeated-batch overfit.

**Speed / memory (this A40 run):** transformers ~1.7–2.2× slower on
nntile and a bit more VRAM. ResNet / VGG / MobileNet were faster on
nntile; MobileNet is the VRAM outlier (~9.7 GiB CUDA vs ~14.4 GiB
nntile).

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
