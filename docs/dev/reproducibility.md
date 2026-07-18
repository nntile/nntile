# Reproducibility: torch-native CPU / GPU overhead protocol

This note is the **measurement recipe** for comparing stock (torch-native)
HuggingFace / CNN / DiT training on plain PyTorch vs `device=nntile`.
Use it on any host (Cloud Agent CPU VM, GPU server, laptop) so walls stay
comparable.

Specialized `torch_nntile.models.*` stacks are **out of scope**: they are
disabled under `NNTILE_TORCH_NATIVE_OPS`. Only stock modules that go through
PrivateUse1 aten ops are measured.

## What we measure

| Suite | Purpose | Typical wall |
|-------|---------|--------------|
| **Tiny smoke** | Correctness + overhead-dominated timings | ≪1 s train loop |
| **Middle** | Same recipes with larger models / batches so compute dominates | ~1 min train loop |

Wall times are the printed **train-loop** times (`wall=…s` or GPT-2
`timing … train wall`), **not** process elapsed (imports / HF init /
`init_context`).

## Single-core host protocol (required)

Overhead comparisons pin **one host compute thread**. StarPU worker count
defaults to **one** (`--ncpu 1`); pass `--ncpu 2` (or higher) only when
measuring multi-worker scaling — keep host BLAS single-threaded either way:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export STARPU_SILENT=1 STARPU_FXT_TRACE=0 STARPU_WORKERS_NOBIND=1
```

Train helpers call `configure_single_thread_host()`
(`torch.set_num_threads(1)` + the env defaults above). Pass `--ncpu 1` or
`--ncpu 2` for `device=nntile`. Bench runners also force the env in the
child process.

## Build / install (torch 2.9.1)

```bash
# CPU StarPU build (Cloud Agent / CPU-only hosts)
export PKG_CONFIG_PATH=/opt/starpu/lib/pkgconfig
python3 -m venv .venv && source .venv/bin/activate
pip install 'torch==2.9.1' 'torchvision==0.24.1' \
  --index-url https://download.pytorch.org/whl/cpu
pip install 'transformers<4.53' datasets 'diffusers==0.32.2'

TORCH_PREFIX=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
  -DBUILD_TESTING=OFF \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -GNinja
cmake --build build --target nntile torch_nntile -j$(nproc)

export NNTILE_BUILD_DIR=$PWD/build TORCH_NNTILE_BUILD_DIR=$PWD/build
export NNTILE_SOURCE_DIR=$PWD
export LD_LIBRARY_PATH=$PWD/build/nntile:$PWD/build/torch_nntile:/opt/starpu/lib
CXX=g++ pip install -e ./torch_nntile --no-build-isolation --force-reinstall
```

On a **CUDA** host, rebuild with `-DUSE_CUDA=ON` against a matching
`torch==2.9.1` CUDA wheel, and install `diffusers` / `datasets` the same way.

## Tiny smokes (all torch-native models)

```bash
# nntile only (default DEVICE=nntile, NCPU=1)
./torch_nntile/examples/run_all_torch_native_smokes.sh

# CPU vs nntile tables
python torch_nntile/examples/bench_hf_tiny_cpu_vs_nntile.py --ncpu 1
python torch_nntile/examples/bench_cnn_tiny_cpu_vs_nntile.py --ncpu 1
python torch_nntile/examples/bench_dit_hf_tiny_cpu_vs_nntile.py --ncpu 1

# Optional: two StarPU CPU workers (tiny stays overhead-dominated)
python torch_nntile/examples/bench_hf_tiny_cpu_vs_nntile.py --ncpu 2
python torch_nntile/examples/bench_cnn_tiny_cpu_vs_nntile.py --ncpu 2
python torch_nntile/examples/bench_dit_hf_tiny_cpu_vs_nntile.py --ncpu 2
```

Documented results:

- [hf_tiny_cpu_vs_nntile_showcase.md](hf_tiny_cpu_vs_nntile_showcase.md)
- [cnn_tiny_cpu_vs_nntile_showcase.md](cnn_tiny_cpu_vs_nntile_showcase.md)
- [dit_tiny_cpu_vs_nntile_showcase.md](dit_tiny_cpu_vs_nntile_showcase.md)

## Middle suite (~1 minute / train)

Committed JSON configs (`*_middle_config.json`) plus step / batch / seq
recipes:

[`torch_nntile/examples/torch_native_middle_recipes.json`](../../torch_nntile/examples/torch_native_middle_recipes.json)

```bash
python torch_nntile/examples/bench_torch_native_middle_cpu_vs_nntile.py \
  --families hf,cnn,dit \
  --markdown-out /tmp/torch_native_middle.md

# Two StarPU CPU workers (nntile only; compare to CPU column from above)
python torch_nntile/examples/bench_torch_native_middle_cpu_vs_nntile.py \
  --families hf,cnn,dit --devices nntile --ncpu 2 \
  --markdown-out /tmp/torch_native_middle_ncpu2.md
```

Override a single model while tuning a new machine::

```bash
python torch_nntile/examples/bench_torch_native_middle_cpu_vs_nntile.py \
  --families hf --only llama \
  --markdown-out /tmp/llama_middle.md
```

Expect **nntile/CPU wall ratios closer to 1.0** than on tiny smokes: StarPU
submit + compile/run + host sync become a smaller fraction of the math.
With `--ncpu 2`, several middle HF models beat single-thread torch.

Results summary: [torch_native_middle_cpu_vs_nntile.md](torch_native_middle_cpu_vs_nntile.md).

## GPU server checklist (manual)

Goal: see whether `device=nntile` shows overhead vs torch on a GPU host, and
whether middle-sized work still amortizes that overhead.

1. Build with `USE_CUDA=ON` and install the matching CUDA `torch==2.9.1`.
2. Keep host BLAS single-threaded (`OMP_NUM_THREADS=1`, …) so CPU prep does
   not hide device time.
3. Compare **three** runs with the same middle config / seed / steps:

| Run | Command sketch |
|-----|----------------|
| Torch CUDA | `--device cuda` (GPT-2 HF already; for HF/CNN/DiT commons add a cuda device if needed, or use GPT-2 / a thin cuda wrapper) |
| Nntile CUDA workers | `--device nntile --ncpu 0 --ncuda 1 --restrict-cuda` |
| Nntile CPU worker (control) | `--device nntile --ncpu 1 --ncuda 0 --restrict-cpu` |

Example (GPT-2 middle)::

```bash
python torch_nntile/examples/train_gpt2_hf.py train \
  --device cuda --seed 0 \
  --config torch_nntile/examples/gpt2_hf_middle_config.json \
  --epochs 1 --seq-len 256 --batch-size 8 --max-sequences 192 \
  --output-dir /tmp/gpt2_cuda --no-shuffle --disable-tf32

python torch_nntile/examples/train_gpt2_hf.py train \
  --device nntile --seed 0 \
  --config torch_nntile/examples/gpt2_hf_middle_config.json \
  --epochs 1 --seq-len 256 --batch-size 8 --max-sequences 192 \
  --ncpu 0 --ncuda 1 --restrict-cuda \
  --output-dir /tmp/gpt2_nntile_cuda --no-shuffle
```

HF / CNN / DiT commons accept `--ncuda` / `--restrict-cuda` the same way
(they currently expose `cpu` and `nntile` devices; use GPT-2 HF or a local
`--device cuda` fork when you need a pure torch CUDA baseline for those
models).

4. Record: host model / CUDA driver / torch build, printed train walls, final
   losses, and `nntile/torch` ratio. Attach a short table next to the CPU
   middle results in
   [torch_native_middle_cpu_vs_nntile.md](torch_native_middle_cpu_vs_nntile.md).

## Recording a new machine

When you re-run on another server, update the showcase / middle docs with:

- date, hostname class (CPU VM / GPU SKU), `torch` version, `USE_CUDA`
- `ncpu` / `ncuda` / host thread pin
- the markdown table from the bench script
- note if recipe steps were scaled (keep configs; only change `steps` in
  `torch_native_middle_recipes.json` or via a local override)

Do **not** mix multi-thread CPU torch walls with `ncpu=1` nntile walls in the
same table.
