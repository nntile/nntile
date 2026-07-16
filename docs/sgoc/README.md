# SGOC scheduler

**SGOC** (*Single-GPU offload-checkpoint*) is a loadable StarPU scheduling policy
that records one training batch as a dependency graph over data handles, plans
GPU residency (with optional activation checkpoints), and replays tasks in
topological order. It targets **single-GPU training under a tight VRAM budget**.

SGOC lives in the [**nntile/starpu**](https://github.com/nntile/starpu) fork
(`new_sched/` → `libgraph_sgoc_sched.so`). It is **not** the NNTile Graph API
(see [graph-wip.md](../graph-wip.md)).

## Installation

The recommended path is the Docker **sandbox** stage, which builds StarPU and
SGOC before NNTile is compiled.

From [`Dockerfile`](../../Dockerfile) (sandbox stage):

1. Download and build **StarPU** from `nntile/starpu` (branch `master`) into
   `${CONDA_PREFIX}`.
2. Build the scheduler DSO:

   ```bash
   cd new_sched
   make -j STARPU_BUILD="${CONDA_PREFIX}" EXTRA_CFLAGS=-O2
   install -D -m 755 libgraph_sgoc_sched.so /usr/local/lib/libgraph_sgoc_sched.so
   ldconfig
   ```

The full image (`docker build . -t nntile:latest`) performs these steps
automatically. Override fork/branch at image build time if needed:

```shell
docker build . -t nntile:latest \
  --build-arg STARPU_GITHUB_REPO=nntile/starpu \
  --build-arg STARPU_GIT_BRANCH=master
```

## Runtime configuration

| Variable | Role |
|----------|------|
| `STARPU_SCHED=sgoc` | Select the SGOC policy |
| `STARPU_SCHED_LIB=/usr/local/lib/libgraph_sgoc_sched.so` | Path to the scheduler DSO (required for graph capture) |
| `STARPU_LIMIT_CUDA_MEM=N` | Artificial VRAM cap in **mebibytes** (StarPU docs) |

Example prefix for a training script:

```bash
STARPU_LIMIT_CUDA_MEM=14000 \
STARPU_SCHED_LIB=/usr/local/lib/libgraph_sgoc_sched.so \
STARPU_SCHED=sgoc \
python torch_nntile/examples/train_gpt2_hf.py train --device nntile ...
```

With default `STARPU_SCHED=dmdasd`, training works unchanged; graph capture calls
are no-ops.

## NNTile integration

### Current Python integration status

The former Python bindings exposed graph-capture helpers and
`Pipeline.train_async` hooks for SGOC. The current user-facing Python package is
[`torch_nntile`](../../torch_nntile/README.md); SGOC is still selected through
the StarPU environment variables above, but equivalent automatic graph-capture
hooks have not yet been migrated into `torch_nntile`.

## Benchmarks in notebooks

Six notebooks include a section **“New graph scheduler for a single GPU with
limited memory”**: same training CLI as the DMDASD run, with SGOC enabled only
via environment variables on the second run.

| Notebook | Training script | VRAM limit (MiB) |
|----------|-----------------|------------------|
| [notebooks/bert.ipynb](../../notebooks/bert.ipynb) | `bert_training.py` | 4000 |
| [notebooks/roberta.ipynb](../../notebooks/roberta.ipynb) | `roberta_training.py` | 1000 |
| [notebooks/gpt2_lmhead.ipynb](../../notebooks/gpt2_lmhead.ipynb) | `gpt2_lmhead_training.py` | 14000 |
| [notebooks/gpt_neo_lmhead.ipynb](../../notebooks/gpt_neo_lmhead.ipynb) | `gpt_neo_training.py` | 6000 |
| [notebooks/gpt_neox_lmhead.ipynb](../../notebooks/gpt_neox_lmhead.ipynb) | `gpt_neox_training.py` | 5000 |
| [notebooks/llama_lmhead.ipynb](../../notebooks/llama_lmhead.ipynb) | `llama_training.py` | 14000 |

`Llama.ipynb` and `t5_lmhead.ipynb` do not include this comparison yet.

### Methodology

1. Set `STARPU_SCHED=dmdasd` in the notebook environment.
2. Run training with `STARPU_LIMIT_CUDA_MEM=<N>` (baseline).
3. Re-run the **same** command with `STARPU_SCHED=sgoc` and `STARPU_SCHED_LIB=...`.

Primary metric: wall-clock **`NNTile training time`** in the log. Loss values
should be similar; the comparison targets scheduling and data movement, not
accuracy.

### Summary results (saved notebook outputs)

Relative change: Δ = (T_SGOC − T_DMDASD) / T_DMDASD.

| Model | VRAM (MiB) | DMDASD (s) | SGOC (s) | Δ |
|-------|------------|------------|----------|-----|
| BERT | 4000 | 41.46 | 31.52 | −23.9% |
| RoBERTa | 1000 | 85.73 | 72.64 | −15.3% |
| GPT-2 LMHead | 14000 | 664.04 | 478.85 | −27.9% |
| GPT-Neo LMHead | 6000 | 145.27 | 176.35 | **+21.4%** |
| GPT-NeoX LMHead | 5000 | 233.25 | 182.33 | −21.8% |
| LLaMA LMHead | 14000 | 471.89 | 77.36 | −83.6% |

GPT-Neo is the case where SGOC increased wall time despite lower transfer
volume in logs—tuning of topo-order and checkpoint heuristics may be needed.

## See also

- [build/README.md](../build/README.md) — Docker and StarPU build
- [python/training.md](../python/training.md) — training scripts used in notebooks
