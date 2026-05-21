# Training

## Loss

| Class | Module | Use |
|-------|--------|-----|
| `CrossEntropy` | [`loss/crossentropy.py`](../../wrappers/python/nntile/loss/crossentropy.py) | Classification / LM loss |
| `Frob` | [`loss/frob.py`](../../wrappers/python/nntile/loss/frob.py) | Frobenius norm loss |

## Optimizers

| Class | Module |
|-------|--------|
| `Adam` | [`optimizer/adam.py`](../../wrappers/python/nntile/optimizer/adam.py) |
| `AdamW` | [`optimizer/adamw.py`](../../wrappers/python/nntile/optimizer/adamw.py) |
| `SGD` | [`optimizer/sgd.py`](../../wrappers/python/nntile/optimizer/sgd.py) |
| `Empty` | [`optimizer/empty.py`](../../wrappers/python/nntile/optimizer/empty.py) |

Fused low-level steps: `fused_adam_step`, `fused_adamw_step`, `fused_sgd_step` in
[functions.md](functions.md).

## Pipeline

[`pipeline.py`](../../wrappers/python/nntile/pipeline.py) runs epoch loops:

- Input batches `x`, labels `y`
- `model.forward_async` / `model.backward_async` over minibatches
- `loss.calc_async()`
- `opt.step()` after gradient accumulation over minibatches
- Optional **SGOC** graph capture per batch (`graph_recording_begin` / `end`)

See [sgoc/README.md](../sgoc/README.md).

## Training scripts

Under [`wrappers/python/examples/`](../../wrappers/python/examples/):

| Script | Task |
|--------|------|
| [`gpt2_custom_training.py`](../../wrappers/python/examples/gpt2_custom_training.py) | Custom GPT-2 implementation (recommended first training example) |
| [`gpt2_lmhead_training.py`](../../wrappers/python/examples/gpt2_lmhead_training.py) | GPT-2 LM head |
| [`gpt2_training.py`](../../wrappers/python/examples/gpt2_training.py) | GPT-2 (WikiText-103 inline) |
| [`bert_training.py`](../../wrappers/python/examples/bert_training.py) | BERT masked LM |
| [`roberta_training.py`](../../wrappers/python/examples/roberta_training.py) | RoBERTa masked LM |
| [`gpt_neo_training.py`](../../wrappers/python/examples/gpt_neo_training.py) | GPT-Neo causal LM |
| [`gpt_neox_training.py`](../../wrappers/python/examples/gpt_neox_training.py) | GPT-NeoX causal LM |
| [`llama_training.py`](../../wrappers/python/examples/llama_training.py) | Llama causal LM |
| [`t5_lmhead_training.py`](../../wrappers/python/examples/t5_lmhead_training.py) | T5 conditional generation |
| [`deep_relu_image_classification.py`](../../wrappers/python/examples/deep_relu_image_classification.py) | Image classification (MNIST/CIFAR) |
| [`deep_relu.py`](../../wrappers/python/examples/deep_relu.py) | Small DeepReLU demo |
| [`deep_linear.py`](../../wrappers/python/examples/deep_linear.py) | Deep linear demo |
| [`mlp_mixer_nntile.py`](../../wrappers/python/examples/mlp_mixer_nntile.py) | MLP-Mixer |
| [`gpt2_perf_workflow.py`](../../wrappers/python/examples/gpt2_perf_workflow.py) | Forward/backward performance |

Typical CLI knobs: `--batch`, `--minibatch`, `--seq-tile`, `--embd-tile`,
`--restrict=cuda`, `--dtype`, `--optimizer`, `--nepochs`, `--dataset`,
`--config-path`, checkpoint paths. Run the script with `--help` for all options.

### Quick try (inside Docker)

After [building the image](../build/README.md):

```shell
docker run -it --gpus all nntile:latest
# inside the container:
CUDA_VISIBLE_DEVICES=0 STARPU_NCPU=2 \
  python /workspace/nntile/wrappers/python/examples/gpt2_custom_training.py \
  --config-path=/workspace/nntile/wrappers/python/examples/gpt2_default_config.json \
  --tokenizer=gpt2 --tokenizer-path=data \
  --batch=1024 --minibatch=4 --minibatch-tile=4 \
  --seq-tile=1024 --embd-tile=768 --inner-tile=3072 --head-tile=12 \
  --restrict=cuda --flashattention \
  --nforward=10 --nforward-warmup=10 --nbackward=10 --nbackward-warmup=10 \
  --dataset=WikiText-103 --dataset-path=data --dataset-select=40000 \
  --optimizer=fusedadamw --optimizer-eps=1e-8 --weight-decay=0.1 \
  --loss-reduction=mean --lr=3e-4 --start-lr=0 --full-lr-iter=10 \
  --nepochs=1 --nepochs-warmup=1
```

## Jupyter notebooks

[`notebooks/`](../../notebooks/) mirror several models with HuggingFace comparison and
training cells:

- `bert.ipynb`, `roberta.ipynb`
- `gpt2_lmhead.ipynb`, `gpt_neo_lmhead.ipynb`, `gpt_neox_lmhead.ipynb`
- `llama_lmhead.ipynb`, `Llama.ipynb`, `t5_lmhead.ipynb`

Six notebooks include **DMDASD vs SGOC** benchmarks — [sgoc/README.md](../sgoc/README.md).

Launch Jupyter Lab inside Docker:

```shell
docker run -it --gpus all -p 8888:8888 nntile:latest \
  jupyter lab --notebook-dir=/workspace/nntile --ip='*' --port=8888 \
  --no-browser --allow-root
```

For the classic Notebook UI, use `jupyter notebook` instead of `jupyter lab`.
TensorBoard: expose port `6006` with `-p 6006:6006`.

## Inference

Generation scripts, `nntile.inference` engines, the **nntile_gateway** HTTP service,
and the **nntile_tgbot** Telegram front-end are documented in
[inference/README.md](../inference/README.md).

## Environment variables (common)

| Variable | Role |
|----------|------|
| `CUDA_VISIBLE_DEVICES` | GPUs visible to StarPU |
| `STARPU_NCPU` | CPU worker count |
| `STARPU_SCHED` / `STARPU_SCHED_LIB` | Scheduler (see SGOC doc) |
| `STARPU_LIMIT_CUDA_MEM` | VRAM cap for experiments |

## See also

- [data-preparation.md](data-preparation.md) — `.bin` datasets
- [models.md](models.md) — model classes
- [build/README.md](../build/README.md) — build and test
