# Data preparation

Scripts under [`wrappers/python/examples/`](../../wrappers/python/examples/)
prepare tokenized datasets for training.

## Causal language modeling

[`causal_lm_data_preparation.py`](../../wrappers/python/examples/causal_lm_data_preparation.py)

- Loads a Hugging Face dataset (default: `roneneldan/TinyStories`)
- Tokenizes with `--hf-tokenizer`
- Packs token IDs into a NanoGPT-style binary stream (`uint16` / `int64` layout
  used by training scripts)
- Writes `train.bin` (and related files under `--dataset-path`)

Common options:

```shell
python wrappers/python/examples/causal_lm_data_preparation.py \
  --hf-dataset roneneldan/TinyStories \
  --hf-tokenizer openai-community/gpt2 \
  --dataset-path .data \
  --seq-len 1024 \
  --batch-size 256 \
  --dataset-select 5000
```

Used by GPT-2, GPT-Neo, GPT-NeoX, Llama training scripts and notebooks
(`--dataset-file=tinystories/train.bin` or similar).

## Masked language modeling

[`mlm_data_preparation.py`](../../wrappers/python/examples/mlm_data_preparation.py)

- Same HF → tokenize → `.bin` workflow for **masked LM**
- Masking is applied in the training script (`bert_training.py`, `roberta_training.py`),
  not in the prep script

## Vision (MLP-Mixer)

[`mlp_mixer_data_preparation.py`](../../wrappers/python/examples/mlp_mixer_data_preparation.py)

Helper module used by
[`mlp_mixer_training.py`](../../wrappers/python/examples/mlp_mixer_training.py):

- `mnist_data_loader_to_nntile` / `cifar_data_loader_to_nntile` — patch images to
  `[n_patches, minibatch, patch_dim]` and build StarPU batch lists
- `color_image_patching` — RGB (CIFAR-style) patching
- `DTYPE_TO_ACTIVATION_TENSOR` — map `--dtype` (`fp32`, `bf16`, `tf32`, …) to
  activation tensor types (must match the model)

The training script downloads MNIST, Fashion-MNIST, or CIFAR-10 via torchvision
(`--dataset`, `--data-root`); you do not run the prep module as a standalone CLI.

Example:

```shell
python wrappers/python/examples/mlp_mixer_training.py \
  --dataset mnist --data-root ./data \
  --batch-size 60 --minibatch-size 3 --patch-size 7 \
  --hidden-dim 2048 --num-mixer-layers 8 \
  --dtype fp32 --restrict cuda --nepochs 1
```

Checkpoints: `--save-checkpoint-path` writes a `.pt` with `model_state_dict`;
resume with `--checkpoint-path` (architecture flags must match). See
[`mlp_mixer.ipynb`](../../notebooks/mlp_mixer.ipynb) for notebook workflows.

## Workflow

1. Run the appropriate prep script once.
2. Point training at the output, e.g. `--dataset train.bin` or
   `--dataset-file=tinystories/train.bin`.
3. Ensure `--tokenizer` / `--tokenizer-path` match the prep script’s
   `--hf-tokenizer`.

Some scripts (e.g. `gpt2_custom_training.py`) can use `--dataset WikiText-103`
to download inline instead of a `.bin` file.

## See also

- [training.md](training.md) — training entry points
- [models.md](models.md) — model-specific configs in `examples/*_config.json`
