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

- MNIST / CIFAR-10 loading and image patching for
  [`mlp_mixer_nntile.py`](../../wrappers/python/examples/mlp_mixer_nntile.py)

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
