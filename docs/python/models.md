# Models

Models live in [`wrappers/python/nntile/model/`](../../wrappers/python/nntile/model/).
Import concrete classes from their modules (e.g. `nntile.model.gpt2`).

## BaseModel API

[`base_model.py`](../../wrappers/python/nntile/model/base_model.py):

| Method / attribute | Description |
|--------------------|-------------|
| `layers` | Ordered `BaseLayer` list |
| `parameters` | Aggregated parameter `TensorMoments` |
| `forward_async()` / `backward_async()` | Run all layers sequentially with given input to the first layer |
| `forward_dynamic(x)` | Dynamic-shape forward through layers |
| `clear_gradients()` | Clear gradients and activations |
| `append(layer)` | Add layer and extend activations |

## Demo and vision

| Class | Module | `from_pretrained` |
|-------|--------|-------------------|
| `DeepReLU` | `deep_relu.py` | — |
| `DeepLinear` | `deep_linear.py` | — |
| `MlpMixer` | `mlp_mixer.py` | — (`from_torch` via local PyTorch reference in [`torch_models/mlp_mixer.py`](../../wrappers/python/nntile/torch_models/mlp_mixer.py)) |
| `MlpMixerConfig` | `mlp_mixer_config.py` | — |
| `MixerBlock` | `mixer_block.py` | — |

## BERT

| Class | Module | `from_pretrained` |
|-------|--------|-------------------|
| `BertModel` | `bert.py` | Yes |
| `BertForMaskedLM` | `bert.py` | Yes |
| `BertEncoder`, `BertLayer` | `bert_encoder.py`, `bert_layer.py` | — |
| Submodules | `bert_modules.py` | — |
| `BertConfigNNTile` | `bert_config.py` | — |

## RoBERTa

| Class | Module | `from_pretrained` |
|-------|--------|-------------------|
| `RobertaModel` | `roberta.py` | Yes |
| `RobertaForMaskedLM` | `roberta.py` | Yes |
| `RobertaLMHead` | `roberta_modules.py` | — |

## GPT-2

| Class | Module | `from_pretrained` |
|-------|--------|-------------------|
| `GPT2Model` | `gpt2_model.py` | — |
| `GPT2LMHead` | `gpt2_lmhead.py` | Yes |
| Combined stack + generation | `gpt2.py` | Yes |
| `GPT2Block`, `GPT2MLP`, `GPT2Attention` | `gpt2_block.py`, etc. | — |
| `GPT2ConfigNNTile` | `gpt2_config.py` | — |

## GPT-Neo / GPT-NeoX

| Class | Module | `from_pretrained` |
|-------|--------|-------------------|
| `GPTNeoModel`, `GPTNeoBlock` | `gpt_neo_model.py`, `gpt_neo_block.py` | — |
| `GPTNeoForCausalLM` | `gpt_neo_causal.py` | Yes |
| `GPTNeoXModel`, `GPTNeoXBlock` | `gpt_neox_model.py`, `gpt_neox_block.py` | — |
| `GPTNeoXForCausalLM` | `gpt_neox_causal.py` | Yes |

## Llama

| Class | Module | `from_pretrained` |
|-------|--------|-------------------|
| `Llama` | `llama.py` | — |
| `LlamaDecoder`, `LlamaAttention`, `LlamaMLP` | `llama_decoder.py`, etc. | — |
| `LlamaForCausalLM` | `llama_causal.py` | Yes |
| `LlamaConfigNNTile` | `llama_config.py` | — |

## T5

| Class | Module | `from_pretrained` |
|-------|--------|-------------------|
| `T5Model` | `t5_model.py` | — |
| `T5ForSequenceClassification` | `t5_model.py` | — |
| `T5ForConditionalGeneration` | `t5_model.py` | Yes |
| `T5Block`, `T5Stack`, attention/FF modules | `t5_block.py`, `t5_ff.py` | — |

## Example: load and forward

```python
import nntile
from nntile.model.gpt2 import GPT2ConfigNNTile, GPT2Model

context = nntile.Context(ncpu=-1, ncuda=-1, ooc=0, verbose=0)
context.restrict_cuda()

# Paths depend on your checkpoint layout; see gpt2_custom_training.py
model, _ = GPT2Model.from_pretrained(
    config_path="wrappers/python/examples/gpt2_default_config.json",
    tokenizer_path="...",
    # dtype, tiling, restrict, etc.
)

model.forward_async()
nntile.starpu.wait_for_all()
```

`from_pretrained` converts Hugging Face weights into NNTile tensors with chosen
dtype and basetile shapes.

## Generation and inference

Causal LMs use [`model/generation/`](../../wrappers/python/nntile/model/generation/)
(`LLMGenerationMixin`, samplers, beam search).

For inference engines, the HTTP gateway, and the Telegram bot, see
[inference/README.md](../inference/README.md).

## See also

- [training.md](training.md) — training scripts per model family
- [layers.md](layers.md) — building blocks
