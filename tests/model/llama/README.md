# Llama Model Tests

Tests for the NNTile C++ Llama model components, mirroring the Python implementations in `wrappers/python/nntile/model/`.

RMSNorm is a shared module in `nntile::module` (see `include/nntile/module/rms_norm.hh`) and can be reused by other models (GPT, etc.).

## Layout

All tensors use **(hidden_size, seq, batch)** in Fortran (column-major) order. The embedding takes input_ids (seq, batch) and produces (hidden, seq, batch) after transpose. LlamaCausal outputs logits (vocab_size, seq, batch).

## Test Coverage

- **llama_config**: LlamaConfig validation and defaults
- **llama_mlp**: LlamaMLP forward (gated MLP with SiLU)
- **llama_attention**: LlamaAttention forward (sdpa_eager-based)
- **llama_decoder**: LlamaDecoder forward (full block)
- **llama_model**: LlamaModel forward (embed + layers + norm)
- **llama_causal**: LlamaCausal forward (model + lm_head)

## Comparison with Transformers

The Python counterparts in `wrappers/python/nntile/model/` can be validated against HuggingFace `transformers`:

- `LlamaConfig` ↔ `transformers.LlamaConfig`
- `LlamaMLP` ↔ `transformers.models.llama.modeling_llama.LlamaMLP`
- `LlamaAttention` ↔ `transformers.models.llama.modeling_llama.LlamaAttention`
- `LlamaDecoder` ↔ `transformers.models.llama.modeling_llama.LlamaDecoderLayer`
- `LlamaModel` ↔ `transformers.models.llama.modeling_llama.LlamaModel`
- `LlamaCausal` ↔ `transformers.models.llama.modeling_llama.LlamaForCausalLM`

To run PyTorch comparison tests when `NNTILE_HAVE_TORCH` is enabled, build with PyTorch and use the existing test infrastructure (e.g. `tests/module/gated_mlp.cc` pattern).

## Running Tests

```bash
ctest -R tests_model_
```
