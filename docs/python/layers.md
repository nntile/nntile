# Layers

Layers live in [`wrappers/python/nntile/layer/`](../../wrappers/python/nntile/layer/).
Import from `nntile.layer`.

## BaseLayer API

[`base_layer.py`](../../wrappers/python/nntile/layer/base_layer.py):

| Method | Description |
|--------|-------------|
| `generate_simple(...)` | Static factory: build tensors and return layer (per subclass) |
| `forward_async()` / `forward()` | Forward pass (blocking waits on StarPU) |
| `backward_async()` / `backward()` | Backward pass |
| `forward_dynamic(x)` | Optional dynamic-shape forward (subclasses) |
| `init_randn_async()` | Randomize parameters initialization|
| `clear_gradients()` | Zero parameter gradients |

Each layer holds `activations_input`, `activations_output`, `parameters`,
`temporaries` as `TensorMoments` or `Tensor` lists.

## Layer catalog

Exported in [`layer/__init__.py`](../../wrappers/python/nntile/layer/__init__.py):

| Class | Description |
|-------|-------------|
| `Linear` | GEMM linear layer; `side` `'L'`/`'R'`, `trans_x`, optional bias |
| `Act` | Activation: `relu`, `gelu`, `gelutanh`, `silu` |
| `Add` | Elementwise add of two activations |
| `AddSlice` | Add broadcast slice along an axis |
| `Multiply` | Elementwise multiply |
| `Embedding` | Token embedding from int64 indices |
| `LayerNorm` | Layer normalization |
| `RMSNorm` | RMS norm (Llama-style) |
| `Attention` | Multi-head attention |
| `AttentionSingleHead` | Single-head attention |
| `Sdpa` | Scaled dot-product attention (vanilla or cuDNN flash) |
| `BertSelfAttention` | BERT-style self-attention |
| `GPTNeoAttention` | GPT-Neo attention |
| `GPTNeoXAttention` | GPT-NeoX attention |
| `Conv2d` | 2D convolution |
| `BatchNorm2d` | 2D batch normalization |
| `GAP` | Global average pooling over the patch axis ([`gap.py`](../../wrappers/python/nntile/layer/gap.py)) |
| `MixerMlp` | MLP-Mixer token- or channel-mixing MLP (`side` `'L'` / `'R'`) |
| `MixerBlock` | Full Mixer block (LayerNorm + two `MixerMlp` + residuals); implementation in [`model/mixer_block.py`](../../wrappers/python/nntile/model/mixer_block.py), re-exported from `nntile.layer` |

The stacked vision model is [`MlpMixer`](../models.md) in `nntile.model.mlp_mixer`.

Also used by models (not all in `__all__`): `T5Attention` ([`t5_attention.py`](../../wrappers/python/nntile/layer/t5_attention.py)),
KV cache helpers in [`cache_utils.py`](../../wrappers/python/nntile/layer/cache_utils.py).

## Example: Linear layer

```python
import nntile
from nntile.tensor import TensorMoments, TensorTraits, from_array
from nntile.layer import Linear
from nntile.nntile import notrans
import numpy as np

context = nntile.Context(ncpu=1, ncuda=1, ooc=0, verbose=0)
context.restrict_cuda()

x_np = np.random.randn(32, 64).astype(np.float32)
x_val = from_array(x_np, basetile_shape=[32, 64])
x = TensorMoments(x_val, None, False)

layer, y = Linear.generate_simple(
    x,
    side="L",
    trans_x=notrans,
    in_features_ndim=1,
    out_features_shape=[128],
    out_features_basetile_shape=[128],
    bias=True,
)

layer.forward_async()
nntile.starpu.wait_for_all()
```

For a minimal end-to-end stack, see [`examples/deep_relu.py`](../../wrappers/python/examples/deep_relu.py).

## See also

- [models.md](models.md) — composing layers into models
- [functions.md](functions.md) — ops used inside layers
