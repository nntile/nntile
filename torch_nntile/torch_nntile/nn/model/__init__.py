# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/model/__init__.py
# Models built from torch_nntile.nn modules (classic nntile kernels).

"""Models that use classic NNTile kernels via :mod:`torch_nntile.nn.module`.

Stock Hugging Face / ``torch.nn`` models can still run on ``device=nntile``
(torch-native kernels). This package is the nntile-kernel rewrite path.
"""

from .bert import BertConfig, BertMlm, BertModel
from .deep_relu import DeepReLU
from .dit import DiT, DiTConfig
from .gpt2_minimal import GPT2LMHead, GPT2Model
from .gpt_neo import GPTNeoCausal, GPTNeoConfig, GPTNeoModel
from .gpt_neox import GPTNeoXCausal, GPTNeoXConfig, GPTNeoXModel
from .llama import LlamaCausal, LlamaConfig, LlamaModel
from .mlp_mixer import MlpMixer, MlpMixerConfig, MlpMixerCpu
from .roberta import RobertaConfig, RobertaMlm, RobertaModel
from .t5 import T5Config, T5ForConditionalGeneration, T5Model

__all__ = [
    "BertConfig",
    "BertMlm",
    "BertModel",
    "DeepReLU",
    "DiT",
    "DiTConfig",
    "GPT2LMHead",
    "GPT2Model",
    "GPTNeoCausal",
    "GPTNeoConfig",
    "GPTNeoModel",
    "GPTNeoXCausal",
    "GPTNeoXConfig",
    "GPTNeoXModel",
    "LlamaCausal",
    "LlamaConfig",
    "LlamaModel",
    "MlpMixer",
    "MlpMixerConfig",
    "MlpMixerCpu",
    "RobertaConfig",
    "RobertaMlm",
    "RobertaModel",
    "T5Config",
    "T5ForConditionalGeneration",
    "T5Model",
]
