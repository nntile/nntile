# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/layer/__init__.py
# Submodule with neural network layers of NNTile Python package
#
# @version 1.0.0

from .act import Act
#from .fp32_to_fp16 import FP32_to_FP16
#from .fp16_to_fp32 import FP16_to_FP32
from .add_slice import AddSlice
from .attention import Attention
from .attention_single_head import AttentionSingleHead
from .base_layer import BaseLayer
from .batch_norm import BatchNorm2d
from .embedding import Embedding
from .flash_attention import FlashAttention
from .layer_norm import LayerNorm
from .linear import Linear
from .mixer import GAP, Mixer, MixerMlp
from .rms_norm import RMSNorm
