# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/__init__.py
# PyTorch models for the nntile device.

from .deep_relu import DeepReLU
from .gpt2_minimal import GPT2LMHead, GPT2Model

__all__ = ["DeepReLU", "GPT2LMHead", "GPT2Model"]
