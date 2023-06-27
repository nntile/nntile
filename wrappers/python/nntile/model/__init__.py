# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/__init__.py
# Submodule with neural network models of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-02-17

from .base_model import BaseModel
from .deep_linear import DeepLinear
from .deep_relu import DeepReLU
from .gpt2 import GPT2Config, GPT2Model

