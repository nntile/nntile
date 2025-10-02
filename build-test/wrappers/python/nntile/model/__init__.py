# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/__init__.py
# Submodule with neural network models of NNTile Python package
#
# @version 1.1.0

from .base_model import BaseModel
from .deep_linear import DeepLinear
from .deep_relu import DeepReLU
from .mlp_mixer import MlpMixer

__all__ = ('BaseModel', 'DeepLinear', 'DeepReLU', 'MlpMixer')
