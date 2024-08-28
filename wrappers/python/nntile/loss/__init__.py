# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/loss/__init__.py
# Submodule with loss functions of NNTile Python package
#
# @version 1.1.0

from .crossentropy import CrossEntropy
from .frob import Frob

__all__ = ('CrossEntropy', 'Frob')
