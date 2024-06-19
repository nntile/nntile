# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/optimizer/__init__.py
# Optimizer init as a part of NNTile Python package
#
# @version 1.0.0

from .sgd import SGD
from .adam import Adam, FusedAdam
from .adamw import FusedAdamW
from .empty import Empty
