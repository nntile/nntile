# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/optimizer/__init__.py
# Optimizer init as a part of NNTile Python package
#
# @version 1.0.0
# @author Aleksandr Katrutsa
# @date 2023-02-18

from .sgd import SGD
from .adam import Adam, FuseAdam