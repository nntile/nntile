# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/__init__.py
# Main init of NNTile Python package
#
# @version 1.0.0

from .nntile_core import starpu, tile, TransOp, trans, notrans
from . import layer, loss, model, tensor, pipeline, optimizer
