# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/__init__.py
# Python facade for the nntile extension module (nntile::core bindings).
#
# @version 1.1.0

import sys

from .. import nntile
from ..nntile import Context, TransOp, notrans, starpu, tile, trans

tensor = nntile.tensor

sys.modules[f'{__name__}.tensor'] = tensor
sys.modules[f'{__name__}.starpu'] = starpu
sys.modules[f'{__name__}.tile'] = tile

__all__ = [
    'Context',
    'TransOp',
    'notrans',
    'starpu',
    'tile',
    'tensor',
    'trans',
]
