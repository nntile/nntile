# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/core/__init__.py
# Python facade for the nntile_core extension module (nntile::core bindings).
#
# @version 1.1.0

import sys

from .. import nntile_core

_SUBMODULES = ('tensor', 'starpu', 'tile')
for _subname in _SUBMODULES:
    _submod = getattr(nntile_core, _subname)
    sys.modules[f'{__name__}.{_subname}'] = _submod
    globals()[_subname] = _submod

from ..nntile_core import Context, TransOp, notrans, starpu, tile, trans

__all__ = [
    'Context',
    'TransOp',
    'notrans',
    'starpu',
    'tile',
    'tensor',
    'trans',
]
