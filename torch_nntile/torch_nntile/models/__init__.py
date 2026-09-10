# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/models/__init__.py
# Compatibility alias for torch_nntile.nn.model.

"""Compatibility alias for :mod:`torch_nntile.nn.model`.

Prefer::

    from torch_nntile.nn.model import DeepReLU
"""

from __future__ import annotations

import pkgutil
import sys

from torch_nntile.nn import model as _model
from torch_nntile.nn.model import *  # noqa: F403

__all__ = list(_model.__all__)

for _info in pkgutil.iter_modules(_model.__path__):
    _sub = __import__(f"{_model.__name__}.{_info.name}", fromlist=["*"])
    sys.modules[f"{__name__}.{_info.name}"] = _sub
    globals()[_info.name] = _sub
