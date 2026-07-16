# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/_build_info.py
# Build-time flags for the installed package (overwritten by setup.py).

"""Compile-time configuration baked into the wheel / editable install.

``setup.py`` rewrites ``BUILT_WITH_CUDA`` from the linked libnntile
``defs.h`` (or ``TORCH_NNTILE_USE_CUDA``) so import-time NVIDIA checks
and public ``built_with_cuda()`` match the native libraries.
"""

from __future__ import annotations

# Default for source checkouts before ``pip install`` / wheel build.
BUILT_WITH_CUDA = False
