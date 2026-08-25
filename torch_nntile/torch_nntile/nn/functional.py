# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/nn/functional.py
"""NNTile-specific functional kernels (not stock ``torch.nn.functional``).

Stock ``torch.nn`` / ``torch.nn.functional`` on ``device=nntile`` follow the
CUDA dispatch path (torch-native ``aten`` codelets). Use **this** module for
classic NNTile tiled / hand-written ops (``gemm``, ``rope``, ``sum_slice``,
…).

Requires a classic build (``NNTILE_TORCH_NATIVE_OPS=OFF``). Torch-native
wheels expose the same names but raise if called.
"""

from __future__ import annotations

from torch_nntile import TORCH_NATIVE_OPS

__all__: list[str] = []

if not TORCH_NATIVE_OPS:
    from torch_nntile.add_fiber import add_fiber
    from torch_nntile.gemm import gemm, matmul
    from torch_nntile.loss import cross_entropy, mse_loss
    from torch_nntile.norm import vector_norm
    from torch_nntile.normalization import rms_norm
    from torch_nntile.rope import rope, rope_sin_cos_from_position_ids
    from torch_nntile.sum_slice import gap, sum_slice

    __all__ = [
        "add_fiber",
        "apply_nntile_patches",
        "cross_entropy",
        "gap",
        "gemm",
        "matmul",
        "mse_loss",
        "rms_norm",
        "rope",
        "rope_sin_cos_from_position_ids",
        "sum_slice",
        "vector_norm",
    ]

else:

    def _torch_native_only(name: str) -> None:
        raise RuntimeError(
            f"torch_nntile.nn.functional.{name} requires a classic NNTile build "
            "(NNTILE_TORCH_NATIVE_OPS=OFF). On torch-native wheels use stock "
            "torch.nn.functional on device=nntile (CUDA parity)."
        )

    def __getattr__(name: str) -> object:
        if name in {
            "add_fiber",
            "cross_entropy",
            "gap",
            "gemm",
            "matmul",
            "mse_loss",
            "rms_norm",
            "rope",
            "rope_sin_cos_from_position_ids",
            "sum_slice",
            "vector_norm",
        }:
            _torch_native_only(name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def apply_nntile_patches() -> None:
    """Opt-in legacy shim: override selected stock ``F.*`` / ``linalg`` symbols.

    New code should call :func:`rms_norm`, :func:`cross_entropy`, etc. from
    this module explicitly, or use stock ``torch.nn.functional`` on
    ``device=nntile`` (CUDA parity). Do **not** call this in torch-native
    training stacks.
    """
    if TORCH_NATIVE_OPS:
        raise RuntimeError(
            "apply_nntile_patches() is for classic NNTile builds only "
            "(NNTILE_TORCH_NATIVE_OPS=OFF)"
        )
    from torch_nntile.loss import patch_cross_entropy
    from torch_nntile.norm import patch_vector_norm
    from torch_nntile.normalization import patch_rms_norm

    patch_cross_entropy()
    patch_vector_norm()
    patch_rms_norm()
