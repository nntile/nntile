# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/compat/hf.py
# HuggingFace compatibility shims for device="nntile".

"""Route selected HuggingFace ops to supported nntile kernels.

Do **not** monkey-patch HuggingFace ``GPT2Model.forward``. Stock GPT-2
builds ``cache_position`` with ``torch.arange`` on ``device=nntile``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_patched = False
_patched_device = False
_ORIGINAL_LINEAR = F.linear


def _is_nntile_device(device: object) -> bool:
    if device is None:
        return False
    if isinstance(device, str):
        return device == "nntile" or device.startswith("nntile:")
    if isinstance(device, torch.device):
        return device.type == "nntile"
    return False


def patch_hf_device_transfer() -> None:
    """Re-tie HF weights after ``.to('nntile')`` (device move breaks sharing)."""
    global _patched_device
    if _patched_device:
        return

    try:
        from transformers import PreTrainedModel
    except ImportError:
        return

    _orig_to = PreTrainedModel.to

    def _to(self, *args, **kwargs):
        device = args[0] if args else kwargs.get("device")
        result = _orig_to(self, *args, **kwargs)
        if _is_nntile_device(device) and hasattr(self, "tie_weights"):
            self.tie_weights()
        return result

    PreTrainedModel.to = _to  # type: ignore[method-assign]
    _patched_device = True


def retie_tied_weights(model: torch.nn.Module) -> None:
    """Re-apply HF tied embeddings after a manual parameter move."""
    if hasattr(model, "tie_weights"):
        model.tie_weights()


def _nntile_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reshape ND activations to 2D for nntile ``linear`` backward support."""
    if input.device.type == "nntile" and input.dim() > 2:
        out_features = weight.shape[0]
        lead = input.shape[:-1]
        flat = input.reshape(-1, input.size(-1))
        out = _ORIGINAL_LINEAR(flat, weight, bias)
        return out.reshape(*lead, out_features)
    return _ORIGINAL_LINEAR(input, weight, bias)


def patch_hf_activations() -> None:
    """Apply HuggingFace activation / linear shims for device='nntile'."""
    global _patched
    if _patched:
        return

    try:
        from transformers.activations import NewGELUActivation
    except ImportError:
        patch_hf_device_transfer()
        _patched = True
        return

    _orig_forward = NewGELUActivation.forward

    def _forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type == "nntile":
            return torch.nn.functional.gelu(input, approximate="tanh")
        return _orig_forward(self, input)

    NewGELUActivation.forward = _forward
    F.linear = _nntile_linear  # type: ignore[assignment]
    patch_hf_device_transfer()
    _patched = True


try:
    patch_hf_activations()
except Exception:
    # transformers may be missing in minimal CI images.
    pass

__all__ = [
    "patch_hf_activations",
    "patch_hf_device_transfer",
    "retie_tied_weights",
]
