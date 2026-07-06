# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/torch_nntile/compat/hf.py
# HuggingFace compatibility shims for device="nntile".

"""Route selected HuggingFace ops to supported nntile kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_patched = False
_patched_device = False
_patched_gpt2 = False
_ORIGINAL_LINEAR = F.linear


def _is_nntile_device(device: object) -> bool:
    if device is None:
        return False
    if isinstance(device, str):
        return device == "nntile" or device.startswith("nntile:")
    if isinstance(device, torch.device):
        return device.type == "nntile"
    return False


def _gpt2_batch_size(
    *,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
) -> int | None:
    if input_ids is not None:
        return int(input_ids.shape[0])
    if inputs_embeds is not None:
        return int(inputs_embeds.shape[0])
    return None


def _gpt2_sequence_length(
    *,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
) -> int | None:
    if input_ids is not None:
        return int(input_ids.shape[-1])
    if inputs_embeds is not None:
        return int(inputs_embeds.shape[-2])
    return None


def patch_gpt2_cache_position() -> None:
    """Create GPT-2 ``cache_position`` on CPU, then copy to nntile."""
    global _patched_gpt2
    if _patched_gpt2:
        return

    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
    except (ImportError, RuntimeError, ModuleNotFoundError):
        return

    _orig_forward = GPT2Model.forward

    def _forward(self, *args, **kwargs):
        if self.wte.weight.device.type != "nntile":
            return _orig_forward(self, *args, **kwargs)

        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        inputs_embeds = kwargs.get("inputs_embeds")
        batch_size = _gpt2_batch_size(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        if kwargs.get("cache_position") is None:
            seq_len = _gpt2_sequence_length(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )
            if seq_len is not None:
                past_key_values = kwargs.get("past_key_values")
                past_seen = (
                    past_key_values.get_seq_length()
                    if past_key_values is not None
                    else 0
                )
                nntile_device = self.wte.weight.device
                kwargs["cache_position"] = torch.arange(
                    past_seen,
                    past_seen + seq_len,
                    device="cpu",
                ).to(nntile_device)

        _orig_wpe_forward = self.wpe.forward

        def _wpe_forward(position_ids: torch.Tensor) -> torch.Tensor:
            out = _orig_wpe_forward(position_ids)
            if (
                batch_size is not None
                and out.device.type == "nntile"
                and out.shape[0] == 1
                and batch_size > 1
            ):
                return out.expand(batch_size, -1, -1).contiguous()
            return out

        self.wpe.forward = _wpe_forward  # type: ignore[method-assign]
        try:
            return _orig_forward(self, *args, **kwargs)
        finally:
            self.wpe.forward = _orig_wpe_forward  # type: ignore[method-assign]

    GPT2Model.forward = _forward  # type: ignore[method-assign]
    _patched_gpt2 = True


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
    """Apply HuggingFace compatibility shims for device='nntile'."""
    global _patched
    if _patched:
        return

    try:
        from transformers.activations import NewGELUActivation
    except ImportError:
        patch_gpt2_cache_position()
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
    patch_gpt2_cache_position()
    patch_hf_device_transfer()
    _patched = True


try:
    patch_hf_activations()
except Exception:
    # transformers/torchvision may be missing or incompatible in minimal CI images.
    pass

__all__ = [
    "patch_hf_activations",
    "patch_gpt2_cache_position",
    "patch_hf_device_transfer",
    "retie_tied_weights",
]
