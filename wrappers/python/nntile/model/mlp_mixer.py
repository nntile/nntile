# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/mlp_mixer.py
# MLP-Mixer model of NNTile Python package
#
# @version 1.1.0

from typing import List, Optional, Tuple

import torch
from torch import nn

from nntile.tensor import (
    Tensor_bf16, Tensor_fp16, Tensor_fp32, TensorMoments, TensorTraits,
    notrans, to_numpy,
)

from ..layer.gap import GAP
from ..layer.linear import Linear
from .base_model import BaseModel
from .mixer_block import (
    MixerBlock, _copy_param_from_torch, _copy_param_to_torch,
    _weight_layout_transposed,
)
from .mlp_mixer_config import MlpMixerConfig


def _stem_proj_basetile(x: TensorMoments, config: MlpMixerConfig) -> int:
    if config.projected_patch_dim != x.value.shape[-1]:
        return config.projected_patch_dim
    return x.value.basetile_shape[-1]


def _build_stack(
    x: TensorMoments,
    config: MlpMixerConfig,
    torch_mixer: Optional[nn.Module] = None,
) -> Tuple[Linear, List[MixerBlock], GAP, Linear]:
    stem = Linear.generate_simple(
        x, 'L', notrans, 1,
        [config.projected_patch_dim], [_stem_proj_basetile(x, config)],
        bias=False,
    )
    hidden = stem.activations_output[0]
    torch_blocks = (
        None if torch_mixer is None else torch_mixer.mixer_sequence[1:]
    )
    blocks: List[MixerBlock] = []
    for i in range(config.num_mixer_layers):
        if torch_blocks is not None:
            block = MixerBlock.from_torch(torch_blocks[i], hidden, config)
        else:
            block = MixerBlock.generate_simple(hidden, config)
        hidden = block.activations[-1]
        blocks.append(block)
    gap = GAP.generate_simple(hidden)
    classifier = Linear.generate_simple(
        gap.activations_output[0], 'R', notrans, 1,
        [config.n_classes], [config.n_classes], bias=False,
    )
    return stem, blocks, gap, classifier


class MlpMixer(BaseModel):
    stem: Linear
    blocks: List[MixerBlock]
    gap: GAP
    classifier: Linear

    def __init__(
        self,
        x: TensorMoments,
        stem: Linear,
        blocks: List[MixerBlock],
        gap: GAP,
        classifier: Linear,
        config: MlpMixerConfig,
    ):
        if config.num_mixer_layers < 1:
            raise ValueError("num_mixer_layers must be at least 1")

        self.config = config
        self.stem = stem
        self.blocks = blocks
        self.gap = gap
        self.classifier = classifier

        activations = [x]
        activations.extend(stem.activations_output)
        for block in blocks:
            activations.extend(block.activations[1:])
        activations.extend(gap.activations_output)
        activations.extend(classifier.activations_output)

        layers = [stem]
        for block in blocks:
            layers.extend(block.layers)
        layers.extend([gap, classifier])

        super().__init__(activations, layers, config)

    @staticmethod
    def generate_simple(
        x: TensorMoments, config: MlpMixerConfig,
    ) -> "MlpMixer":
        stack = _build_stack(x, config)
        return MlpMixer(x, *stack, config)

    @staticmethod
    def from_torch(
        torch_mlp_mixer: nn.Module,
        batch_size: int,
        n_classes: int,
        config: Optional[MlpMixerConfig] = None,
    ) -> "MlpMixer":
        if config is None:
            config = MlpMixerConfig(
                channel_dim=torch_mlp_mixer.channel_dim,
                init_patch_dim=torch_mlp_mixer.init_patch_dim,
                projected_patch_dim=torch_mlp_mixer.projected_patch_dim,
                num_mixer_layers=torch_mlp_mixer.num_mixer_layers,
                n_classes=n_classes,
            )
        dtype2tensor_type = {
            "fp32": Tensor_fp32,
            "fp16": Tensor_fp16,
            "bf16": Tensor_bf16,
        }
        x_traits = TensorTraits(
            [config.channel_dim, batch_size, config.init_patch_dim],
            [config.channel_dim, batch_size, config.init_patch_dim],
        )
        x_value = dtype2tensor_type[config.dtype](x_traits, [0])
        x = TensorMoments(x_value, None, False)
        stack = _build_stack(x, config, torch_mlp_mixer)
        model = MlpMixer(x, *stack, config)
        for p_nntile, p_torch in zip(
            model.parameters, torch_mlp_mixer.parameters(),
        ):
            _copy_param_from_torch(p_torch, p_nntile)
        return model

    def to_torch(
        self, torch_mlp_mixer: Optional[nn.Module] = None,
    ) -> nn.Module:
        if torch_mlp_mixer is None:
            from nntile.torch_models.mlp_mixer import (
                MlpMixer as TorchMlpMixer,
            )

            torch_mlp_mixer = TorchMlpMixer(
                self.config.channel_dim,
                self.config.init_patch_dim,
                self.config.projected_patch_dim,
                self.config.num_mixer_layers,
                self.config.n_classes,
            )
        with torch.no_grad():
            for p_nntile, p_torch in zip(
                self.parameters, torch_mlp_mixer.parameters(),
            ):
                _copy_param_to_torch(p_nntile, p_torch)
        return torch_mlp_mixer

    def to_torch_with_grads(self) -> nn.Module:
        torch_model = self.to_torch()
        for p_nntile, p_torch in zip(
            self.parameters, torch_model.parameters(),
        ):
            if p_nntile.grad is None:
                continue
            grad_np = to_numpy(p_nntile.grad)
            if _weight_layout_transposed(
                p_nntile.grad.shape, tuple(p_torch.shape),
            ):
                grad_np = grad_np.T
            p_torch.grad = torch.tensor(grad_np, requires_grad=False)
        return torch_model
