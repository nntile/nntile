# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/mixer_block.py
# Mixer block submodule of NNTile Python package
#
# @version 1.1.0

from typing import TYPE_CHECKING

import numpy as np
import torch

from nntile.tensor import TensorMoments, to_numpy

from ..layer.add import Add
from ..layer.layer_norm import LayerNorm
from ..layer.mixer_mlp import MixerMlp
from .base_model import BaseModel
from .mlp_mixer_config import MlpMixerConfig

if TYPE_CHECKING:
    from nntile.torch_models.mlp_mixer import Mixer as TorchMixer


class MixerBlock(BaseModel):
    norm_1: LayerNorm
    mlp_1: MixerMlp
    post_mlp1_add: Add
    norm_2: LayerNorm
    mlp_2: MixerMlp
    post_mlp2_add: Add

    def __init__(
        self,
        x: TensorMoments,
        norm_1: LayerNorm,
        mlp_1: MixerMlp,
        post_mlp1_add: Add,
        norm_2: LayerNorm,
        mlp_2: MixerMlp,
        post_mlp2_add: Add,
        config: MlpMixerConfig,
    ):
        self.config = config
        layers = [
            norm_1,
            mlp_1.linear_1, mlp_1.act, mlp_1.linear_2,
            post_mlp1_add,
            norm_2,
            mlp_2.linear_1, mlp_2.act, mlp_2.linear_2,
            post_mlp2_add,
        ]
        activations = [x]
        activations += norm_1.activations_output
        activations += (
            mlp_1.linear_1.activations_output
            + mlp_1.act.activations_output
            + mlp_1.activations_output
        )
        activations += post_mlp1_add.activations_output
        activations += norm_2.activations_output
        activations += (
            mlp_2.linear_1.activations_output
            + mlp_2.act.activations_output
            + mlp_2.activations_output
        )
        activations += post_mlp2_add.activations_output

        super().__init__(activations, layers, config)

        self.norm_1 = norm_1
        self.mlp_1 = mlp_1
        self.post_mlp1_add = post_mlp1_add
        self.norm_2 = norm_2
        self.mlp_2 = mlp_2
        self.post_mlp2_add = post_mlp2_add

    @staticmethod
    def generate_simple(
        x: TensorMoments, config: MlpMixerConfig,
    ) -> "MixerBlock":
        eps = config.layer_norm_epsilon
        norm_1 = LayerNorm.generate_simple(x, 2, eps)
        mlp_1 = MixerMlp.generate_simple(norm_1.y, 'R')
        post_mlp1_add = Add.generate_simple(x, mlp_1.y)
        norm_2 = LayerNorm.generate_simple(
            post_mlp1_add.activations_output[0], 2, eps,
        )
        mlp_2 = MixerMlp.generate_simple(norm_2.y, 'L')
        post_mlp2_add = Add.generate_simple(
            post_mlp1_add.activations_output[0], mlp_2.y,
        )
        return MixerBlock(
            x, norm_1, mlp_1, post_mlp1_add, norm_2, mlp_2, post_mlp2_add,
            config,
        )

    @staticmethod
    def from_torch(
        torch_mixer: "TorchMixer",
        x: TensorMoments,
        config: MlpMixerConfig,
    ) -> "MixerBlock":
        norm_1 = LayerNorm.from_torch(torch_mixer.norm_1, x)
        mlp_1 = MixerMlp.generate_simple(norm_1.y, 'R')
        post_mlp1_add = Add.generate_simple(x, mlp_1.y)
        norm_2 = LayerNorm.from_torch(
            torch_mixer.norm_2, post_mlp1_add.activations_output[0],
        )
        mlp_2 = MixerMlp.generate_simple(norm_2.y, 'L')
        post_mlp2_add = Add.generate_simple(
            post_mlp1_add.activations_output[0], mlp_2.y,
        )
        block = MixerBlock(
            x, norm_1, mlp_1, post_mlp1_add, norm_2, mlp_2, post_mlp2_add,
            config,
        )
        _copy_torch_mixer_weights(torch_mixer, block)
        return block

    def to_torch(self) -> "TorchMixer":
        from nntile.torch_models.mlp_mixer import Mixer as TorchMixer

        torch_mixer = TorchMixer(
            self.config.channel_dim, self.config.projected_patch_dim,
        )
        _copy_block_weights_to_torch(self, torch_mixer)
        return torch_mixer

    def to_torch_with_grads(self) -> "TorchMixer":
        torch_mixer = self.to_torch()
        for p_nntile, p_torch in zip(
            self.parameters, torch_mixer.parameters(),
        ):
            if p_nntile.grad is None:
                continue
            grad_np = to_numpy(p_nntile.grad)
            if _weight_layout_transposed(
                p_nntile.grad.shape, tuple(p_torch.shape),
            ):
                grad_np = grad_np.T
            p_torch.grad = torch.tensor(grad_np, requires_grad=False)
        return torch_mixer


def _weight_layout_transposed(
    nntile_shape: tuple, torch_shape: tuple,
) -> bool:
    return (
        len(nntile_shape) == 2
        and len(torch_shape) == 2
        and nntile_shape[0] == torch_shape[1]
        and nntile_shape[1] == torch_shape[0]
    )


def _copy_param_to_torch(
    p_nntile: TensorMoments, p_torch: torch.Tensor,
) -> None:
    nntile_np = np.zeros(p_nntile.value.shape, dtype=np.float32, order="F")
    p_nntile.value.to_array(nntile_np)
    if _weight_layout_transposed(nntile_np.shape, tuple(p_torch.shape)):
        p_torch.data.copy_(torch.from_numpy(nntile_np.T))
    else:
        p_torch.data.copy_(torch.from_numpy(nntile_np))


def _copy_param_from_torch(
    p_torch: torch.Tensor, p_nntile: TensorMoments,
) -> None:
    weight_np = p_torch.detach().cpu().numpy()
    if _weight_layout_transposed(
        p_nntile.value.shape, weight_np.shape,
    ):
        p_nntile.value.from_array(weight_np.T)
    else:
        p_nntile.value.from_array(weight_np)


def _copy_torch_mixer_weights(
    torch_mixer: "TorchMixer", block: MixerBlock,
) -> None:
    block.norm_1.gamma.value.from_array(
        torch_mixer.norm_1.weight.detach().cpu().numpy()
    )
    block.norm_1.beta.value.from_array(
        torch_mixer.norm_1.bias.detach().cpu().numpy()
    )
    block.norm_2.gamma.value.from_array(
        torch_mixer.norm_2.weight.detach().cpu().numpy()
    )
    block.norm_2.beta.value.from_array(
        torch_mixer.norm_2.bias.detach().cpu().numpy()
    )
    _copy_param_from_torch(
        torch_mixer.mlp_1.fn[0].weight, block.mlp_1.linear_1.w,
    )
    _copy_param_from_torch(
        torch_mixer.mlp_1.fn[2].weight, block.mlp_1.linear_2.w,
    )
    _copy_param_from_torch(
        torch_mixer.mlp_2.fn[0].weight, block.mlp_2.linear_1.w,
    )
    _copy_param_from_torch(
        torch_mixer.mlp_2.fn[2].weight, block.mlp_2.linear_2.w,
    )


def _copy_block_weights_to_torch(
    block: MixerBlock, torch_mixer: "TorchMixer",
) -> None:
    for p_nntile, p_torch in zip(block.parameters, torch_mixer.parameters()):
        _copy_param_to_torch(p_nntile, p_torch)
