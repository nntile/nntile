# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/t5_ff.py
# T5LayerFF submodule of NNTile Python package
#
# @version 1.1.0

import torch
from transformers.models.t5.modeling_t5 import (
    T5Config as T5ConfigTorch, T5LayerFF as T5LayerFFTorch)

from nntile.layer.act import Act
from nntile.layer.add import Add
from nntile.layer.linear import Linear
from nntile.layer.prod import Prod
from nntile.layer.rms_norm import RMSNorm
from nntile.model.base_model import BaseModel
from nntile.model.t5_config import T5ConfigNNTile
from nntile.tensor import TensorMoments, notrans, to_numpy


class T5DenseGatedActDense(BaseModel):
    next_tag: int

    def __init__(self, x: TensorMoments, config: T5ConfigNNTile, next_tag: int):
        activations = [x]
        layers = []
        self.d_model = config.d_model
        self.d_model_tile = config.d_model_tile
        self.d_ff = config.d_ff
        self.d_ff_tile = config.d_ff_tile
        self.redux = config.redux

        assert config.dropout_rate == 0

        gemm_ndim = 1

        wi_0, next_tag = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.d_ff],
            [self.d_ff_tile],
            next_tag,
            redux=self.redux,
            bias=False,
        )

        layers.append(wi_0)
        activations.extend(wi_0.activations_output)

        act_fn_layer, next_tag = Act.generate_simple(
            activations[-1], config.dense_act_fn, next_tag
        )

        layers.append(act_fn_layer)
        activations.extend(act_fn_layer.activations_output)

        wi_1, next_tag = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.d_ff],
            [self.d_ff_tile],
            next_tag,
            redux=self.redux,
            bias=False,
        )

        layers.append(wi_1)
        activations.extend(wi_1.activations_output)

        prod_layer, next_tag = Prod.generate_simple(
            activations[-1], activations[-2], next_tag
        )

        layers.append(prod_layer)
        activations.extend(prod_layer.activations_output)

        wo, next_tag = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            gemm_ndim,
            [self.d_model],
            [self.d_model_tile],
            next_tag,
            redux=self.redux,
            bias=False,
        )

        # all layers for direct access
        self.wi_0 = wi_0
        self.wi_1 = wi_1
        self.act_fn_layer = act_fn_layer
        self.prod_layer = prod_layer
        self.wo = wo

        layers.append(wo)
        activations.extend(wo.activations_output)
        self.next_tag = next_tag

        super().__init__(activations, layers, config)

    @classmethod
    def from_torch(
        cls, torch_ff_layer, x: TensorMoments, config: T5ConfigNNTile, next_tag: int
    ):
        t5_ff_nntile = cls(x, config, next_tag)

        torch_params = list(torch_ff_layer.parameters())
        for i, p in enumerate(t5_ff_nntile.parameters):
            p.value.from_array(torch_params[i].cpu().detach().numpy())
        return t5_ff_nntile, t5_ff_nntile.next_tag


class T5LayerFF(BaseModel):
    next_tag: int

    def __init__(
        self,
        x: TensorMoments,
        dense_relu_dense: T5DenseGatedActDense,
        rms_norm: RMSNorm,
        residual_add: Add,
        config: T5ConfigNNTile,
        next_tag: int,
    ):
        assert config.is_gated_act

        layers = [rms_norm] + dense_relu_dense.layers + [residual_add]
        activations = (
            [x]
            + rms_norm.activations_output
            + dense_relu_dense.activations[1:]
            + residual_add.activations_output
        )

        self.next_tag = next_tag

        super().__init__(activations, layers, config)

    @classmethod
    def from_torch(
        cls, torch_ff_layer, x: TensorMoments, config: T5ConfigNNTile, next_tag: int
    ):
        rms_norm, next_tag = RMSNorm.from_torch(
            torch_ff_layer.layer_norm,
            x,
            0,
            config.layer_norm_epsilon,
            next_tag,
            redux=config.redux,
        )

        assert config.is_gated_act
        dense_relu_dense, next_tag = T5DenseGatedActDense.from_torch(
            torch_ff_layer.DenseReluDense,
            rms_norm.activations_output[0],
            config,
            next_tag,
        )

        residual_add, next_tag = Add.generate_simple(
            x, dense_relu_dense.activations[-1], next_tag
        )

        return (
            cls(x, dense_relu_dense, rms_norm, residual_add, config, next_tag),
            next_tag,
        )

    def to_torch(self):
        """Convert NNTile T5LayerFF to PyTorch T5LayerFF"""
        torch_config = T5ConfigTorch(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            dropout_rate=0.0,
            dense_act_fn="gelu_new",
            is_gated_act=True,
        )
        torch_layer = T5LayerFFTorch(torch_config)

        # Copy parameters from NNTile to PyTorch
        torch_params = list(torch_layer.parameters())
        torch_params = torch_params[-1:] + torch_params[:-1]
        for p_nntile, p_torch in zip(self.parameters, torch_params):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value), requires_grad=True)

        return torch_layer

    def to_torch_with_grads(self):
        """Convert NNTile T5LayerFF to PyTorch T5LayerFF with gradients"""
        torch_config = T5ConfigTorch(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            dropout_rate=0.0,
            dense_act_fn="gelu_new",
            is_gated_act=True,
        )
        torch_layer = T5LayerFFTorch(torch_config)

        # Copy parameters and gradients from NNTile to PyTorch
        torch_params = list(torch_layer.parameters())
        torch_params = torch_params[-1:] + torch_params[:-1]
        for p_nntile, p_torch in zip(self.parameters, torch_params):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value), requires_grad=True)
            p_torch.grad = torch.tensor(to_numpy(p_nntile.grad))

        return torch_layer
