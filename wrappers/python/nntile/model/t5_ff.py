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
# ruff: noqa: E501

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

    def __init__(self, x: TensorMoments, config: T5ConfigNNTile):
        activations = [x]
        layers = []
        self.d_model = config.d_model
        self.d_model_tile = config.d_model_tile
        self.d_ff = config.d_ff
        self.d_ff_tile = config.d_ff_tile
        self.redux = config.redux

        assert config.dropout_rate == 0

        gemm_ndim = 1

        wi_0 = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.d_ff],
            [self.d_ff_tile],
            redux=self.redux,
            bias=False,
        )

        layers.append(wi_0)
        activations.extend(wi_0.activations_output)

        act_fn_layer = Act.generate_simple(
            activations[-1], config.dense_act_fn
        )

        layers.append(act_fn_layer)
        activations.extend(act_fn_layer.activations_output)

        wi_1 = Linear.generate_simple(
            x,
            "R",
            notrans,
            gemm_ndim,
            [self.d_ff],
            [self.d_ff_tile],
            redux=self.redux,
            bias=False,
        )

        layers.append(wi_1)
        activations.extend(wi_1.activations_output)

        prod_layer = Prod.generate_simple(
            activations[-1], activations[-2]
        )

        layers.append(prod_layer)
        activations.extend(prod_layer.activations_output)

        wo = Linear.generate_simple(
            activations[-1],
            "R",
            notrans,
            gemm_ndim,
            [self.d_model],
            [self.d_model_tile],
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

        super().__init__(activations, layers, config)

    @classmethod
    def from_torch(
        cls, torch_ff_layer, x: TensorMoments, config: T5ConfigNNTile
    ):
        t5_ff_nntile = cls(x, config)

        torch_params = list(torch_ff_layer.parameters())
        for i, p in enumerate(t5_ff_nntile.parameters):
            p.value.from_array(torch_params[i].cpu().detach().numpy())
        return t5_ff_nntile


class T5LayerFF(BaseModel):

    def __init__(
        self,
        x: TensorMoments,
        dense_relu_dense: T5DenseGatedActDense,
        rms_norm: RMSNorm,
        residual_add: Add,
        config: T5ConfigNNTile,
    ):
        assert config.is_gated_act

        layers = [rms_norm] + dense_relu_dense.layers + [residual_add]
        activations = (
            [x]
            + rms_norm.activations_output
            + dense_relu_dense.activations[1:]
            + residual_add.activations_output
        )

        super().__init__(activations, layers, config)

    @classmethod
    def from_torch(
        cls, torch_ff_layer, x: TensorMoments, config: T5ConfigNNTile
    ):
        rms_norm = RMSNorm.from_torch(
            torch_ff_layer.layer_norm,
            x,
            0,
            config.layer_norm_epsilon,
            redux=config.redux,
        )

        assert config.is_gated_act
        dense_relu_dense = T5DenseGatedActDense.from_torch(
            torch_ff_layer.DenseReluDense,
            rms_norm.activations_output[0],
            config,
        )

        residual_add = Add.generate_simple(
            x, dense_relu_dense.activations[-1]
        )

        return (
            cls(x, dense_relu_dense, rms_norm, residual_add, config)
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
