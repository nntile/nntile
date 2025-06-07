# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/t5_block.py
# T5 Block of NNTile Python package
#
# @version 1.1.0
# ruff: noqa: E501

from typing import Optional

import numpy as np
from transformers.models.t5.modeling_t5 import (
    T5Block as T5BlockTorch, T5Config as T5ConfigTorch,
    T5LayerCrossAttention as T5LayerCrossAttentionTorch,
    T5LayerSelfAttention as T5LayerSelfAttentionTorch, T5Stack as T5StackTorch)

import nntile.utils.constructors as nntc
from nntile.layer.add import Add
from nntile.layer.rms_norm import RMSNorm
from nntile.layer.t5_attention import T5Attention
from nntile.model.base_model import BaseModel
from nntile.model.t5_config import T5ConfigNNTile
from nntile.model.t5_ff import T5LayerFF
from nntile.tensor import Tensor, TensorMoments


class T5LayerSelfAttention(BaseModel):
    attention: T5Attention
    layer_norm: RMSNorm
    add: Add

    def __init__(
        self,
        x: TensorMoments,
        attention: T5Attention,
        layer_norm: RMSNorm,
        add: Add,
        config: T5ConfigNNTile,
    ):
        self.attention = attention
        self.layer_norm = layer_norm
        self.add = add

        layers = [layer_norm, attention, add]
        activations = (
            [x]
            + layer_norm.activations_output
            + attention.activations_output
            + add.activations_output
        )
        super().__init__(activations, layers, config)

    @classmethod
    def from_torch(
        cls,
        torch_layer: T5LayerSelfAttentionTorch,
        x: TensorMoments,
        config: T5ConfigNNTile,
        attention_mask: Tensor = None,
    ):
        layer_norm = RMSNorm.from_torch(
            torch_layer.layer_norm,
            x,
            0,
            config.layer_norm_epsilon,
            redux=config.redux,
        )

        attention = T5Attention.from_torch(
            torch_layer.SelfAttention,
            layer_norm.activations_output[0],
            attention_mask,
            config,
        )
        add = Add.generate_simple(
            x, attention.activations_output[0]
        )
        layer = cls(x, attention, layer_norm, add, config)
        return layer

    def to_torch(self):
        """Convert NNTile T5LayerSelfAttention
        to PyTorch T5LayerSelfAttention"""
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            num_heads=self.config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            is_gated_act=self.config.is_gated_act,
            is_encoder_decoder=True,
        )

        # Create PyTorch layer
        torch_layer = T5LayerSelfAttentionTorch(torch_config)

        # Convert components
        torch_layer.layer_norm = self.layer_norm.to_torch()
        torch_layer.SelfAttention = self.attention.to_torch()

        return torch_layer


class T5LayerCrossAttention(BaseModel):
    cross_attention: T5Attention
    layer_norm: RMSNorm
    add: Add

    def __init__(
        self,
        x: TensorMoments,
        encoder_output: TensorMoments,
        attention: T5Attention,
        layer_norm: RMSNorm,
        add: Add,
        config: T5ConfigNNTile,
    ):
        self.attention = attention
        self.layer_norm = layer_norm
        self.add = add

        layers = [layer_norm, attention, add]
        activations = (
            [x, encoder_output]
            + layer_norm.activations_output
            + attention.activations_output
            + add.activations_output
        )
        super().__init__(activations, layers, config)

    @classmethod
    def from_torch(
        cls,
        torch_layer: T5LayerCrossAttentionTorch,
        x: TensorMoments,
        encoder_output: TensorMoments,
        config: T5ConfigNNTile,
    ):
        layer_norm = RMSNorm.from_torch(
            torch_layer.layer_norm,
            x,
            0,
            config.layer_norm_epsilon,
            redux=config.redux,
        )
        attention = T5Attention.from_torch(
            torch_layer.EncDecAttention,
            layer_norm.activations_output[0],
            None,
            config,
            encoder_output=encoder_output,
        )
        add = Add.generate_simple(
            x, attention.activations_output[0]
        )
        layer = cls(x, encoder_output, attention, layer_norm, add, config)
        return layer

    def to_torch(self):
        """Convert NNTile T5LayerCrossAttention
        to PyTorch T5LayerCrossAttention"""
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            num_heads=self.config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            is_gated_act=self.config.is_gated_act,
            is_encoder_decoder=True,
        )

        # Create PyTorch layer
        torch_layer = T5LayerCrossAttentionTorch(torch_config)

        # Convert components
        torch_layer.layer_norm = self.layer_norm.to_torch()
        torch_layer.EncDecAttention = self.attention.to_torch()

        return torch_layer


class T5Block(BaseModel):
    is_decoder: bool
    attention: T5LayerSelfAttention
    cross_attention: Optional[T5LayerCrossAttention]
    feed_forward: T5LayerFF

    def __init__(
        self,
        x: TensorMoments,
        attention: T5LayerSelfAttention,
        feed_forward: T5LayerFF,
        config: T5ConfigNNTile,
        cross_attention=None,
    ):
        assert (
            not config.is_decoder or cross_attention is not None
        ), "Cross attention must be provided for decoder blocks"

        self.is_decoder = config.is_decoder
        self.attention = attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        layers = attention.layers
        if cross_attention is not None:
            layers.extend(cross_attention.layers)
        layers.extend(feed_forward.layers)

        activations = [x]
        activations.extend(attention.activations[1:])
        if cross_attention is not None:
            activations.extend(cross_attention.activations)
        activations.extend(feed_forward.activations[1:])

        super().__init__(activations, layers, config)

    @classmethod
    def from_torch(
        cls,
        torch_block: T5BlockTorch,
        x: TensorMoments,
        config: T5ConfigNNTile,
        attention_mask: Tensor = None,
        encoder_output: TensorMoments = None,
    ):
        attention = T5LayerSelfAttention.from_torch(
            torch_block.layer[0], x, config, attention_mask=attention_mask
        )
        cross_attention = (
            T5LayerCrossAttention.from_torch(
                torch_block.layer[1],
                attention.activations[-1],
                encoder_output,
                config,
            )
            if config.is_decoder
            else None
        )
        ff_layer_torch = (
            torch_block.layer[2] if config.is_decoder else torch_block.layer[1]
        )
        ff_input = (
            cross_attention.activations[-1]
            if cross_attention is not None
            else attention.activations[-1]
        )
        feed_forward = T5LayerFF.from_torch(
            ff_layer_torch, ff_input, config
        )
        block = cls(x, attention, feed_forward, config, cross_attention=cross_attention)
        return block

    def to_torch(self):
        """Convert NNTile T5Block to PyTorch T5Block"""
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            num_heads=self.config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            is_gated_act=self.config.is_gated_act,
            is_encoder_decoder=True,
            is_decoder=self.is_decoder,
        )

        # Create PyTorch block
        torch_block = T5BlockTorch(torch_config)

        # Convert layers
        torch_block.layer[0] = self.attention.to_torch()
        if self.is_decoder:
            torch_block.layer[1] = self.cross_attention.to_torch()
            torch_block.layer[2] = self.feed_forward.to_torch()
        else:
            torch_block.layer[1] = self.feed_forward.to_torch()

        return torch_block


class T5Stack(BaseModel):
    blocks: list[T5Block]
    final_layer_norm: RMSNorm

    def __init__(
        self,
        x: TensorMoments,
        blocks: list[T5Block],
        final_layer_norm: RMSNorm,
        config: T5ConfigNNTile,
    ):
        self.blocks = blocks
        self.final_layer_norm = final_layer_norm

        activations = [x]
        for block in blocks:
            activations.extend(block.activations[1:])
        activations.append(final_layer_norm.activations_output[0])
        layers = []
        for block in blocks:
            layers.extend(block.layers)
        layers.append(final_layer_norm)
        super().__init__(activations, layers, config)

    @classmethod
    def from_torch(
        cls,
        torch_stack: T5StackTorch,
        x: TensorMoments,
        config: T5ConfigNNTile,
        encoder_output: TensorMoments = None,
    ):
        attention_mask = None
        if config.is_decoder:
            attention_mask_np = np.tril(
                np.ones((x.value.shape[1], x.value.shape[1]), dtype=bool), k=0
            )
            attention_mask = nntc.from_array(
                attention_mask_np.T,
                basetile_shape=(x.value.basetile_shape[1], x.value.basetile_shape[1]),
            )

        blocks = []
        next_inp = x
        for layer_idx in range(len(torch_stack.block)):
            torch_block = torch_stack.block[layer_idx]
            block = T5Block.from_torch(
                torch_block,
                next_inp,
                config,
                encoder_output=encoder_output,
                attention_mask=attention_mask,
            )
            if layer_idx > 0:
                # this is temporal crutch to be compatible with hugginface model
                # TODO: just store embeddings outside and pass to constructor
                block.attention.attention.has_relative_bias = True
                block.attention.attention.relative_bias = blocks[
                    0
                ].attention.attention.relative_bias
                block.attention.attention.relative_bias_embedding = blocks[
                    0
                ].attention.attention.relative_bias_embedding
            blocks.append(block)
            next_inp = block.activations[-1]

        final_layer_norm = RMSNorm.from_torch(
            torch_stack.final_layer_norm,
            next_inp,
            0,
            config.layer_norm_epsilon,
            redux=config.redux,
        )

        stack = cls(x, blocks, final_layer_norm, config)
        return stack

    def to_torch(self):
        """Convert NNTile T5Stack to PyTorch T5Stack"""
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            num_layers=len(self.blocks),
            num_heads=self.config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
            is_gated_act=self.config.is_gated_act,
            is_encoder_decoder=True,
            is_decoder=self.config.is_decoder,
            use_cache=False if not self.config.is_decoder else True,
        )

        # Create PyTorch stack
        torch_stack = T5StackTorch(torch_config)

        # Convert blocks
        for i, block in enumerate(self.blocks):
            torch_block = block.to_torch()
            torch_stack.block[i] = torch_block

        # Convert final layer norm
        torch_stack.final_layer_norm = self.final_layer_norm.to_torch()

        return torch_stack
