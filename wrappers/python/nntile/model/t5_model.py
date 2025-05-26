# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/t5_model.py
# T5 Model of NNTile Python package
#
# @version 1.1.0
# ruff: noqa: E501

import copy

from transformers.models.t5.modeling_t5 import (
    T5Config as T5ConfigTorch,
    T5ForConditionalGeneration as T5ForConditionalGenerationTorch,
    T5ForSequenceClassification as T5ForSequenceClassificationTorch,
    T5Model as T5ModelTorch)

import nntile
from nntile.layer.linear import Linear
from nntile.model.base_model import BaseModel
from nntile.model.t5_block import T5Stack
from nntile.model.t5_config import T5ConfigNNTile, T5EncoderDecoderConfig
from nntile.model.t5_lmhead import T5ClassificationHead
from nntile.tensor import (
    Tensor_bf16, Tensor_fp32, Tensor_fp32_fast_bf16, Tensor_fp32_fast_fp16,
    Tensor_fp32_fast_tf32, TensorMoments)


class T5Model(BaseModel):
    def __init__(
        self,
        x: TensorMoments,
        decoder_x: TensorMoments,
        encoder: T5Stack,
        decoder: T5Stack,
        encoder_config: T5ConfigNNTile,
        decoder_config: T5ConfigNNTile,
    ):
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        self.x = x
        self.decoder_x = decoder_x

        self.encoder = encoder
        self.decoder = decoder

        activations = (
            [x]
            + self.encoder.activations[1:]
            + [decoder_x]
            + self.decoder.activations[1:]
        )
        layers = self.encoder.layers + self.decoder.layers

        super().__init__(
            activations, layers, T5EncoderDecoderConfig(encoder_config, decoder_config)
        )

    @classmethod
    def from_torch(
        cls,
        torch_model: T5ModelTorch,
        x: TensorMoments,
        decoder_x: TensorMoments,
        config: T5ConfigNNTile,
        next_tag: int = 0,
    ):
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True

        encoder, next_tag = T5Stack.from_torch(
            torch_model.encoder, x, encoder_config, next_tag=next_tag
        )
        encoder_output = encoder.activations[-1]
        decoder, next_tag = T5Stack.from_torch(
            torch_model.decoder,
            decoder_x,
            decoder_config,
            next_tag=next_tag,
            encoder_output=encoder_output,
        )

        return (
            cls(x, decoder_x, encoder, decoder, encoder_config, decoder_config),
            next_tag,
        )

    def to_torch(self):
        """Convert NNTile T5Model to PyTorch T5Model"""
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.encoder_config.d_model,
            d_ff=self.encoder_config.d_ff,
            num_layers=self.encoder_config.num_layers,
            num_decoder_layers=self.decoder_config.num_layers,
            num_heads=self.encoder_config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.encoder_config.layer_norm_epsilon,
            is_gated_act=True,
            is_encoder_decoder=True,
        )

        # Create PyTorch model
        torch_model = T5ModelTorch(torch_config)

        # Convert encoder and decoder
        torch_model.encoder = self.encoder.to_torch()
        torch_model.decoder = self.decoder.to_torch()

        return torch_model


class T5ForSequenceClassification(BaseModel):
    def __init__(
        self,
        x: TensorMoments,
        decoder_x: TensorMoments,
        embedding_layer,
        embedding_layer_decoder,
        transformer: T5Model,
        lm_head: T5ClassificationHead,
        next_tag: int,
    ):
        self.embedding = embedding_layer
        self.embedding_decoder = embedding_layer_decoder
        self.transformer = transformer
        self.classification_head = lm_head

        activations = (
            [x, decoder_x]
            + [self.embedding.activations_output[0]]
            + transformer.activations[1:]
            + lm_head.activations[1:]
        )
        layers = (
            [self.embedding, self.embedding_decoder]
            + transformer.layers
            + lm_head.layers
        )

        super().__init__(activations, layers, transformer.config)

    @classmethod
    def from_torch(
        cls,
        torch_model: T5ForSequenceClassificationTorch,
        x: TensorMoments,
        decoder_x: TensorMoments,
        config: T5ConfigNNTile,
        next_tag: int = 0,
    ):
        dtype2tensor_type = {
            "fp32": Tensor_fp32,
            "bf16": Tensor_bf16,
            "fp32_fast_tf32": Tensor_fp32_fast_tf32,
            "fp32_fast_fp16": Tensor_fp32_fast_fp16,
            "fp32_fast_bf16": Tensor_fp32_fast_bf16,
        }

        tensor_type = dtype2tensor_type[config.dtype]

        embedding_layer, next_tag = nntile.layer.embedding.Embedding.from_torch(
            torch_model.transformer.shared,
            x,
            next_tag,
            dtype=tensor_type,
            embedding_tile_size=config.d_model_tile,
        )
        transformer, next_tag = T5Model.from_torch(
            torch_model.transformer,
            embedding_layer.activations_output[0],
            embedding_layer.activations_output[0],
            config,
            next_tag,
        )
        lm_head, next_tag = T5ClassificationHead.from_torch(
            torch_model.classification_head,
            transformer.activations[-1],
            config,
            torch_model.config.num_labels,
            next_tag,
        )

        return (
            cls(
                x,
                decoder_x,
                embedding_layer,
                embedding_layer,
                transformer,
                lm_head,
                next_tag,
            ),
            next_tag,
        )

    def to_torch(self):
        """Convert NNTile T5ForSequenceClassification
        to PyTorch T5ForSequenceClassification
        """
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.transformer.encoder_config.d_model,
            d_ff=self.transformer.encoder_config.d_ff,
            num_layers=self.transformer.encoder_config.num_layers,
            num_decoder_layers=self.transformer.decoder_config.num_layers,
            num_heads=self.transformer.encoder_config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.transformer.encoder_config.layer_norm_epsilon,
            is_gated_act=True,
            is_encoder_decoder=True,
            num_labels=self.classification_head.num_labels,
            decoder_start_token_id=0,
            pad_token_id=0,
        )

        # Create PyTorch model
        torch_model = T5ForSequenceClassificationTorch(torch_config)

        # Convert transformer and classification head
        torch_model.transformer = self.transformer.to_torch()
        torch_model.transformer.shared = self.embedding.to_torch()
        torch_model.transformer.encoder.embed_tokens = torch_model.transformer.shared
        torch_model.transformer.decoder.embed_tokens = torch_model.transformer.shared

        torch_model.classification_head = self.classification_head.to_torch()

        return torch_model


class T5ForConditionalGeneration(BaseModel):
    def __init__(
        self,
        x: TensorMoments,
        decoder_x: TensorMoments,
        embedding_layer,
        embedding_layer_decoder,
        transformer: T5Model,
        lm_head: Linear,
        next_tag: int,
    ):
        self.embedding = embedding_layer
        self.embedding_decoder = embedding_layer_decoder
        self.transformer = transformer
        self.lm_head = lm_head

        activations = (
            [x, decoder_x]
            + [self.embedding.activations_output[0]]
            + transformer.activations[1:]
            + [lm_head.activations_output[0]]
        )
        layers = (
            [self.embedding, self.embedding_decoder]
            + transformer.layers
            + [lm_head]
        )

        super().__init__(activations, layers, transformer.config)

    @classmethod
    def from_torch(
        cls,
        torch_model: T5ForConditionalGenerationTorch,
        x: TensorMoments,
        decoder_x: TensorMoments,
        config: T5ConfigNNTile,
        next_tag: int = 0,
    ):
        dtype2tensor_type = {
            "fp32": Tensor_fp32,
            "bf16": Tensor_bf16,
            "fp32_fast_tf32": Tensor_fp32_fast_tf32,
            "fp32_fast_fp16": Tensor_fp32_fast_fp16,
            "fp32_fast_bf16": Tensor_fp32_fast_bf16,
        }

        tensor_type = dtype2tensor_type[config.dtype]

        embedding_layer, next_tag = nntile.layer.embedding.Embedding.from_torch(
            torch_model.shared,
            x,
            next_tag,
            dtype=tensor_type,
            embedding_tile_size=config.d_model_tile,
        )
        transformer, next_tag = T5Model.from_torch(
            torch_model,
            embedding_layer.activations_output[0],
            embedding_layer.activations_output[0],
            config,
            next_tag,
        )
        print(f"transformer.activations[-1].value.basetile_shape: {transformer.activations[-1].value.basetile_shape}")
        lm_head, next_tag = Linear.from_torch(
            torch_model.lm_head,
            transformer.activations[-1],
            torch_model.lm_head.out_features,
            config.redux,
            next_tag,
        )

        return (
            cls(
                x,
                decoder_x,
                embedding_layer,
                embedding_layer,
                transformer,
                lm_head,
                next_tag,
            ),
            next_tag,
        )

    def to_torch(self):
        """Convert NNTile T5ForConditionalGeneration
        to PyTorch T5ForConditionalGeneration
        """
        # Create PyTorch config
        torch_config = T5ConfigTorch(
            d_model=self.transformer.encoder_config.d_model,
            d_ff=self.transformer.encoder_config.d_ff,
            num_layers=self.transformer.encoder_config.num_layers,
            num_decoder_layers=self.transformer.decoder_config.num_layers,
            num_heads=self.transformer.encoder_config.n_head,
            dropout_rate=0.0,
            layer_norm_epsilon=self.transformer.encoder_config.layer_norm_epsilon,
            is_gated_act=True,
            is_encoder_decoder=True,
            decoder_start_token_id=0,
            pad_token_id=0,
        )

        # Create PyTorch model
        torch_model = T5ForConditionalGenerationTorch(torch_config)

        # Convert transformer and classification head
        transformer = self.transformer.to_torch()
        torch_model.encoder = transformer.encoder
        torch_model.decoder = transformer.decoder
        torch_model.shared = self.embedding.to_torch()
        torch_model.encoder.embed_tokens = torch_model.shared
        torch_model.decoder.embed_tokens = torch_model.shared
        torch_model.lm_head = self.lm_head.to_torch()

        return torch_model
