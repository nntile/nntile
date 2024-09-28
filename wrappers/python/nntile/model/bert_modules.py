# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/bert_modules.py
# Bert modules for entire Bert model as a part of NNTile Python package
#
# @version 1.1.0

import numpy as np
import torch
from transformers.models.bert.modeling_bert import (
    BertConfig as BertConfig_torch, BertEmbeddings as BertEmbeddings_torch,
    BertSelfOutput as BertSelfOutput_torch)

from nntile.tensor import (
    Tensor_bf16, Tensor_fp32, Tensor_fp32_fast_bf16, Tensor_fp32_fast_fp16,
    Tensor_fp32_fast_tf32, Tensor_int64, TensorMoments, TensorTraits, to_numpy)

from ..layer.add import Add
from ..layer.add_slice import AddSlice
from ..layer.embedding import Embedding
from ..layer.layer_norm import LayerNorm
from ..layer.linear import Linear
from .base_model import BaseModel
from .bert_config import BertConfigNNTile


class BertEmbeddings(BaseModel):
    next_tag: int

    def __init__(self, input_ids: TensorMoments,
                  token_type_ids: TensorMoments,
                  pos_ids: TensorMoments,
                  word_embed: Embedding,
                  pos_embed: Embedding,
                  token_type_embed: Embedding,
                  add_input_token_type: AddSlice,
                  add_pos_embed: AddSlice,
                  layer_norm: LayerNorm,
                  config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config

        activations = [input_ids, token_type_ids, pos_ids]
        activations.extend(word_embed.activations_output)
        activations.extend(pos_embed.activations_output)
        activations.extend(token_type_embed.activations_output)
        activations.extend(add_input_token_type.activations_output)
        activations.extend(add_pos_embed.activations_output)
        activations.extend(layer_norm.activations_output)

        layers = [word_embed,
                  pos_embed,
                  token_type_embed,
                  add_input_token_type,
                  add_pos_embed,
                  layer_norm]

        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    @staticmethod
    def from_torch(bert_embedding_torch, batch_size, batch_size_tile,
                   seq_len, seq_len_tile, config: BertConfigNNTile,
                   next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        positional_ids_traits = TensorTraits([seq_len], [seq_len_tile])
        positional_ids_distr = [0] * positional_ids_traits.grid.nelems
        positional_ids_value = Tensor_int64(
            positional_ids_traits, positional_ids_distr, next_tag
        )
        next_tag = positional_ids_value.next_tag
        positional_ids_value.from_array(
            np.array(np.arange(seq_len), order="F", dtype=np.int64)
        )
        positional_ids = TensorMoments(positional_ids_value, None, False)

        x_shape = [seq_len, batch_size]
        x_basetile = [seq_len_tile, batch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_distr = [0] * x_traits.grid.nelems
        x_value = Tensor_int64(x_traits, x_distr, 0)
        X = TensorMoments(x_value, None, False)

        dtype2tensor_type = {"fp32": Tensor_fp32,
                            "bf16": Tensor_bf16,
                            "fp32_fast_tf32": Tensor_fp32_fast_tf32,
                            "fp32_fast_fp16": Tensor_fp32_fast_fp16,
                            "fp32_fast_bf16": Tensor_fp32_fast_bf16
                            }

        tensor_type = dtype2tensor_type[config.dtype]

        word_embed_layer, next_tag = Embedding.generate_simple(
                                x_value,
                                tensor_type,
                                0,
                                config.vocab_size,
                                config.hidden_size,
                                config.hidden_size_tile,
                                config.vocab_embed_dim_tile,
                                next_tag)

        pos_embed_layer, next_tag = Embedding.generate_simple(
                                positional_ids.value,
                                tensor_type,
                                0,
                                config.max_position_embeddings,
                                config.hidden_size,
                                config.hidden_size_tile,
                                config.vocab_embed_dim_tile,
                                next_tag)

        token_type_ids_traits = TensorTraits([seq_len], [seq_len_tile])
        token_type_ids_distr = [0] * token_type_ids_traits.grid.nelems
        token_type_ids_value = Tensor_int64(
            token_type_ids_traits, token_type_ids_distr, next_tag
        )
        next_tag = token_type_ids_value.next_tag
        token_type_ids_value.from_array(
            np.array(np.zeros((seq_len,)), order="F", dtype=np.int64)
        )
        token_type_ids = TensorMoments(token_type_ids_value, None, False)

        token_type_embed_layer, next_tag = Embedding.generate_simple(
                                token_type_ids.value,
                                tensor_type,
                                0,
                                config.type_vocab_size,
                                config.hidden_size,
                                config.hidden_size_tile,
                                config.vocab_embed_dim_tile,
                                next_tag)

        word_embed_layer.w.value.from_array(
            bert_embedding_torch.word_embeddings.weight.cpu().detach().numpy().T
        )
        pos_embed_layer.w.value.from_array(
            bert_embedding_torch.position_embeddings.weight.cpu().detach().numpy().T
        )
        token_type_embed_layer.w.value.from_array(
            bert_embedding_torch.token_type_embeddings.weight.cpu().detach().numpy().T
        )

        add_slice_wt_layer, next_tag = AddSlice.generate_simple(
                word_embed_layer.activations_output[0],
                token_type_embed_layer.activations_output[0],
                2, next_tag, redux=config.redux
            )
        add_slice_pos_layer, next_tag = AddSlice.generate_simple(
                add_slice_wt_layer.activations_output[0],
                pos_embed_layer.activations_output[0],
                2, next_tag, redux=config.redux
            )
        lnorm, next_tag = LayerNorm.from_torch(
                                bert_embedding_torch.LayerNorm,
                                add_slice_pos_layer.activations_output[0],
                                next_tag, config.redux)

        bert_embedding_nntile = BertEmbeddings(X,
                                                token_type_ids,
                                                positional_ids,
                                                word_embed_layer,
                                                pos_embed_layer,
                                                token_type_embed_layer,
                                                add_slice_wt_layer,
                                                add_slice_pos_layer,
                                                lnorm, config)
        return bert_embedding_nntile, next_tag

    def to_torch(self):
        config_torch = BertConfig_torch()
        config_torch.vocab_size = self.config.vocab_size
        config_torch.hidden_size = self.config.hidden_size
        config_torch.max_position_embeddings = \
                        self.config.max_position_embeddings
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.type_vocab_size = self.config.type_vocab_size
        config_torch.hidden_dropout_prob = 0.

        bert_embeddings_torch = BertEmbeddings_torch(config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_embeddings_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value).T,
                                        requires_grad=True)
        return bert_embeddings_torch

    def to_torch_with_grads(self):
        bert_embeddins_torch = self.to_torch()
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_embeddins_torch.parameters()):
            p_torch.grad = torch.tensor(to_numpy(p_nntile.grad).T)
        return bert_embeddins_torch


class BertSelfOutput(BaseModel):
    next_tag: int

    def __init__(self, hidden_states: TensorMoments,
                  input_tensor: TensorMoments,
                  lin_layer: Linear,
                  add_layer: Add,
                  layer_norm: LayerNorm,
                  config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config

        activations = [hidden_states, input_tensor]
        activations.extend(lin_layer.activations_output)
        activations.extend(add_layer.activations_output)
        activations.extend(layer_norm.activations_output)

        layers = [lin_layer,
                  add_layer,
                  layer_norm]

        # Fill Base Model with the generated data
        super().__init__(activations, layers)

    @staticmethod
    def from_torch(bert_selfoutput_torch, batch_size, batch_size_tile,
                   seq_len, seq_len_tile, hidden_dim, hidden_dim_tile,
                   config: BertConfigNNTile, next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        dtype2tensor_type = {"fp32": Tensor_fp32,
                            "bf16": Tensor_bf16,
                            "fp32_fast_tf32": Tensor_fp32_fast_tf32,
                            "fp32_fast_fp16": Tensor_fp32_fast_fp16,
                            "fp32_fast_bf16": Tensor_fp32_fast_bf16
                            }
        tensor_type = dtype2tensor_type[config.dtype]

        x_shape = [hidden_dim, seq_len, batch_size]
        x_basetile = [hidden_dim_tile, seq_len_tile, batch_size_tile]
        x_traits = TensorTraits(x_shape, x_basetile)
        x_distr = [0] * x_traits.grid.nelems
        x_value = tensor_type(x_traits, x_distr, 0)
        x_grad = tensor_type(x_traits, x_distr, 0)
        X = TensorMoments(x_value, x_grad, True)

        input_tensor_traits = TensorTraits(x_shape, x_basetile)
        input_tensor_distr = [0] * input_tensor_traits.grid.nelems
        input_tensor_value = tensor_type(input_tensor_traits,
                                         input_tensor_distr, 0)
        input_tensor_grad = tensor_type(input_tensor_traits,
                                        input_tensor_distr, 0)
        input_tensor = TensorMoments(input_tensor_value,
                                     input_tensor_grad,
                                     True)

        lin_layer, next_tag = Linear.from_torch(bert_selfoutput_torch.dense, X,
                                                hidden_dim_tile,
                                                config.redux, next_tag)

        add_layer, next_tag = Add.generate_simple(
                                lin_layer.activations_output[0],
                                input_tensor,
                                next_tag)
        lnorm, next_tag = LayerNorm.from_torch(
                                bert_selfoutput_torch.LayerNorm,
                                add_layer.activations_output[0],
                                next_tag, config.redux)

        bert_selfoutput_nntile = BertSelfOutput(X,
                                                input_tensor,
                                                lin_layer, add_layer,
                                                lnorm, config)
        return bert_selfoutput_nntile, next_tag

    def to_torch(self):
        config_torch = BertConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.hidden_dropout_prob = 0.

        bert_selfoutput_torch = BertSelfOutput_torch(config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_selfoutput_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value),
                                        requires_grad=True)
        return bert_selfoutput_torch

    def to_torch_with_grads(self):
        bert_selfoutput_torch = self.to_torch()
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_selfoutput_torch.parameters()):
            p_torch.grad = torch.tensor(to_numpy(p_nntile.grad))
        return bert_selfoutput_torch
