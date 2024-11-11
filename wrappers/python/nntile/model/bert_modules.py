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
    BertAttention as BertAttention_torch, BertConfig as BertConfig_torch,
    BertEmbeddings as BertEmbeddings_torch,
    BertIntermediate as BertIntermediate_torch,
    BertLMPredictionHead as BertLMPredictionHead_torch,
    BertOutput as BertOutput_torch,
    BertPredictionHeadTransform as BertPredictionHeadTransform_torch,
    BertSelfOutput as BertSelfOutput_torch)

from nntile.tensor import (
    Tensor_bf16, Tensor_fp32, Tensor_fp32_fast_bf16, Tensor_fp32_fast_fp16,
    Tensor_fp32_fast_tf32, Tensor_int64, TensorMoments, TensorTraits, notrans,
    to_numpy)

from ..layer.act import Act
from ..layer.add import Add
from ..layer.add_slice import AddSlice
from ..layer.bert_selfattention import BertSelfAttention
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

        self.word_embed = word_embed
        self.pos_embed = pos_embed
        self.token_type_embed = token_type_embed
        self.l_norm = layer_norm

        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

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
        config_torch.pad_token_id = None

        bert_embeddings_torch = BertEmbeddings_torch(config_torch)
        bert_embeddings_torch.word_embeddings = self.word_embed.to_torch()
        bert_embeddings_torch.position_embeddings = self.pos_embed.to_torch()
        bert_embeddings_torch.token_type_embeddings = \
            self.token_type_embed.to_torch()
        bert_embeddings_torch.LayerNorm = self.l_norm.to_torch()
        return bert_embeddings_torch

    def to_torch_with_grads(self):
        config_torch = BertConfig_torch()
        config_torch.vocab_size = self.config.vocab_size
        config_torch.hidden_size = self.config.hidden_size
        config_torch.max_position_embeddings = \
                        self.config.max_position_embeddings
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.type_vocab_size = self.config.type_vocab_size
        config_torch.hidden_dropout_prob = 0.
        config_torch.pad_token_id = None

        bert_embeddings_torch = BertEmbeddings_torch(config_torch)
        bert_embeddings_torch.word_embeddings = \
            self.word_embed.to_torch_with_grads()
        bert_embeddings_torch.position_embeddings = \
            self.pos_embed.to_torch_with_grads()
        bert_embeddings_torch.token_type_embeddings = \
            self.token_type_embed.to_torch_with_grads()
        bert_embeddings_torch.LayerNorm = self.l_norm.to_torch_with_grads()
        return bert_embeddings_torch


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
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(bert_selfoutput_torch,
                   X,
                   input_tensor,
                   hidden_dim,
                   hidden_dim_tile,
                   config: BertConfigNNTile, next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        lin_layer, next_tag = Linear.generate_simple(X,
                                                     "R",
                                                     notrans,
                                                     2,
                                                     [hidden_dim],
                                                     [hidden_dim_tile],
                                                     next_tag)
        w_torch = bert_selfoutput_torch.dense.weight.data

        target_w_shape = lin_layer.w.value.shape
        w_torch = w_torch.reshape((target_w_shape[0],
                         target_w_shape[1],
                         target_w_shape[2]))
        lin_layer.w.value.from_array(w_torch.numpy())
        b_torch = bert_selfoutput_torch.dense.bias.data.numpy()
        lin_layer.b.value.from_array(b_torch)

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
        for p_nntile, (n, p_torch) in zip(self.parameters,
                                    bert_selfoutput_torch.named_parameters()):
            submodule_name = n.split(".")[0]
            parameter_name = n.split(".")[1]
            if parameter_name == "weight" and submodule_name == "dense":
                nntile_weight = torch.tensor(to_numpy(p_nntile.value))
                # nntile_weight = nntile_weight.transpose(1, 2)
                target_shape = bert_selfoutput_torch.dense.weight.shape
                nntile_weight = nntile_weight.reshape(target_shape)
                p_torch.data = nntile_weight.clone().detach()
                p_torch.data.requires_grad_(True)
            else:
                p_torch.data = torch.tensor(to_numpy(p_nntile.value),
                                        requires_grad=True)
        return bert_selfoutput_torch

    def to_torch_with_grads(self):
        bert_selfoutput_torch = self.to_torch()
        for p_nntile, (n, p_torch) in zip(self.parameters,
                                    bert_selfoutput_torch.named_parameters()):
            submodule_name = n.split(".")[0]
            parameter_name = n.split(".")[1]
            if parameter_name == "weight" and submodule_name == "dense":
                nntile_grad = torch.tensor(to_numpy(p_nntile.grad))
                # nntile_grad = nntile_grad.transpose(1, 2)
                target_shape = bert_selfoutput_torch.dense.weight.shape
                nntile_grad = nntile_grad.reshape(target_shape)
                p_torch.grad = nntile_grad.clone().detach()
            else:
                p_torch.grad = torch.tensor(to_numpy(p_nntile.grad))
        return bert_selfoutput_torch


class BertIntermediate(BaseModel):
    next_tag: int

    def __init__(self, hidden_states: TensorMoments,
                  lin_layer: Linear,
                  activation_layer: Act,
                  config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config

        activations = [hidden_states]
        activations.extend(lin_layer.activations_output)
        activations.extend(activation_layer.activations_output)

        layers = [lin_layer,
                  activation_layer]

        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(bert_intermediate_torch, X,
                   intermediate_size_tile,
                   config: BertConfigNNTile, next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        lin_layer, next_tag = Linear.from_torch(bert_intermediate_torch.dense,
                                                X,
                                                intermediate_size_tile,
                                                config.redux, next_tag)

        activation_layer, next_tag = Act.generate_simple(
            lin_layer.activations_output[0],
            config.activation_function, next_tag
        )

        bert_intermediate_nntile = BertIntermediate(X, lin_layer,
                                                    activation_layer, config)
        return bert_intermediate_nntile, next_tag

    def to_torch(self):
        config_torch = BertConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.hidden_dropout_prob = 0.

        bert_intermediate_torch = BertIntermediate_torch(config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_intermediate_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value),
                                        requires_grad=True)
        return bert_intermediate_torch

    def to_torch_with_grads(self):
        bert_intermediate_torch = self.to_torch()
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_intermediate_torch.parameters()):
            p_torch.grad = torch.tensor(to_numpy(p_nntile.grad))
        return bert_intermediate_torch


class BertOutput(BaseModel):
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
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(bert_output_torch, X, input_tensor,
                   hidden_size_tile,
                   config: BertConfigNNTile, next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        lin_layer, next_tag = Linear.from_torch(bert_output_torch.dense, X,
                                                hidden_size_tile,
                                                config.redux, next_tag)

        add_layer, next_tag = Add.generate_simple(
                                lin_layer.activations_output[0],
                                input_tensor,
                                next_tag)
        lnorm, next_tag = LayerNorm.from_torch(
                                bert_output_torch.LayerNorm,
                                add_layer.activations_output[0],
                                next_tag, config.redux)

        bert_output_nntile = BertOutput(X,
                                        input_tensor,
                                        lin_layer, add_layer,
                                        lnorm, config)
        return bert_output_nntile, next_tag

    def to_torch(self):
        config_torch = BertConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.hidden_dropout_prob = 0.

        bert_output_torch = BertOutput_torch(config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_output_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value),
                                        requires_grad=True)
        return bert_output_torch

    def to_torch_with_grads(self):
        bert_output_torch = self.to_torch()
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_output_torch.parameters()):
            p_torch.grad = torch.tensor(to_numpy(p_nntile.grad))
        return bert_output_torch


class BertAttention(BaseModel):
    next_tag: int

    def __init__(self, hidden_states: TensorMoments,
                  attention: BertSelfAttention,
                  self_output_layer: BertSelfOutput,
                  config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config
        self.self_attention = attention
        self.selfoutput = self_output_layer

        activations = [hidden_states]
        activations.extend(attention.activations_output)
        activations.extend(self_output_layer.activations[2:])

        layers = [attention] + self_output_layer.layers

        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(bert_attention_torch, X,
                   config: BertConfigNNTile, next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        selfattention_layer, next_tag = BertSelfAttention.from_torch(
            bert_attention_torch.self, X, X, X, config, next_tag)

        selfoutput_layer, next_tag = BertSelfOutput.from_torch(
            bert_attention_torch.output,
            selfattention_layer.activations_output[0],
            X,
            config.hidden_size,
            config.hidden_size_tile,
            config, next_tag)
        bert_attention_nntile = BertAttention(X,
                                              selfattention_layer,
                                              selfoutput_layer,
                                              config)
        return bert_attention_nntile, next_tag

    def to_torch(self):
        config_torch = BertConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.num_attention_heads = self.config.num_attention_heads
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.hidden_dropout_prob = 0.
        config_torch.attention_probs_dropout_prob = 0.
        bert_attention_torch = BertAttention_torch(config_torch)
        bert_attention_torch.self = self.self_attention.to_torch()
        bert_attention_torch.output = self.selfoutput.to_torch()
        return bert_attention_torch

    def to_torch_with_grads(self):
        config_torch = BertConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.num_attention_heads = self.config.num_attention_heads
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.hidden_dropout_prob = 0.
        config_torch.attention_probs_dropout_prob = 0.
        bert_attention_torch = BertAttention_torch(config_torch)
        bert_attention_torch.self = self.self_attention.to_torch_with_grads()
        bert_attention_torch.output = self.selfoutput.to_torch_with_grads()

        return bert_attention_torch


class BertPredictionHeadTransform(BaseModel):
    next_tag: int

    def __init__(self, hidden_states: TensorMoments,
                  lin_layer: Linear,
                  act_layer: Act,
                  layer_norm: LayerNorm,
                  config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config

        activations = [hidden_states]
        activations.extend(lin_layer.activations_output)
        activations.extend(act_layer.activations_output)
        activations.extend(layer_norm.activations_output)

        layers = [lin_layer,
                  act_layer,
                  layer_norm]

        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(bert_pred_head_trans_torch, X,
                   hidden_size_tile,
                   config: BertConfigNNTile, next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        lin_layer, next_tag = Linear.from_torch(
                                    bert_pred_head_trans_torch.dense, X,
                                    hidden_size_tile,
                                    config.redux, next_tag)

        activation_layer, next_tag = Act.generate_simple(
            lin_layer.activations_output[0],
            config.activation_function, next_tag
        )
        lnorm, next_tag = LayerNorm.from_torch(
                                bert_pred_head_trans_torch.LayerNorm,
                                activation_layer.activations_output[0],
                                next_tag, config.redux)

        bert_output_nntile = BertPredictionHeadTransform(X,
                                        lin_layer, activation_layer,
                                        lnorm, config)
        return bert_output_nntile, next_tag

    def to_torch(self):
        config_torch = BertConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.hidden_dropout_prob = 0.
        config_torch.hidden_act = "gelu_pytorch_tanh"

        bert_pred_head_transform_torch = \
            BertPredictionHeadTransform_torch(config_torch)
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_pred_head_transform_torch.parameters()):
            p_torch.data = torch.tensor(to_numpy(p_nntile.value),
                                        requires_grad=True)
        return bert_pred_head_transform_torch

    def to_torch_with_grads(self):
        bert_pred_head_transform_torch = self.to_torch()
        for p_nntile, p_torch in zip(self.parameters,
                                    bert_pred_head_transform_torch.parameters()):
            p_torch.grad = torch.tensor(to_numpy(p_nntile.grad))
        return bert_pred_head_transform_torch


class BertLMPredictionHead(BaseModel):
    next_tag: int

    def __init__(self,
                 transform: BertPredictionHeadTransform,
                 lin_layer: Linear,
                 config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config

        activations = []
        activations.extend(transform.activations)
        activations.extend(lin_layer.activations_output)

        layers = []
        layers.extend(transform.layers)
        layers.append(lin_layer)
        self.transform = transform
        self.lin_decoder = lin_layer

        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(bert_lm_pred_head, X,
                   config: BertConfigNNTile, next_tag: int):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        transform, next_tag = BertPredictionHeadTransform.from_torch(
                                        bert_lm_pred_head.transform,
                                        X, config.hidden_size_tile,
                                        config, next_tag)
        lin_layer, next_tag = Linear.from_torch(
                                    bert_lm_pred_head.decoder,
                                    transform.activations[-1],
                                    config.vocab_size,
                                    config.redux, next_tag)

        bert_output_nntile = BertLMPredictionHead(transform,
                                            lin_layer, config)
        return bert_output_nntile, next_tag

    def to_torch(self):
        config_torch = BertConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.vocab_size = self.config.vocab_size
        config_torch.hidden_act = "gelu_pytorch_tanh"

        bert_lm_pred_head_torch = \
            BertLMPredictionHead_torch(config_torch)
        bert_lm_pred_head_torch.transform = self.transform.to_torch()
        bert_lm_pred_head_torch.decoder = self.lin_decoder.to_torch()
        bert_lm_pred_head_torch.bias = torch.nn.Parameter(
            torch.tensor(to_numpy(self.lin_decoder.b.value),
                        requires_grad=True))
        return bert_lm_pred_head_torch

    def to_torch_with_grads(self):
        config_torch = BertConfig_torch()
        config_torch.hidden_size = self.config.hidden_size
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.vocab_size = self.config.vocab_size
        config_torch.hidden_act = self.config.activation_function

        bert_lm_pred_head_torch = \
            BertLMPredictionHead_torch(config_torch)
        bert_lm_pred_head_torch.transform = \
            self.transform.to_torch_with_grads()
        bert_lm_pred_head_torch.decoder = \
            self.lin_decoder.to_torch_with_grads()
        bert_lm_pred_head_torch.bias.grad = torch.tensor(
            to_numpy(self.lin_decoder.b.grad))
        return bert_lm_pred_head_torch
