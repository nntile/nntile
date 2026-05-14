# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/nntile/model/roberta.py
# Roberta related models as a part of NNTile Python package
#
# @version 1.1.0

from transformers.models.roberta.modeling_roberta import (
    RobertaConfig as RobertaConfig_torch,
    RobertaForMaskedLM as RobertaForMaskedLM_torch,
    RobertaModel as RobertaModel_torch)

from .base_model import BaseModel
from .bert_config import BertConfigNNTile
from .bert_encoder import BertEncoder
from .bert_modules import BertEmbeddings
from .roberta_modules import RobertaLMHead


class RobertaModel(BaseModel):

    def __init__(self,
                  embed_layer: BertEmbeddings,
                  bert_encoder: BertEncoder,
                  config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config
        layers = []
        layers.extend(embed_layer.layers)
        layers.extend(bert_encoder.layers)
        activations = []
        activations.extend(embed_layer.activations)
        activations.extend(bert_encoder.activations[1:])
        self.bert_embed = embed_layer
        self.bert_encoder = bert_encoder

        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(bert_torch, batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   config: BertConfigNNTile):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        bert_embedding = BertEmbeddings.from_torch(
                                                    bert_torch.embeddings,
                                                    batch_size,
                                                    batch_size_tile,
                                                    seq_len,
                                                    seq_len_tile,
                                                    config)
        bert_encoder = BertEncoder.from_torch(bert_torch.encoder,
                                              bert_embedding.activations[-1],
                                              config)
        bert_model_nntile = RobertaModel(bert_embedding,
                                      bert_encoder,
                                      config)
        return bert_model_nntile

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        seq_len: int,
        batch_size: int = 1,
        seq_len_tile: int = None,
        batch_size_tile: int = None,
        hidden_size_tile: int = None,
        vocab_embed_dim_tile: int = None,
        intermediate_size_tile: int = None,
        n_head_tile: int = None,
        dtype: str = "fp32",
        cache_dir: str = None,
    ):
        """Load a HuggingFace RobertaModel checkpoint and convert to NNTile.

        Same shape as BertModel.from_pretrained: encoder-only, outputs a
        (hidden_size, seq_len, batch_size) hidden state at
        self.activations[-1]. No encoder attention mask is plumbed
        through; heavily padded inputs degrade quality."""
        roberta_torch = RobertaModel_torch.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=False,
        )
        roberta_torch.eval()
        tc = roberta_torch.config

        config = BertConfigNNTile(
            vocab_size=tc.vocab_size,
            vocab_embed_dim_tile=vocab_embed_dim_tile or tc.hidden_size,
            hidden_size=tc.hidden_size,
            hidden_size_tile=hidden_size_tile or tc.hidden_size,
            intermediate_size=tc.intermediate_size,
            intermediate_size_tile=(
                intermediate_size_tile or tc.intermediate_size),
            num_attention_heads=tc.num_attention_heads,
            n_head_tile=n_head_tile or tc.num_attention_heads,
            activation_function=tc.hidden_act,
            layer_norm_epsilon=tc.layer_norm_eps,
            type_vocab_size=tc.type_vocab_size,
            max_position_embeddings=tc.max_position_embeddings,
            num_hidden_layers=tc.num_hidden_layers,
            dtype=dtype,
        )
        return cls.from_torch(
            roberta_torch, batch_size, batch_size_tile or batch_size,
            seq_len, seq_len_tile or seq_len, config)

    def _make_default_torch_model(self):
        config_torch = RobertaConfig_torch()
        config_torch.vocab_size = self.config.vocab_size
        config_torch.hidden_size = self.config.hidden_size
        config_torch.max_position_embeddings = \
                        self.config.max_position_embeddings
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.type_vocab_size = self.config.type_vocab_size
        config_torch.hidden_dropout_prob = 0.
        config_torch.attention_probs_dropout_prob = 0.
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.num_attention_heads = self.config.num_attention_heads
        config_torch._attn_implementation = "eager"
        config_torch.add_cross_attention = False
        config_torch.chunk_size_feed_forward = 0
        config_torch.is_decoder = False
        config_torch.num_hidden_layers = self.config.num_hidden_layers
        config_torch.pad_token_id = None

        roberta_model_torch = RobertaModel_torch(config_torch,
                                           add_pooling_layer=False)

        return roberta_model_torch

    def to_torch(self):

        roberta_model_torch = self._make_default_torch_model()
        roberta_model_torch.embeddings = self.bert_embed.to_torch()
        roberta_model_torch.encoder = self.bert_encoder.to_torch()
        return roberta_model_torch

    def to_torch_with_grads(self):

        roberta_model_torch = self._make_default_torch_model()
        roberta_model_torch.embeddings = self.bert_embed.to_torch_with_grads()
        roberta_model_torch.encoder = self.bert_encoder.to_torch_with_grads()

        return roberta_model_torch


class RobertaForMaskedLM(BaseModel):

    def __init__(self,
                  roberta: RobertaModel,
                  lm_head: RobertaLMHead,
                  config: BertConfigNNTile):

        self.dtype = config.dtype

        self.config = config
        layers = []
        layers.extend(roberta.layers)
        layers.extend(lm_head.layers)
        activations = []
        activations.extend(roberta.activations)
        activations.extend(lm_head.activations[1:])
        self.roberta = roberta
        self.lm_head = lm_head

        # Fill Base Model with the generated data
        super().__init__(activations, layers, config)

    @staticmethod
    def from_torch(bert_for_masked_lm_torch,
                   batch_size, batch_size_tile,
                   seq_len, seq_len_tile,
                   config: BertConfigNNTile):

        if config.dtype not in ["fp32", "fp32_fast_tf32", "bf16",
                            "fp32_fast_fp16", "fp32_fast_bf16"]:
            raise TypeError("Only fp32, fp32_fast_tf32, bf16,"
            "fp32_fast_fp16, and fp32_fast_bf16 supported for weight type")

        roberta = RobertaModel.from_torch(
                                            bert_for_masked_lm_torch.roberta,
                                            batch_size,
                                            batch_size_tile,
                                            seq_len,
                                            seq_len_tile,
                                            config)
        lm_head = RobertaLMHead.from_torch(
                            bert_for_masked_lm_torch.lm_head,
                            roberta.activations[-1],
                            config)
        bert_model_nntile = RobertaForMaskedLM(roberta,
                                      lm_head,
                                      config)
        return bert_model_nntile

    def _make_default_torch_model(self):
        config_torch = RobertaConfig_torch()
        config_torch.vocab_size = self.config.vocab_size
        config_torch.hidden_size = self.config.hidden_size
        config_torch.max_position_embeddings = \
                        self.config.max_position_embeddings
        config_torch.layer_norm_eps = self.config.layer_norm_epsilon
        config_torch.type_vocab_size = self.config.type_vocab_size
        config_torch.hidden_dropout_prob = 0.
        config_torch.attention_probs_dropout_prob = 0.
        config_torch.intermediate_size = self.config.intermediate_size
        config_torch.num_attention_heads = self.config.num_attention_heads
        config_torch._attn_implementation = "eager"
        config_torch.add_cross_attention = False
        config_torch.chunk_size_feed_forward = 0
        config_torch.is_decoder = False
        config_torch.num_hidden_layers = self.config.num_hidden_layers
        config_torch.pad_token_id = None

        bert_layer_torch = RobertaForMaskedLM_torch(config_torch)
        return bert_layer_torch

    def to_torch(self):

        roberta_model_torch = self._make_default_torch_model()
        roberta_model_torch.roberta = self.roberta.to_torch()
        roberta_model_torch.lm_head = self.lm_head.to_torch()
        return roberta_model_torch

    def to_torch_with_grads(self):

        roberta_model_torch = self._make_default_torch_model()
        roberta_model_torch.roberta = self.roberta.to_torch_with_grads()
        roberta_model_torch.lm_head = self.lm_head.to_torch_with_grads()

        return roberta_model_torch
