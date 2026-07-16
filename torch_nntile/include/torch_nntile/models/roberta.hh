/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/roberta.hh
 * RoBERTa MLM for device=nntile (pad-aware positions; NNGraph roberta).
 */

#pragma once

#include <torch_nntile/models/bert.hh>

#include <string>

namespace torch_nntile::models
{

struct RobertaConfig
{
    int64_t vocab_size = 50265;
    int64_t hidden_size = 64;
    int64_t num_hidden_layers = 2;
    int64_t num_attention_heads = 2;
    int64_t intermediate_size = 128;
    int64_t max_position_embeddings = 514;
    int64_t type_vocab_size = 1;
    int64_t pad_token_id = 1;
    double layer_norm_eps = 1e-5;
    std::string hidden_act = "gelu";

    BertConfig to_bert_config() const
    {
        BertConfig cfg;
        cfg.vocab_size = vocab_size;
        cfg.hidden_size = hidden_size;
        cfg.num_hidden_layers = num_hidden_layers;
        cfg.num_attention_heads = num_attention_heads;
        cfg.intermediate_size = intermediate_size;
        cfg.max_position_embeddings = max_position_embeddings;
        cfg.type_vocab_size = type_vocab_size;
        cfg.layer_norm_eps = layer_norm_eps;
        cfg.hidden_act = hidden_act;
        cfg.pad_token_id = pad_token_id;
        return cfg;
    }
};

//! RoBERTa MLM: pad-aware positions + BertLayer encoder + LM head.
struct RobertaMlmImpl : torch::nn::Module
{
    RobertaConfig config;
    torch::nn::Embedding word_embeddings{nullptr};
    torch::nn::Embedding position_embeddings{nullptr};
    torch::nn::Embedding token_type_embeddings{nullptr};
    torch::nn::LayerNorm emb_ln{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::Linear lm_dense{nullptr};
    torch::nn::LayerNorm lm_ln{nullptr};
    torch::nn::Linear lm_decoder{nullptr};
    bool gelu_tanh = false;

    explicit RobertaMlmImpl(RobertaConfig cfg);
    torch::Tensor forward(
        torch::Tensor input_ids,
        torch::Tensor token_type_ids);
};

TORCH_MODULE(RobertaMlm);

} // namespace torch_nntile::models
