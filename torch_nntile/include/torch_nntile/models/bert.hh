/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/bert.hh
 * Tiny BERT MLM for device=nntile.
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace torch_nntile::models
{

struct BertConfig
{
    int64_t vocab_size = 30522;
    int64_t hidden_size = 64;
    int64_t num_hidden_layers = 2;
    int64_t num_attention_heads = 2;
    int64_t intermediate_size = 128;
    int64_t max_position_embeddings = 128;
    int64_t type_vocab_size = 2;
    double layer_norm_eps = 1e-12;
};

struct BertMlmImpl : torch::nn::Module
{
    BertConfig config;
    torch::nn::Embedding word_embeddings{nullptr};
    torch::nn::Embedding position_embeddings{nullptr};
    torch::nn::Embedding token_type_embeddings{nullptr};
    torch::nn::LayerNorm emb_ln{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::Linear cls{nullptr};

    explicit BertMlmImpl(BertConfig cfg);
    torch::Tensor forward(
        torch::Tensor input_ids,
        torch::Tensor token_type_ids);
};

TORCH_MODULE(BertMlm);

} // namespace torch_nntile::models
