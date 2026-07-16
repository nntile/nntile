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
    //! ``< 0``: BERT ``0..S-1`` positions; ``>= 0``: RoBERTa pad-aware ids.
    int64_t pad_token_id = -1;
};

//! Encoder block shared by BERT and RoBERTa LibTorch stacks.
struct BertLayerImpl : torch::nn::Module
{
    torch::nn::LayerNorm ln1{nullptr};
    torch::nn::Linear qkv{nullptr};
    torch::nn::Linear out{nullptr};
    torch::nn::LayerNorm ln2{nullptr};
    torch::nn::Linear ff_in{nullptr};
    torch::nn::Linear ff_out{nullptr};
    int64_t n_head = 0;
    int64_t hidden = 0;

    explicit BertLayerImpl(BertConfig const& cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BertLayer);

//! Absolute position ids: BERT arange or RoBERTa pad-skipping.
torch::Tensor bert_position_ids_from_input_ids(
    torch::Tensor const& input_ids,
    int64_t pad_token_id);

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
