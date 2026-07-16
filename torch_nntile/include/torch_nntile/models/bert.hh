/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/bert.hh
 * BERT MLM for device=nntile (LibTorch port of NNGraph bert).
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>

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
    //! ``gelu`` (exact) or ``gelu_pytorch_tanh``.
    std::string hidden_act = "gelu";
    //! ``< 0``: BERT ``0..S-1`` positions; ``>= 0``: pad-aware ids.
    int64_t pad_token_id = -1;
};

//! Absolute position ids: BERT arange or RoBERTa pad-skipping.
//! Built on host when needed, then uploaded (nntile lacks long arange/ne).
torch::Tensor bert_position_ids_from_input_ids(
    torch::Tensor const& input_ids,
    int64_t pad_token_id);

//! Encoder block: post-norm attention + FFN (NNGraph BertLayer).
struct BertLayerImpl : torch::nn::Module
{
    torch::nn::Linear query{nullptr};
    torch::nn::Linear key{nullptr};
    torch::nn::Linear value{nullptr};
    torch::nn::Linear attn_dense{nullptr};
    torch::nn::LayerNorm attn_ln{nullptr};
    torch::nn::Linear intermediate{nullptr};
    torch::nn::Linear output_dense{nullptr};
    torch::nn::LayerNorm output_ln{nullptr};
    int64_t n_head = 0;
    int64_t hidden = 0;
    int64_t head_dim = 0;
    bool gelu_tanh = false;

    explicit BertLayerImpl(BertConfig const& cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BertLayer);

struct BertMlmImpl : torch::nn::Module
{
    BertConfig config;
    torch::nn::Embedding word_embeddings{nullptr};
    torch::nn::Embedding position_embeddings{nullptr};
    torch::nn::Embedding token_type_embeddings{nullptr};
    torch::nn::LayerNorm emb_ln{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::Linear transform_dense{nullptr};
    torch::nn::LayerNorm transform_ln{nullptr};
    torch::nn::Linear decoder{nullptr};
    bool gelu_tanh = false;
    //! Cached host-built index tables uploaded to device.
    torch::Tensor cached_pos_;
    torch::Tensor cached_tt_;
    int64_t cache_batch_ = -1;
    int64_t cache_seq_ = -1;

    explicit BertMlmImpl(BertConfig cfg);
    torch::Tensor forward(
        torch::Tensor input_ids,
        torch::Tensor token_type_ids);
};

TORCH_MODULE(BertMlm);

} // namespace torch_nntile::models
