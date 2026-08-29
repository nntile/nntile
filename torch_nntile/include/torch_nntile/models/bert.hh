/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/bert.hh
 * BERT MLM matching deleted ``nntile::model::bert`` (not HF ATen).
 */

#pragma once

#include <torch/torch.h>
#include <torch_nntile/classic_nn.hh>

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

    int64_t head_dim() const
    {
        return hidden_size / num_attention_heads;
    }
};

torch::Tensor bert_position_ids_from_input_ids(
    torch::Tensor const &input_ids,
    int64_t pad_token_id);

//! ``BertSelfAttention``: gemm -> transpose(1) -> add_fiber -> sdpa ->
//! transpose(3).
struct BertSelfAttentionImpl : torch::nn::Module
{
    int64_t n_heads = 0;
    int64_t head_size = 0;
    int64_t hidden = 0;
    torch::Tensor q_weight;
    torch::Tensor k_weight;
    torch::Tensor v_weight;
    torch::Tensor q_bias;
    torch::Tensor k_bias;
    torch::Tensor v_bias;

    explicit BertSelfAttentionImpl(BertConfig const &cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BertSelfAttention);

//! ``BertSelfOutput``: gemm(ndim=2) + residual + LayerNorm.
struct BertSelfOutputImpl : torch::nn::Module
{
    torch::Tensor dense_weight;
    torch::Tensor dense_bias;
    nn_classic::LayerNorm ln{nullptr};

    explicit BertSelfOutputImpl(BertConfig const &cfg);
    torch::Tensor forward(
        torch::Tensor attn_heads,
        torch::Tensor residual);
};

TORCH_MODULE(BertSelfOutput);

struct BertAttentionImpl : torch::nn::Module
{
    BertSelfAttention self{nullptr};
    BertSelfOutput output{nullptr};

    explicit BertAttentionImpl(BertConfig const &cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BertAttention);

struct BertLayerImpl : torch::nn::Module
{
    BertAttention attention{nullptr};
    torch::Tensor inter_weight;
    torch::Tensor inter_bias;
    torch::Tensor out_weight;
    torch::Tensor out_bias;
    nn_classic::LayerNorm out_ln{nullptr};
    bool gelu_tanh = false;

    explicit BertLayerImpl(BertConfig const &cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(BertLayer);

struct BertMlmImpl : torch::nn::Module
{
    BertConfig config;
    nn_classic::Embedding word_embeddings{nullptr};
    nn_classic::Embedding position_embeddings{nullptr};
    nn_classic::Embedding token_type_embeddings{nullptr};
    nn_classic::LayerNorm emb_ln{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::Tensor transform_weight;
    torch::Tensor transform_bias;
    nn_classic::LayerNorm transform_ln{nullptr};
    torch::Tensor decoder_weight;
    torch::Tensor decoder_bias;
    bool gelu_tanh = false;
    torch::Tensor cached_pos_;
    int64_t cache_batch_ = -1;
    int64_t cache_seq_ = -1;

    explicit BertMlmImpl(BertConfig cfg);
    torch::Tensor forward(
        torch::Tensor input_ids,
        torch::Tensor token_type_ids);
};

TORCH_MODULE(BertMlm);

} // namespace torch_nntile::models
