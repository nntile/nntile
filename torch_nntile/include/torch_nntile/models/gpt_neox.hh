/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/gpt_neox.hh
 * GPT-NeoX causal LM matching deleted ``nntile::model::gptneox``.
 */

#pragma once

#include <torch/torch.h>
#include <torch_nntile/classic_nn.hh>

#include <cstdint>

namespace torch_nntile::models
{

struct GptNeoXConfig
{
    int64_t vocab_size = 50280;
    int64_t hidden_size = 64;
    int64_t intermediate_size = 256;
    int64_t num_hidden_layers = 2;
    int64_t num_attention_heads = 2;
    int64_t max_position_embeddings = 128;
    double layer_norm_eps = 1e-5;
    double rotary_pct = 0.25;
    double rotary_emb_base = 10000.0;
    bool use_parallel_residual = true;
    bool attention_bias = true;
};

struct GptNeoXAttentionImpl : torch::nn::Module
{
    int64_t n_heads = 0;
    int64_t head_size = 0;
    int64_t hidden = 0;
    int64_t rotary_ndims = 0;
    bool attention_bias = true;
    torch::Tensor q_weight;
    torch::Tensor k_weight;
    torch::Tensor v_weight;
    torch::Tensor o_weight;
    torch::Tensor q_bias;
    torch::Tensor k_bias;
    torch::Tensor v_bias;
    torch::Tensor o_bias;

    explicit GptNeoXAttentionImpl(GptNeoXConfig const &cfg);
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor const &sin,
        torch::Tensor const &cos,
        torch::Tensor const &mask);
};

TORCH_MODULE(GptNeoXAttention);

struct GptNeoXMLPImpl : torch::nn::Module
{
    torch::Tensor fc1_weight;
    torch::Tensor fc1_bias;
    torch::Tensor fc2_weight;
    torch::Tensor fc2_bias;

    explicit GptNeoXMLPImpl(GptNeoXConfig const &cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(GptNeoXMLP);

struct GptNeoXDecoderImpl : torch::nn::Module
{
    nn_classic::LayerNorm input_norm{nullptr};
    GptNeoXAttention attn{nullptr};
    nn_classic::LayerNorm post_attn_norm{nullptr};
    GptNeoXMLP mlp{nullptr};
    bool parallel_residual = true;

    explicit GptNeoXDecoderImpl(GptNeoXConfig const &cfg);
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor const &sin,
        torch::Tensor const &cos,
        torch::Tensor const &mask);
};

TORCH_MODULE(GptNeoXDecoder);

struct GptNeoXCausalImpl : torch::nn::Module
{
    GptNeoXConfig config;
    nn_classic::Embedding embed_in{nullptr};
    torch::nn::ModuleList layers{nullptr};
    nn_classic::LayerNorm final_layer_norm{nullptr};
    torch::Tensor lm_weight;
    torch::Tensor rope_sin_;
    torch::Tensor rope_cos_;
    torch::Tensor cached_mask_;
    int64_t rope_cache_batch_ = -1;
    int64_t rope_cache_seq_ = -1;

    explicit GptNeoXCausalImpl(GptNeoXConfig cfg);
    void warm_rope_cache(
        int64_t batch,
        int64_t seq,
        torch::Device device);
    torch::Tensor forward(torch::Tensor input_ids);
};

TORCH_MODULE(GptNeoXCausal);

} // namespace torch_nntile::models
