/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/gpt_neo.hh
 * GPT-Neo causal LM matching deleted ``nntile::model::gptneo``.
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace torch_nntile::models
{

struct GptNeoConfig
{
    int64_t vocab_size = 50257;
    int64_t hidden_size = 64;
    int64_t intermediate_size = 256;
    int64_t num_hidden_layers = 2;
    int64_t num_attention_heads = 2;
    int64_t max_position_embeddings = 128;
    int64_t window_size = 256;
    double layer_norm_eps = 1e-5;
    //! ``"global"`` / ``"local"`` per layer; empty -> HF alternate default.
    std::vector<std::string> attention_layers;

    bool is_local_layer(int64_t layer_id) const
    {
        if (attention_layers.empty())
        {
            return (layer_id % 2) == 1;
        }
        return attention_layers.at(
                   static_cast<std::size_t>(layer_id)) == "local";
    }
};

struct GptNeoAttentionImpl : torch::nn::Module
{
    int64_t n_heads = 0;
    int64_t head_size = 0;
    int64_t hidden = 0;
    bool local = false;
    int64_t window_size = 0;
    torch::Tensor q_weight;
    torch::Tensor k_weight;
    torch::Tensor v_weight;
    torch::Tensor o_weight;
    torch::Tensor o_bias;

    GptNeoAttentionImpl(
        GptNeoConfig const &cfg,
        bool local_attn);
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor const &global_mask,
        torch::Tensor const &local_mask);
};

TORCH_MODULE(GptNeoAttention);

struct GptNeoMLPImpl : torch::nn::Module
{
    torch::Tensor fc1_weight;
    torch::Tensor fc1_bias;
    torch::Tensor fc2_weight;
    torch::Tensor fc2_bias;

    explicit GptNeoMLPImpl(GptNeoConfig const &cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(GptNeoMLP);

struct GptNeoDecoderImpl : torch::nn::Module
{
    torch::nn::LayerNorm input_norm{nullptr};
    GptNeoAttention attn{nullptr};
    torch::nn::LayerNorm post_attn_norm{nullptr};
    GptNeoMLP mlp{nullptr};

    GptNeoDecoderImpl(
        GptNeoConfig const &cfg,
        bool local_attn);
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor const &global_mask,
        torch::Tensor const &local_mask);
};

TORCH_MODULE(GptNeoDecoder);

struct GptNeoCausalImpl : torch::nn::Module
{
    GptNeoConfig config;
    torch::nn::Embedding wte{nullptr};
    torch::nn::Embedding wpe{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::LayerNorm ln_f{nullptr};
    torch::Tensor lm_weight;
    torch::Tensor cached_pos_;
    torch::Tensor cached_global_mask_;
    torch::Tensor cached_local_mask_;
    int64_t cache_batch_ = -1;
    int64_t cache_seq_ = -1;

    explicit GptNeoCausalImpl(GptNeoConfig cfg);
    void warm_position_cache(
        int64_t batch,
        int64_t seq,
        torch::Device device);
    torch::Tensor forward(torch::Tensor input_ids);
};

TORCH_MODULE(GptNeoCausal);

} // namespace torch_nntile::models
