/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/gpt_neo.hh
 * GPT-Neo causal LM for device=nntile (LibTorch).
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
    //! ``"global"`` / ``"local"`` per layer; empty → HF alternate default.
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

struct GptNeoCausalImpl : torch::nn::Module
{
    GptNeoConfig config;
    torch::nn::Embedding wte{nullptr};
    torch::nn::Embedding wpe{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};

    explicit GptNeoCausalImpl(GptNeoConfig cfg);
    torch::Tensor forward(torch::Tensor input_ids);
};

TORCH_MODULE(GptNeoCausal);

} // namespace torch_nntile::models
