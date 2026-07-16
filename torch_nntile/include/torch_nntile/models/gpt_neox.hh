/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/gpt_neox.hh
 * GPT-NeoX causal LM matching deleted ``nntile::model::gptneox``.
 */

#pragma once

#include <torch/torch.h>

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

struct GptNeoXCausalImpl : torch::nn::Module
{
    GptNeoXConfig config;
    torch::nn::Embedding embed_in{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::LayerNorm final_layer_norm{nullptr};
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
