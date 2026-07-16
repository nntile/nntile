/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/llama.hh
 * Tiny Llama causal LM for device=nntile (LibTorch).
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace torch_nntile::models
{

struct LlamaConfig
{
    int64_t vocab_size = 32000;
    int64_t hidden_size = 64;
    int64_t intermediate_size = 128;
    int64_t num_hidden_layers = 2;
    int64_t num_attention_heads = 2;
    int64_t num_key_value_heads = 2;
    int64_t max_position_embeddings = 128;
    double rms_norm_eps = 1e-6;
    double rope_theta = 10000.0;
};

struct LlamaCausalImpl : torch::nn::Module
{
    LlamaConfig config;
    torch::nn::Embedding embed_tokens{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::Linear lm_head{nullptr};
    torch::Tensor weight_rms;
    //! One-shot RoPE tables (NNGraph bind_data); not recomputed per step.
    torch::Tensor rope_sin_;
    torch::Tensor rope_cos_;
    int64_t rope_cache_batch_ = -1;
    int64_t rope_cache_seq_ = -1;

    explicit LlamaCausalImpl(LlamaConfig cfg);
    void warm_rope_cache(
        int64_t batch,
        int64_t seq,
        torch::Device device);
    torch::Tensor forward(torch::Tensor input_ids);
};

TORCH_MODULE(LlamaCausal);

} // namespace torch_nntile::models
