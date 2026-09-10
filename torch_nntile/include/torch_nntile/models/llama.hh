/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/llama.hh
 * Llama causal LM matching deleted ``nntile::model::llama``.
 */

#pragma once

#include <torch/torch.h>
#include <torch_nntile/classic_nn.hh>

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

struct LlamaAttentionImpl : torch::nn::Module
{
    int64_t n_heads = 0;
    int64_t n_kv = 0;
    int64_t head_size = 0;
    int64_t n_rep = 0;
    bool use_gqa = false;
    torch::Tensor q_weight;
    torch::Tensor k_weight;
    torch::Tensor v_weight;
    torch::Tensor o_weight;

    explicit LlamaAttentionImpl(LlamaConfig const &cfg);
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor const &sin,
        torch::Tensor const &cos,
        torch::Tensor const &mask);
};

TORCH_MODULE(LlamaAttention);

struct LlamaMLPImpl : torch::nn::Module
{
    torch::Tensor gate_weight;
    torch::Tensor up_weight;
    torch::Tensor down_weight;

    explicit LlamaMLPImpl(LlamaConfig const &cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(LlamaMLP);

struct LlamaDecoderImpl : torch::nn::Module
{
    torch::Tensor input_norm_w;
    LlamaAttention attn{nullptr};
    torch::Tensor post_attn_norm_w;
    LlamaMLP mlp{nullptr};
    double rms_eps = 1e-6;

    explicit LlamaDecoderImpl(LlamaConfig const &cfg);
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor const &sin,
        torch::Tensor const &cos,
        torch::Tensor const &mask);
};

TORCH_MODULE(LlamaDecoder);

struct LlamaCausalImpl : torch::nn::Module
{
    LlamaConfig config;
    nn_classic::Embedding embed_tokens{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::Tensor lm_weight;
    torch::Tensor weight_rms;
    torch::Tensor rope_sin_;
    torch::Tensor rope_cos_;
    torch::Tensor cached_mask_;
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
