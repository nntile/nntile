/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/gpt2.hh
 * GPT-2 causal LM matching deleted ``nntile::model::gpt2``.
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace torch_nntile::models
{

struct Gpt2Config
{
    int64_t vocab_size = 50257;
    int64_t n_embd = 64;
    int64_t n_head = 2;
    int64_t n_layer = 2;
    int64_t n_positions = 128;
    int64_t n_inner = -1;
    double layer_norm_epsilon = 1e-5;

    int64_t intermediate_size() const
    {
        return n_inner > 0 ? n_inner : 4 * n_embd;
    }

    int64_t head_size() const
    {
        return n_embd / n_head;
    }
};

//! NNGraph ``Gpt2Attention``: gemm → transpose(1) → add_fiber → sdpa →
//! transpose(3) → gemm(ndim=2).
struct Gpt2AttentionImpl : torch::nn::Module
{
    int64_t n_head = 0;
    int64_t head_size = 0;
    int64_t hidden = 0;
    torch::Tensor q_weight;
    torch::Tensor k_weight;
    torch::Tensor v_weight;
    torch::Tensor o_weight;
    torch::Tensor q_bias;
    torch::Tensor k_bias;
    torch::Tensor v_bias;
    torch::Tensor o_bias;

    explicit Gpt2AttentionImpl(Gpt2Config const &cfg);
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor const &causal_mask);
};

TORCH_MODULE(Gpt2Attention);

struct Gpt2MLPImpl : torch::nn::Module
{
    torch::Tensor fc1_weight;
    torch::Tensor fc1_bias;
    torch::Tensor fc2_weight;
    torch::Tensor fc2_bias;

    explicit Gpt2MLPImpl(Gpt2Config const &cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(Gpt2MLP);

struct Gpt2BlockImpl : torch::nn::Module
{
    torch::nn::LayerNorm ln_1{nullptr};
    Gpt2Attention attn{nullptr};
    torch::nn::LayerNorm ln_2{nullptr};
    Gpt2MLP mlp{nullptr};

    explicit Gpt2BlockImpl(Gpt2Config const &cfg);
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor const &causal_mask);
};

TORCH_MODULE(Gpt2Block);

struct Gpt2CausalImpl : torch::nn::Module
{
    Gpt2Config config;
    torch::nn::Embedding wte{nullptr};
    torch::nn::Embedding wpe{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::LayerNorm ln_f{nullptr};
    torch::Tensor lm_weight;
    torch::Tensor cached_pos_;
    torch::Tensor cached_mask_;
    int64_t cache_batch_ = -1;
    int64_t cache_seq_ = -1;

    explicit Gpt2CausalImpl(Gpt2Config cfg);
    void warm_sequence_cache(
        int64_t batch,
        int64_t seq,
        torch::Device device);
    torch::Tensor forward(torch::Tensor input_ids);
};

TORCH_MODULE(Gpt2Causal);

} // namespace torch_nntile::models
