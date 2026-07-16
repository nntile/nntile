/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/gpt2.hh
 * Tiny GPT-2 causal LM for device=nntile (LibTorch).
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>

namespace torch_nntile::models
{

struct Gpt2Config
{
    int64_t vocab_size = 50257;
    int64_t n_embd = 64;
    int64_t n_head = 2;
    int64_t n_layer = 2;
    int64_t n_positions = 128;
    double layer_norm_epsilon = 1e-5;
};

struct Gpt2BlockImpl : torch::nn::Module
{
    torch::nn::LayerNorm ln_1{nullptr};
    torch::nn::Linear qkv{nullptr};
    torch::nn::Linear c_proj{nullptr};
    torch::nn::LayerNorm ln_2{nullptr};
    torch::nn::Linear fc_in{nullptr};
    torch::nn::Linear fc_out{nullptr};
    int64_t n_head = 0;
    int64_t n_embd = 0;

    Gpt2BlockImpl(Gpt2Config const& cfg);
    torch::Tensor forward(torch::Tensor x, torch::Tensor const& causal_mask);
};

TORCH_MODULE(Gpt2Block);

struct Gpt2CausalImpl : torch::nn::Module
{
    Gpt2Config config;
    torch::nn::Embedding wte{nullptr};
    torch::nn::Embedding wpe{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::LayerNorm ln_f{nullptr};
    torch::nn::Linear lm_head{nullptr};
    //! One-shot host aux tables uploaded for reuse (not activations).
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
