/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/t5.hh
 * Tiny T5 encoder-decoder for device=nntile.
 */

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace torch_nntile::models
{

struct T5Config
{
    int64_t vocab_size = 32128;
    int64_t d_model = 64;
    int64_t d_ff = 128;
    int64_t num_layers = 1;
    int64_t num_heads = 2;
    double layer_norm_epsilon = 1e-6;
};

struct T5ForConditionalGenerationImpl : torch::nn::Module
{
    T5Config config;
    torch::nn::Embedding shared{nullptr};
    torch::nn::Linear lm_head{nullptr};
    torch::nn::Linear enc_attn{nullptr};
    torch::nn::Linear enc_ff{nullptr};
    torch::nn::Linear dec_attn{nullptr};
    torch::nn::Linear dec_cross{nullptr};
    torch::nn::Linear dec_ff{nullptr};
    torch::Tensor enc_norm_w;
    torch::Tensor dec_norm_w;

    explicit T5ForConditionalGenerationImpl(T5Config cfg);
    torch::Tensor forward(
        torch::Tensor encoder_input_ids,
        torch::Tensor decoder_input_ids);
};

TORCH_MODULE(T5ForConditionalGeneration);

} // namespace torch_nntile::models
