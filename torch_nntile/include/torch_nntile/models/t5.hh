/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/include/torch_nntile/models/t5.hh
 * T5 encoder-decoder matching deleted ``nntile::model::t5``.
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
    int64_t d_kv = 32;
    int64_t d_ff = 128;
    int64_t num_layers = 1;
    int64_t num_decoder_layers = 1;
    int64_t num_heads = 2;
    double layer_norm_epsilon = 1e-6;
    bool tie_word_embeddings = false;
};

struct T5AttentionImpl : torch::nn::Module
{
    bool is_cross = false;
    int64_t n_heads = 0;
    int64_t head_size = 0;
    torch::Tensor q_weight;
    torch::Tensor k_weight;
    torch::Tensor v_weight;
    torch::Tensor o_weight;

    T5AttentionImpl(T5Config const &cfg, bool cross);
    //! ``encoder_hidden`` may be an undefined tensor for self-attention.
    //! ``mask`` may be an undefined tensor (encoder cross-attn has no mask).
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor encoder_hidden,
        torch::Tensor const &mask);
};

TORCH_MODULE(T5Attention);

struct T5LayerFFImpl : torch::nn::Module
{
    torch::Tensor ln_weight;
    torch::Tensor gate_weight;
    torch::Tensor up_weight;
    torch::Tensor down_weight;
    double eps = 1e-6;

    explicit T5LayerFFImpl(T5Config const &cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(T5LayerFF);

struct T5EncoderBlockImpl : torch::nn::Module
{
    torch::Tensor ln0_weight;
    T5Attention self_attn{nullptr};
    T5LayerFF ff{nullptr};
    double eps = 1e-6;

    explicit T5EncoderBlockImpl(T5Config const &cfg);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(T5EncoderBlock);

struct T5DecoderBlockImpl : torch::nn::Module
{
    torch::Tensor ln0_weight;
    torch::Tensor ln1_weight;
    T5Attention self_attn{nullptr};
    T5Attention cross_attn{nullptr};
    T5LayerFF ff{nullptr};
    double eps = 1e-6;

    explicit T5DecoderBlockImpl(T5Config const &cfg);
    //! ``self_mask`` is the causal mask for the decoder self-attention.
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor encoder_hidden,
        torch::Tensor const &self_mask);
};

TORCH_MODULE(T5DecoderBlock);

struct T5ForConditionalGenerationImpl : torch::nn::Module
{
    T5Config config;
    torch::nn::Embedding shared{nullptr};
    torch::nn::ModuleList encoder_blocks{nullptr};
    torch::nn::ModuleList decoder_blocks{nullptr};
    torch::Tensor enc_final_w;
    torch::Tensor dec_final_w;
    torch::Tensor lm_weight;

    explicit T5ForConditionalGenerationImpl(T5Config cfg);
    torch::Tensor forward(
        torch::Tensor encoder_input_ids,
        torch::Tensor decoder_input_ids);
};

TORCH_MODULE(T5ForConditionalGeneration);

} // namespace torch_nntile::models
