/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/t5.cpp
 */

#include <torch_nntile/models/t5.hh>

namespace torch_nntile::models
{

namespace
{

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, double eps)
{
    auto var = x.pow(2).mean(-1, true);
    return x * torch::rsqrt(var + eps) * w;
}

} // namespace

T5ForConditionalGenerationImpl::T5ForConditionalGenerationImpl(T5Config cfg) :
    config(std::move(cfg))
{
    shared = register_module(
        "shared",
        torch::nn::Embedding(config.vocab_size, config.d_model));
    enc_attn = register_module(
        "enc_attn",
        torch::nn::Linear(
            torch::nn::LinearOptions(config.d_model, config.d_model)
                .bias(false)));
    enc_ff = register_module(
        "enc_ff",
        torch::nn::Linear(
            torch::nn::LinearOptions(config.d_model, config.d_model)
                .bias(false)));
    dec_attn = register_module(
        "dec_attn",
        torch::nn::Linear(
            torch::nn::LinearOptions(config.d_model, config.d_model)
                .bias(false)));
    dec_cross = register_module(
        "dec_cross",
        torch::nn::Linear(
            torch::nn::LinearOptions(config.d_model, config.d_model)
                .bias(false)));
    dec_ff = register_module(
        "dec_ff",
        torch::nn::Linear(
            torch::nn::LinearOptions(config.d_model, config.d_model)
                .bias(false)));
    lm_head = register_module(
        "lm_head",
        torch::nn::Linear(
            torch::nn::LinearOptions(config.d_model, config.vocab_size)
                .bias(false)));
    enc_norm_w = register_parameter(
        "enc_norm_w",
        torch::ones({config.d_model}));
    dec_norm_w = register_parameter(
        "dec_norm_w",
        torch::ones({config.d_model}));
}

torch::Tensor T5ForConditionalGenerationImpl::forward(
    torch::Tensor encoder_input_ids,
    torch::Tensor decoder_input_ids)
{
    auto enc = shared->forward(encoder_input_ids);
    enc = enc + enc_attn->forward(rms_norm(enc, enc_norm_w,
        config.layer_norm_epsilon));
    enc = enc + torch::relu(enc_ff->forward(enc));

    auto dec = shared->forward(decoder_input_ids);
    dec = dec + dec_attn->forward(rms_norm(dec, dec_norm_w,
        config.layer_norm_epsilon));
    // Simplified cross-attn: project encoder mean as context.
    auto ctx = enc.mean(/*dim=*/1, /*keepdim=*/true).expand_as(dec);
    dec = dec + dec_cross->forward(ctx);
    dec = dec + torch::relu(dec_ff->forward(dec));
    return lm_head->forward(dec);
}

} // namespace torch_nntile::models
