/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/t5.cpp
 * T5 — LibTorch port of deleted NNGraph ``nntile::model::t5`` (no relative bias).
 */

#include <torch_nntile/models/t5.hh>

#include "nntile_rms_norm.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace torch_nntile::models
{

namespace
{

torch::Tensor t5_rms_norm(torch::Tensor x, torch::Tensor w, double eps)
{
    auto out_rstd = torch_nntile::rms_norm_forward(
        x,
        /*normalized_shape=*/std::vector<int64_t>{x.size(-1)},
        w,
        eps);
    return std::get<0>(out_rstd);
}

struct T5AttentionImpl : torch::nn::Module
{
    torch::nn::Linear q{nullptr};
    torch::nn::Linear k{nullptr};
    torch::nn::Linear v{nullptr};
    torch::nn::Linear o{nullptr};
    int64_t n_heads = 0;
    int64_t d_kv = 0;
    int64_t inner = 0;
    int64_t d_model = 0;

    T5AttentionImpl(T5Config const& cfg)
    {
        n_heads = cfg.num_heads;
        d_kv = cfg.d_kv;
        d_model = cfg.d_model;
        inner = n_heads * d_kv;
        q = register_module(
            "q",
            torch::nn::Linear(
                torch::nn::LinearOptions(d_model, inner).bias(false)));
        k = register_module(
            "k",
            torch::nn::Linear(
                torch::nn::LinearOptions(d_model, inner).bias(false)));
        v = register_module(
            "v",
            torch::nn::Linear(
                torch::nn::LinearOptions(d_model, inner).bias(false)));
        o = register_module(
            "o",
            torch::nn::Linear(
                torch::nn::LinearOptions(inner, d_model).bias(false)));
    }

    torch::Tensor shape(torch::Tensor x) const
    {
        int64_t b = x.size(0);
        int64_t s = x.size(1);
        return x.view({b, s, n_heads, d_kv}).transpose(1, 2);
    }

    torch::Tensor forward(
        torch::Tensor hidden,
        c10::optional<torch::Tensor> key_value_states,
        bool is_causal)
    {
        int64_t b = hidden.size(0);
        int64_t s = hidden.size(1);
        auto qq = shape(q->forward(hidden));
        auto kv_in = key_value_states.has_value() ?
            key_value_states.value() :
            hidden;
        auto kk = shape(k->forward(kv_in));
        auto vv = shape(v->forward(kv_in));
        // HF T5 scores are unscaled; cancel SDPA 1/sqrt(d) on-device.
        qq = at::mul(qq, std::sqrt(static_cast<double>(d_kv)));
        auto out = at::scaled_dot_product_attention(
            qq,
            kk,
            vv,
            /*attn_mask=*/c10::nullopt,
            /*dropout_p=*/0.0,
            is_causal);
        out = out.transpose(1, 2).contiguous().view({b, s, inner});
        return o->forward(out);
    }
};

TORCH_MODULE(T5Attention);

struct T5EncoderBlockImpl : torch::nn::Module
{
    torch::Tensor attn_norm_w;
    T5Attention self_attn{nullptr};
    torch::Tensor ff_norm_w;
    torch::nn::Linear wi_0{nullptr};
    torch::nn::Linear wi_1{nullptr};
    torch::nn::Linear wo{nullptr};
    double eps = 1e-6;

    explicit T5EncoderBlockImpl(T5Config const& cfg)
    {
        eps = cfg.layer_norm_epsilon;
        attn_norm_w = register_parameter(
            "attn_norm_w",
            torch::ones({cfg.d_model}));
        self_attn = register_module("self_attn", T5Attention(cfg));
        ff_norm_w = register_parameter(
            "ff_norm_w",
            torch::ones({cfg.d_model}));
        wi_0 = register_module(
            "wi_0",
            torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_model, cfg.d_ff)
                    .bias(false)));
        wi_1 = register_module(
            "wi_1",
            torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_model, cfg.d_ff)
                    .bias(false)));
        wo = register_module(
            "wo",
            torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_ff, cfg.d_model)
                    .bias(false)));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto h = t5_rms_norm(x, attn_norm_w, eps);
        x = x + self_attn->forward(h, c10::nullopt, /*is_causal=*/false);
        h = t5_rms_norm(x, ff_norm_w, eps);
        auto gated = torch::gelu(wi_0->forward(h), "tanh") *
            wi_1->forward(h);
        return x + wo->forward(gated);
    }
};

TORCH_MODULE(T5EncoderBlock);

struct T5DecoderBlockImpl : torch::nn::Module
{
    torch::Tensor self_norm_w;
    T5Attention self_attn{nullptr};
    torch::Tensor cross_norm_w;
    T5Attention cross_attn{nullptr};
    torch::Tensor ff_norm_w;
    torch::nn::Linear wi_0{nullptr};
    torch::nn::Linear wi_1{nullptr};
    torch::nn::Linear wo{nullptr};
    double eps = 1e-6;

    explicit T5DecoderBlockImpl(T5Config const& cfg)
    {
        eps = cfg.layer_norm_epsilon;
        self_norm_w = register_parameter(
            "self_norm_w",
            torch::ones({cfg.d_model}));
        self_attn = register_module("self_attn", T5Attention(cfg));
        cross_norm_w = register_parameter(
            "cross_norm_w",
            torch::ones({cfg.d_model}));
        cross_attn = register_module("cross_attn", T5Attention(cfg));
        ff_norm_w = register_parameter(
            "ff_norm_w",
            torch::ones({cfg.d_model}));
        wi_0 = register_module(
            "wi_0",
            torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_model, cfg.d_ff)
                    .bias(false)));
        wi_1 = register_module(
            "wi_1",
            torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_model, cfg.d_ff)
                    .bias(false)));
        wo = register_module(
            "wo",
            torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_ff, cfg.d_model)
                    .bias(false)));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor enc)
    {
        auto h = t5_rms_norm(x, self_norm_w, eps);
        x = x + self_attn->forward(h, c10::nullopt, /*is_causal=*/true);
        h = t5_rms_norm(x, cross_norm_w, eps);
        x = x +
            cross_attn->forward(h, enc, /*is_causal=*/false);
        h = t5_rms_norm(x, ff_norm_w, eps);
        auto gated = torch::gelu(wi_0->forward(h), "tanh") *
            wi_1->forward(h);
        return x + wo->forward(gated);
    }
};

TORCH_MODULE(T5DecoderBlock);

} // namespace

T5ForConditionalGenerationImpl::T5ForConditionalGenerationImpl(
    T5Config cfg) :
    config(std::move(cfg))
{
    if (config.d_kv <= 0 || config.num_heads <= 0)
    {
        throw std::invalid_argument(
            "T5: d_kv and num_heads must be > 0");
    }
    shared = register_module(
        "shared",
        torch::nn::Embedding(config.vocab_size, config.d_model));
    torch::nn::ModuleList enc;
    for (int64_t i = 0; i < config.num_layers; ++i)
    {
        enc->push_back(T5EncoderBlock(config));
    }
    encoder_blocks = register_module("encoder_blocks", enc);
    torch::nn::ModuleList dec;
    for (int64_t i = 0; i < config.num_decoder_layers; ++i)
    {
        dec->push_back(T5DecoderBlock(config));
    }
    decoder_blocks = register_module("decoder_blocks", dec);
    enc_final_w = register_parameter(
        "enc_final_w",
        torch::ones({config.d_model}));
    dec_final_w = register_parameter(
        "dec_final_w",
        torch::ones({config.d_model}));
    lm_head = register_module(
        "lm_head",
        torch::nn::Linear(
            torch::nn::LinearOptions(config.d_model, config.vocab_size)
                .bias(false)));
    // Weight tying intentionally unsupported (independent lm_head).
}

torch::Tensor T5ForConditionalGenerationImpl::forward(
    torch::Tensor encoder_input_ids,
    torch::Tensor decoder_input_ids)
{
    auto enc = shared->forward(encoder_input_ids);
    for (auto& module : *encoder_blocks)
    {
        enc = module->as<T5EncoderBlockImpl>()->forward(enc);
    }
    enc = t5_rms_norm(enc, enc_final_w, config.layer_norm_epsilon);

    auto dec = shared->forward(decoder_input_ids);
    for (auto& module : *decoder_blocks)
    {
        dec = module->as<T5DecoderBlockImpl>()->forward(dec, enc);
    }
    dec = t5_rms_norm(dec, dec_final_w, config.layer_norm_epsilon);
    return lm_head->forward(dec);
}

} // namespace torch_nntile::models
