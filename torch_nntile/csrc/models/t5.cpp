/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/t5.cpp
 * T5 encoder-decoder - port of deleted ``nntile::model::t5``.
 */

#include <torch_nntile/models/t5.hh>

#include "nntile_gemm.h"
#include "nntile_rms_norm.h"
#include "nntile_sdpa.h"
#include "nntile_transpose.h"

#include <optional>
#include <stdexcept>
#include <vector>

namespace torch_nntile::models
{

namespace
{

torch::Tensor causal_mask_host(int64_t seq)
{
    auto opts = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(torch::kCPU);
    auto q = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
        .unsqueeze(1);
    auto k = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU))
        .unsqueeze(0);
    return (k <= q).to(opts);
}

torch::Tensor rms_norm(
    torch::Tensor x,
    torch::Tensor weight,
    double eps)
{
    auto out_rstd = rms_norm_forward(
        x,
        /*normalized_shape=*/std::vector<int64_t>{x.size(-1)},
        weight,
        eps);
    return std::get<0>(out_rstd);
}

} // namespace

// -- T5AttentionImpl -------------------------------------------------------

T5AttentionImpl::T5AttentionImpl(T5Config const &cfg, bool cross) :
    is_cross(cross),
    n_heads(cfg.num_heads),
    head_size(cfg.d_kv)
{
    int64_t const d = cfg.d_model;
    int64_t const hs = head_size;
    int64_t const nh = n_heads;
    q_weight = register_parameter(
        "q_weight",
        torch::empty({d, hs, nh}));
    k_weight = register_parameter(
        "k_weight",
        torch::empty({d, hs, nh}));
    v_weight = register_parameter(
        "v_weight",
        torch::empty({d, hs, nh}));
    o_weight = register_parameter(
        "o_weight",
        torch::empty({hs, nh, d}));
    torch::nn::init::normal_(q_weight, 0.0, 0.02);
    torch::nn::init::normal_(k_weight, 0.0, 0.02);
    torch::nn::init::normal_(v_weight, 0.0, 0.02);
    torch::nn::init::normal_(o_weight, 0.0, 0.02);
}

torch::Tensor T5AttentionImpl::forward(
    torch::Tensor x,
    torch::Tensor encoder_hidden,
    torch::Tensor const &mask)
{
    auto const &kv_src =
        (is_cross && encoder_hidden.defined()) ? encoder_hidden : x;
    auto q = model_transpose(
        gemm(x, q_weight, /*ndim=*/1, /*batch_ndim=*/0),
        /*model_ndim=*/1);
    auto k = model_transpose(
        gemm(kv_src, k_weight, /*ndim=*/1, /*batch_ndim=*/0),
        /*model_ndim=*/1);
    auto v = model_transpose(
        gemm(kv_src, v_weight, /*ndim=*/1, /*batch_ndim=*/0),
        /*model_ndim=*/1);
    // HF T5 omits 1/sqrt(d) scale; nntile sdpa uses default scale.
    // Match NNGraph path layout; scale parity is a known residual.
    auto attn = sdpa_kernel(
        q,
        k,
        v,
        mask.defined() ? std::optional<torch::Tensor>(mask)
                       : std::nullopt,
        /*batch_ndim=*/2);
    attn = model_transpose(attn, /*model_ndim=*/3);
    return gemm(attn, o_weight, /*ndim=*/2, /*batch_ndim=*/0);
}

// -- T5LayerFFImpl ---------------------------------------------------------

T5LayerFFImpl::T5LayerFFImpl(T5Config const &cfg) :
    eps(cfg.layer_norm_epsilon)
{
    // Deleted T5 FF uses RMSNorm + GatedMlp (GELUTANH).
    ln_weight = register_parameter(
        "ln_weight",
        torch::ones({cfg.d_model}));
    int64_t const d = cfg.d_model;
    int64_t const ff = cfg.d_ff;
    gate_weight = register_parameter(
        "gate_weight",
        torch::empty({ff, d}));
    up_weight = register_parameter(
        "up_weight",
        torch::empty({ff, d}));
    down_weight = register_parameter(
        "down_weight",
        torch::empty({d, ff}));
    torch::nn::init::normal_(gate_weight, 0.0, 0.02);
    torch::nn::init::normal_(up_weight, 0.0, 0.02);
    torch::nn::init::normal_(down_weight, 0.0, 0.02);
}

torch::Tensor T5LayerFFImpl::forward(torch::Tensor x)
{
    auto x_norm = rms_norm(x, ln_weight, eps);
    auto gate = gemm(
        x_norm,
        gate_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
    auto up = gemm(
        x_norm,
        up_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
    auto hidden = torch::gelu(gate, "tanh") * up;
    auto ff_out = gemm(
        hidden,
        down_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
    return x + ff_out;
}

// -- T5EncoderBlockImpl ----------------------------------------------------

T5EncoderBlockImpl::T5EncoderBlockImpl(T5Config const &cfg) :
    eps(cfg.layer_norm_epsilon)
{
    ln0_weight = register_parameter(
        "ln0_weight",
        torch::ones({cfg.d_model}));
    self_attn = register_module(
        "self_attn",
        T5Attention(cfg, /*cross=*/false));
    ff = register_module("ff", T5LayerFF(cfg));
}

torch::Tensor T5EncoderBlockImpl::forward(torch::Tensor x)
{
    auto x_norm = rms_norm(x, ln0_weight, eps);
    auto attn = self_attn->forward(x_norm, {}, {});
    return ff->forward(x + attn);
}

// -- T5DecoderBlockImpl ----------------------------------------------------

T5DecoderBlockImpl::T5DecoderBlockImpl(T5Config const &cfg) :
    eps(cfg.layer_norm_epsilon)
{
    ln0_weight = register_parameter(
        "ln0_weight",
        torch::ones({cfg.d_model}));
    ln1_weight = register_parameter(
        "ln1_weight",
        torch::ones({cfg.d_model}));
    self_attn = register_module(
        "self_attn",
        T5Attention(cfg, /*cross=*/false));
    cross_attn = register_module(
        "cross_attn",
        T5Attention(cfg, /*cross=*/true));
    ff = register_module("ff", T5LayerFF(cfg));
}

torch::Tensor T5DecoderBlockImpl::forward(
    torch::Tensor x,
    torch::Tensor encoder_hidden,
    torch::Tensor const &self_mask)
{
    auto x_norm = rms_norm(x, ln0_weight, eps);
    auto self_out = self_attn->forward(x_norm, {}, self_mask);
    auto post = x + self_out;
    auto y_norm = rms_norm(post, ln1_weight, eps);
    auto cross_out = cross_attn->forward(
        y_norm,
        encoder_hidden,
        {});
    return ff->forward(post + cross_out);
}

// -- T5ForConditionalGenerationImpl ----------------------------------------

T5ForConditionalGenerationImpl::T5ForConditionalGenerationImpl(
    T5Config cfg) :
    config(std::move(cfg))
{
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
    lm_weight = register_parameter(
        "lm_weight",
        torch::empty({config.vocab_size, config.d_model}));
    torch::nn::init::normal_(lm_weight, 0.0, 0.02);
}

torch::Tensor T5ForConditionalGenerationImpl::forward(
    torch::Tensor encoder_input_ids,
    torch::Tensor decoder_input_ids)
{
    auto enc = shared->forward(encoder_input_ids);
    for (auto &module : *encoder_blocks)
    {
        enc = module->as<T5EncoderBlockImpl>()->forward(enc);
    }
    enc = rms_norm(enc, enc_final_w, config.layer_norm_epsilon);
    int64_t const s = decoder_input_ids.size(1);
    auto self_mask = causal_mask_host(s);
    if (!decoder_input_ids.device().is_cpu())
    {
        self_mask = self_mask.to(decoder_input_ids.device());
    }
    auto dec = shared->forward(decoder_input_ids);
    for (auto &module : *decoder_blocks)
    {
        dec = module->as<T5DecoderBlockImpl>()->forward(
            dec,
            enc,
            self_mask);
    }
    dec = rms_norm(dec, dec_final_w, config.layer_norm_epsilon);
    return gemm(
        dec,
        lm_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
}

} // namespace torch_nntile::models
