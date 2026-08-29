/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/gpt_neox.cpp
 * GPT-NeoX causal LM - port of deleted ``nntile::model::gptneox``.
 */

#include <torch_nntile/models/gpt_neox.hh>

#include "nntile_add_fiber.h"
#include "nntile_gemm.h"
#include "nntile_nn_classic.h"
#include "nntile_rope.h"
#include "nntile_sdpa.h"
#include "nntile_model_transpose.h"

#include <cmath>
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

int64_t gptneox_rope_dim(GptNeoXConfig const &cfg, int64_t head_dim)
{
    // Match Python GPTNeoXConfig.rotary_ndims: disabled pct => 0, else
    // round and clamp to an even width in [2, head_dim].
    if (cfg.rotary_pct <= 0.0)
    {
        return 0;
    }
    int64_t dim = static_cast<int64_t>(
        std::lround(
            static_cast<double>(head_dim) * cfg.rotary_pct));
    if (dim < 2)
    {
        dim = 2;
    }
    if (dim % 2 != 0)
    {
        --dim;
    }
    if (dim > head_dim)
    {
        dim = head_dim - (head_dim % 2);
    }
    return dim;
}

void rope_sin_cos_host(
    int64_t batch,
    int64_t seq,
    int64_t rope_dim,
    int64_t head_dim,
    double base,
    torch::Tensor &sin_out,
    torch::Tensor &cos_out)
{
    // Frequencies for ``rope_dim`` (HF ``rotary_pct``); remaining pairs
    // of the full head are identity (sin=0, cos=1) so ``rope`` can run
    // on the unsplit last axis (narrow+cat disagrees with the kernel).
    int64_t const half_rot = rope_dim / 2;
    int64_t const half_full = head_dim / 2;
    std::vector<float> inv(static_cast<std::size_t>(half_rot));
    for (int64_t i = 0; i < half_rot; ++i)
    {
        double idx = static_cast<double>(2 * i);
        inv[static_cast<std::size_t>(i)] = static_cast<float>(
            1.0
            / std::pow(base, idx / static_cast<double>(rope_dim)));
    }
    sin_out = torch::empty(
        {batch, seq, half_full},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU));
    cos_out = torch::empty_like(sin_out);
    auto sin_a = sin_out.accessor<float, 3>();
    auto cos_a = cos_out.accessor<float, 3>();
    for (int64_t b = 0; b < batch; ++b)
    {
        for (int64_t s = 0; s < seq; ++s)
        {
            for (int64_t h = 0; h < half_full; ++h)
            {
                if (h < half_rot)
                {
                    double angle = static_cast<double>(s)
                        * static_cast<double>(
                            inv[static_cast<std::size_t>(h)]);
                    sin_a[b][s][h] = static_cast<float>(
                        std::sin(angle));
                    cos_a[b][s][h] = static_cast<float>(
                        std::cos(angle));
                }
                else
                {
                    sin_a[b][s][h] = 0.0f;
                    cos_a[b][s][h] = 1.0f;
                }
            }
        }
    }
}

torch::Tensor apply_partial_rope(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int64_t rotary_ndims)
{
    // x: [n_heads, batch, seq, head_dim]; sin/cos padded to
    // [batch, seq, head_dim/2] (identity on unused pairs).
    if (rotary_ndims <= 0)
    {
        return x;
    }
    return rope(sin, cos, x);
}

} // namespace

// -- GptNeoXAttentionImpl --------------------------------------------------

GptNeoXAttentionImpl::GptNeoXAttentionImpl(GptNeoXConfig const &cfg) :
    n_heads(cfg.num_attention_heads),
    head_size(cfg.hidden_size / cfg.num_attention_heads),
    hidden(cfg.hidden_size),
    rotary_ndims(gptneox_rope_dim(cfg, head_size)),
    attention_bias(cfg.attention_bias)
{
    if (cfg.hidden_size % cfg.num_attention_heads != 0)
    {
        throw std::invalid_argument(
            "GptNeoXAttention: hidden_size % n_heads != 0");
    }
    int64_t const hs = head_size;
    int64_t const nh = n_heads;
    int64_t const h = hidden;
    q_weight = register_parameter(
        "q_weight",
        torch::empty({h, hs, nh}));
    k_weight = register_parameter(
        "k_weight",
        torch::empty({h, hs, nh}));
    v_weight = register_parameter(
        "v_weight",
        torch::empty({h, hs, nh}));
    o_weight = register_parameter(
        "o_weight",
        torch::empty({hs, nh, h}));
    if (attention_bias)
    {
        // ``(n_heads, head_dim)`` after transpose into SDPA layout.
        q_bias = register_parameter(
            "q_bias",
            torch::zeros({nh, hs}));
        k_bias = register_parameter(
            "k_bias",
            torch::zeros({nh, hs}));
        v_bias = register_parameter(
            "v_bias",
            torch::zeros({nh, hs}));
        o_bias = register_parameter("o_bias", torch::zeros({h}));
    }
    torch::nn::init::normal_(q_weight, 0.0, 0.02);
    torch::nn::init::normal_(k_weight, 0.0, 0.02);
    torch::nn::init::normal_(v_weight, 0.0, 0.02);
    torch::nn::init::normal_(o_weight, 0.0, 0.02);
}

torch::Tensor GptNeoXAttentionImpl::forward(
    torch::Tensor x,
    torch::Tensor const &sin,
    torch::Tensor const &cos,
    torch::Tensor const &mask)
{
    auto q = model_transpose(
        gemm(x, q_weight, /*ndim=*/1, /*batch_ndim=*/0),
        /*model_ndim=*/1);
    auto k = model_transpose(
        gemm(x, k_weight, /*ndim=*/1, /*batch_ndim=*/0),
        /*model_ndim=*/1);
    auto v = model_transpose(
        gemm(x, v_weight, /*ndim=*/1, /*batch_ndim=*/0),
        /*model_ndim=*/1);
    if (attention_bias)
    {
        q = add_fiber(q_bias, q, /*axis=*/3, /*batch_ndim=*/1);
        k = add_fiber(k_bias, k, /*axis=*/3, /*batch_ndim=*/1);
        v = add_fiber(v_bias, v, /*axis=*/3, /*batch_ndim=*/1);
    }
    if (rotary_ndims > 0)
    {
        q = apply_partial_rope(q, sin, cos, rotary_ndims);
        k = apply_partial_rope(k, sin, cos, rotary_ndims);
    }
    auto attn = sdpa_kernel(
        q,
        k,
        v,
        mask,
        /*batch_ndim=*/2);
    attn = model_transpose(attn, /*model_ndim=*/3);
    auto out = gemm(attn, o_weight, /*ndim=*/2, /*batch_ndim=*/0);
    if (attention_bias)
    {
        out = add_fiber(
            o_bias,
            out,
            /*axis=*/out.dim() - 1,
            /*batch_ndim=*/0);
    }
    return out;
}

// -- GptNeoXMLPImpl --------------------------------------------------------

GptNeoXMLPImpl::GptNeoXMLPImpl(GptNeoXConfig const &cfg)
{
    int64_t const h = cfg.hidden_size;
    int64_t const inner = cfg.intermediate_size;
    fc1_weight = register_parameter(
        "fc1_weight",
        torch::empty({inner, h}));
    fc1_bias = register_parameter("fc1_bias", torch::zeros({inner}));
    fc2_weight = register_parameter(
        "fc2_weight",
        torch::empty({h, inner}));
    fc2_bias = register_parameter("fc2_bias", torch::zeros({h}));
    torch::nn::init::normal_(fc1_weight, 0.0, 0.02);
    torch::nn::init::normal_(fc2_weight, 0.0, 0.02);
}

torch::Tensor GptNeoXMLPImpl::forward(torch::Tensor x)
{
    // Weight layout ``[out, in]`` with ``trans_b`` (NNGraph Linear).
    x = gemm(
        x,
        fc1_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
    x = add_fiber(
        fc1_bias,
        x,
        /*axis=*/x.dim() - 1,
        /*batch_ndim=*/0);
    x = nn_classic::gelu(x, false);
    x = gemm(
        x,
        fc2_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
    return add_fiber(
        fc2_bias,
        x,
        /*axis=*/x.dim() - 1,
        /*batch_ndim=*/0);
}

// -- GptNeoXDecoderImpl ----------------------------------------------------

GptNeoXDecoderImpl::GptNeoXDecoderImpl(GptNeoXConfig const &cfg) :
    parallel_residual(cfg.use_parallel_residual)
{
    input_norm = register_module(
        "input_norm",
        nn_classic::LayerNorm(cfg.hidden_size, cfg.layer_norm_eps));
    attn = register_module("attention", GptNeoXAttention(cfg));
    post_attn_norm = register_module(
        "post_attn_norm",
        nn_classic::LayerNorm(cfg.hidden_size, cfg.layer_norm_eps));
    mlp = register_module("mlp", GptNeoXMLP(cfg));
}

torch::Tensor GptNeoXDecoderImpl::forward(
    torch::Tensor x,
    torch::Tensor const &sin,
    torch::Tensor const &cos,
    torch::Tensor const &mask)
{
    auto attn_out = attn->forward(
        input_norm->forward(x),
        sin,
        cos,
        mask);
    if (parallel_residual)
    {
        auto mlp_out = mlp->forward(post_attn_norm->forward(x));
        return nn_classic::add(
            nn_classic::add(x, attn_out),
            mlp_out);
    }
    auto post = nn_classic::add(x, attn_out);
    return nn_classic::add(
        post,
        mlp->forward(post_attn_norm->forward(post)));
}

// -- GptNeoXCausalImpl -----------------------------------------------------

GptNeoXCausalImpl::GptNeoXCausalImpl(GptNeoXConfig cfg) :
    config(std::move(cfg))
{
    if (config.hidden_size % config.num_attention_heads != 0)
    {
        throw std::invalid_argument(
            "GptNeoXCausal: hidden_size % n_heads != 0");
    }
    embed_in = register_module(
        "embed_in",
        nn_classic::Embedding(config.vocab_size, config.hidden_size));
    torch::nn::ModuleList list;
    for (int64_t i = 0; i < config.num_hidden_layers; ++i)
    {
        list->push_back(GptNeoXDecoder(config));
    }
    layers = register_module("layers", list);
    final_layer_norm = register_module(
        "final_layer_norm",
        nn_classic::LayerNorm(
            config.hidden_size,
            config.layer_norm_eps));
    lm_weight = register_parameter(
        "lm_weight",
        torch::empty({config.vocab_size, config.hidden_size}));
    torch::nn::init::normal_(lm_weight, 0.0, 0.02);
}

void GptNeoXCausalImpl::warm_rope_cache(
    int64_t batch,
    int64_t seq,
    torch::Device device)
{
    int64_t const head_dim =
        config.hidden_size / config.num_attention_heads;
    int64_t const rope_dim = gptneox_rope_dim(config, head_dim);
    auto mask = causal_mask_host(seq);
    if (!device.is_cpu())
    {
        mask = mask.to(device);
    }
    cached_mask_ = mask;
    if (rope_dim > 0)
    {
        torch::Tensor sin_h;
        torch::Tensor cos_h;
        rope_sin_cos_host(
            batch,
            seq,
            rope_dim,
            head_dim,
            config.rotary_emb_base,
            sin_h,
            cos_h);
        if (!device.is_cpu())
        {
            sin_h = sin_h.to(device);
            cos_h = cos_h.to(device);
        }
        rope_sin_ = sin_h;
        rope_cos_ = cos_h;
    }
    else
    {
        rope_sin_ = torch::Tensor();
        rope_cos_ = torch::Tensor();
    }
    rope_cache_batch_ = batch;
    rope_cache_seq_ = seq;
}

torch::Tensor GptNeoXCausalImpl::forward(torch::Tensor input_ids)
{
    int64_t const b = input_ids.size(0);
    int64_t const s = input_ids.size(1);
    int64_t const rope_dim = gptneox_rope_dim(
        config,
        config.hidden_size / config.num_attention_heads);
    bool const need_rope_refresh =
        !cached_mask_.defined()
        || rope_cache_batch_ != b
        || rope_cache_seq_ != s
        || cached_mask_.device() != input_ids.device()
        || (rope_dim > 0
            && (!rope_sin_.defined()
                || rope_sin_.device() != input_ids.device()));
    if (need_rope_refresh)
    {
        warm_rope_cache(b, s, input_ids.device());
    }
    auto x = embed_in->forward(input_ids);
    for (auto &module : *layers)
    {
        x = module->as<GptNeoXDecoderImpl>()->forward(
            x,
            rope_sin_,
            rope_cos_,
            cached_mask_);
    }
    x = final_layer_norm->forward(x);
    return gemm(
        x,
        lm_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
}

} // namespace torch_nntile::models
