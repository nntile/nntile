/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/gpt_neox.cpp
 * GPT-NeoX causal LM — port of deleted ``nntile::model::gptneox``.
 */

#include <torch_nntile/models/gpt_neox.hh>

#include "nntile_add_fiber.h"
#include "nntile_gemm.h"
#include "nntile_rope.h"
#include "nntile_sdpa.h"
#include "nntile_transpose.h"

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
    double base,
    torch::Tensor &sin_out,
    torch::Tensor &cos_out)
{
    int64_t const half = rope_dim / 2;
    std::vector<float> inv(static_cast<std::size_t>(half));
    for (int64_t i = 0; i < half; ++i)
    {
        double idx = static_cast<double>(2 * i);
        inv[static_cast<std::size_t>(i)] = static_cast<float>(
            1.0
            / std::pow(base, idx / static_cast<double>(rope_dim)));
    }
    sin_out = torch::empty(
        {batch, seq, half},
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
            for (int64_t h = 0; h < half; ++h)
            {
                double angle = static_cast<double>(s)
                    * static_cast<double>(
                        inv[static_cast<std::size_t>(h)]);
                sin_a[b][s][h] = static_cast<float>(std::sin(angle));
                cos_a[b][s][h] = static_cast<float>(std::cos(angle));
            }
        }
    }
}

torch::Tensor apply_partial_rope(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int64_t rotary_ndims,
    int64_t head_dim)
{
    // x: [n_heads, batch, seq, head_dim]; sin/cos: [batch, seq, half].
    int64_t const n_heads = x.size(0);
    if (sin.dim() == 3)
    {
        int64_t const b = sin.size(0);
        int64_t const s = sin.size(1);
        int64_t const half = sin.size(2);
        sin = sin.view({1, b, s, half}).repeat({n_heads, 1, 1, 1});
        cos = cos.view({1, b, s, half}).repeat({n_heads, 1, 1, 1});
    }
    if (rotary_ndims == head_dim)
    {
        return rope(sin, cos, x);
    }
    auto x_rot = x.narrow(
        /*dim=*/-1, /*start=*/0, /*length=*/rotary_ndims);
    auto x_pass = x.narrow(
        /*dim=*/-1,
        /*start=*/rotary_ndims,
        /*length=*/head_dim - rotary_ndims);
    x_rot = rope(sin, cos, x_rot);
    return torch::cat({x_rot, x_pass}, /*dim=*/-1);
}

torch::Tensor linear_gemm(
    torch::Tensor const &x,
    torch::Tensor const &weight,
    torch::Tensor const &bias)
{
    auto out = gemm(
        x,
        weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
    return add_fiber(
        bias,
        out,
        /*axis=*/out.dim() - 1,
        /*batch_ndim=*/0);
}

} // namespace

// ── GptNeoXAttentionImpl ──────────────────────────────────────────────────

GptNeoXAttentionImpl::GptNeoXAttentionImpl(GptNeoXConfig const &cfg) :
    n_heads(cfg.num_attention_heads),
    head_size(cfg.hidden_size / cfg.num_attention_heads),
    hidden(cfg.hidden_size),
    rotary_ndims(gptneox_rope_dim(cfg, head_size))
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
    q = apply_partial_rope(q, sin, cos, rotary_ndims, head_size);
    k = apply_partial_rope(k, sin, cos, rotary_ndims, head_size);
    auto attn = sdpa_kernel(
        q,
        k,
        v,
        mask,
        /*batch_ndim=*/2);
    attn = model_transpose(attn, /*model_ndim=*/3);
    return gemm(attn, o_weight, /*ndim=*/2, /*batch_ndim=*/0);
}

// ── GptNeoXMLPImpl ────────────────────────────────────────────────────────

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
    x = linear_gemm(x, fc1_weight, fc1_bias);
    x = torch::gelu(x);
    return linear_gemm(x, fc2_weight, fc2_bias);
}

// ── GptNeoXDecoderImpl ────────────────────────────────────────────────────

GptNeoXDecoderImpl::GptNeoXDecoderImpl(GptNeoXConfig const &cfg) :
    parallel_residual(cfg.use_parallel_residual)
{
    input_norm = register_module(
        "input_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({cfg.hidden_size})
                .eps(cfg.layer_norm_eps)));
    attn = register_module("attention", GptNeoXAttention(cfg));
    post_attn_norm = register_module(
        "post_attn_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({cfg.hidden_size})
                .eps(cfg.layer_norm_eps)));
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
        return x + attn_out + mlp_out;
    }
    auto post = x + attn_out;
    return post + mlp->forward(post_attn_norm->forward(post));
}

// ── GptNeoXCausalImpl ─────────────────────────────────────────────────────

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
        torch::nn::Embedding(config.vocab_size, config.hidden_size));
    torch::nn::ModuleList list;
    for (int64_t i = 0; i < config.num_hidden_layers; ++i)
    {
        list->push_back(GptNeoXDecoder(config));
    }
    layers = register_module("layers", list);
    final_layer_norm = register_module(
        "final_layer_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.hidden_size})
                .eps(config.layer_norm_eps)));
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
    torch::Tensor sin_h;
    torch::Tensor cos_h;
    rope_sin_cos_host(
        batch,
        seq,
        rope_dim,
        config.rotary_emb_base,
        sin_h,
        cos_h);
    auto mask = causal_mask_host(seq);
    if (!device.is_cpu())
    {
        sin_h = sin_h.to(device);
        cos_h = cos_h.to(device);
        mask = mask.to(device);
    }
    rope_sin_ = sin_h;
    rope_cos_ = cos_h;
    cached_mask_ = mask;
    rope_cache_batch_ = batch;
    rope_cache_seq_ = seq;
}

torch::Tensor GptNeoXCausalImpl::forward(torch::Tensor input_ids)
{
    int64_t const b = input_ids.size(0);
    int64_t const s = input_ids.size(1);
    if (!rope_sin_.defined() || rope_cache_batch_ != b
        || rope_cache_seq_ != s
        || rope_sin_.device() != input_ids.device())
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
