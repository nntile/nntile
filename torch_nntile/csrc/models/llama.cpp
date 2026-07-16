/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/llama.cpp
 * Llama causal LM - port of deleted ``nntile::model::llama``.
 */

#include <torch_nntile/models/llama.hh>

#include "nntile_gemm.h"
#include "nntile_rms_norm.h"
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

void rope_sin_cos_host(
    int64_t batch,
    int64_t seq,
    int64_t head_dim,
    double rope_theta,
    torch::Tensor &sin_out,
    torch::Tensor &cos_out)
{
    if (head_dim % 2 != 0)
    {
        throw std::invalid_argument("rope: head_dim must be even");
    }
    int64_t const half = head_dim / 2;
    std::vector<float> inv(static_cast<std::size_t>(half));
    for (int64_t i = 0; i < half; ++i)
    {
        double idx = static_cast<double>(2 * i);
        inv[static_cast<std::size_t>(i)] = static_cast<float>(
            1.0
            / std::pow(
                rope_theta,
                idx / static_cast<double>(head_dim)));
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

torch::Tensor apply_rope(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos)
{
    // x layout: [n_heads, batch, seq, head_dim] (after model_transpose).
    int64_t const n_heads = x.size(0);
    if (sin.dim() == 3)
    {
        int64_t const b = sin.size(0);
        int64_t const s = sin.size(1);
        int64_t const half = sin.size(2);
        sin = sin.view({1, b, s, half}).repeat({n_heads, 1, 1, 1});
        cos = cos.view({1, b, s, half}).repeat({n_heads, 1, 1, 1});
    }
    return rope(sin, cos, x);
}

torch::Tensor repeat_kv_heads(torch::Tensor x, int64_t n_rep)
{
    // NNGraph scale_slice on axis 1 for GQA:
    // [n_kv, B, S, D] -> [n_kv, n_rep, B, S, D] via view+repeat.
    if (n_rep == 1)
    {
        return x;
    }
    int64_t const n_kv = x.size(0);
    int64_t const b = x.size(1);
    int64_t const s = x.size(2);
    int64_t const d = x.size(3);
    return x.view({n_kv, 1, b, s, d})
        .repeat({1, n_rep, 1, 1, 1});
}

} // namespace

// -- LlamaAttentionImpl ----------------------------------------------------

LlamaAttentionImpl::LlamaAttentionImpl(LlamaConfig const &cfg) :
    n_heads(cfg.num_attention_heads),
    n_kv(cfg.num_key_value_heads),
    head_size(cfg.hidden_size / cfg.num_attention_heads),
    n_rep(cfg.num_attention_heads / cfg.num_key_value_heads),
    use_gqa(cfg.num_key_value_heads < cfg.num_attention_heads)
{
    if (cfg.hidden_size % cfg.num_attention_heads != 0)
    {
        throw std::invalid_argument(
            "LlamaAttention: hidden_size % n_heads != 0");
    }
    if (cfg.num_attention_heads % cfg.num_key_value_heads != 0)
    {
        throw std::invalid_argument(
            "LlamaAttention: n_heads % n_kv != 0");
    }
    int64_t const h = cfg.hidden_size;
    int64_t const hs = head_size;
    if (use_gqa)
    {
        q_weight = register_parameter(
            "q_weight",
            torch::empty({h, hs, n_kv, n_rep}));
        o_weight = register_parameter(
            "o_weight",
            torch::empty({hs, n_kv, n_rep, h}));
    }
    else
    {
        q_weight = register_parameter(
            "q_weight",
            torch::empty({h, hs, n_heads}));
        o_weight = register_parameter(
            "o_weight",
            torch::empty({hs, n_heads, h}));
    }
    k_weight = register_parameter(
        "k_weight",
        torch::empty({h, hs, n_kv}));
    v_weight = register_parameter(
        "v_weight",
        torch::empty({h, hs, n_kv}));
    torch::nn::init::normal_(q_weight, 0.0, 0.02);
    torch::nn::init::normal_(k_weight, 0.0, 0.02);
    torch::nn::init::normal_(v_weight, 0.0, 0.02);
    torch::nn::init::normal_(o_weight, 0.0, 0.02);
}

torch::Tensor LlamaAttentionImpl::forward(
    torch::Tensor x,
    torch::Tensor const &sin,
    torch::Tensor const &cos,
    torch::Tensor const &mask)
{
    auto q = gemm(x, q_weight, /*ndim=*/1, /*batch_ndim=*/0);
    q = model_transpose(q, use_gqa ? 2 : 1);
    auto k = model_transpose(
        gemm(x, k_weight, /*ndim=*/1, /*batch_ndim=*/0),
        /*model_ndim=*/1);
    auto v = model_transpose(
        gemm(x, v_weight, /*ndim=*/1, /*batch_ndim=*/0),
        /*model_ndim=*/1);
    q = apply_rope(q, sin, cos);
    k = apply_rope(k, sin, cos);
    auto k_sdpa = k;
    auto v_sdpa = v;
    int64_t batch_ndim = 2;
    if (use_gqa)
    {
        k_sdpa = repeat_kv_heads(k, n_rep);
        v_sdpa = repeat_kv_heads(v, n_rep);
        batch_ndim = 3;
    }
    auto attn = sdpa_kernel(
        q,
        k_sdpa,
        v_sdpa,
        mask,
        batch_ndim);
    attn = model_transpose(attn, /*model_ndim=*/3);
    int64_t const out_ndim = use_gqa ? 3 : 2;
    return gemm(attn, o_weight, out_ndim, /*batch_ndim=*/0);
}

// -- LlamaMLPImpl ---------------------------------------------------------

LlamaMLPImpl::LlamaMLPImpl(LlamaConfig const &cfg)
{
    int64_t const h = cfg.hidden_size;
    int64_t const i = cfg.intermediate_size;
    gate_weight = register_parameter(
        "gate_weight",
        torch::empty({i, h}));
    up_weight = register_parameter(
        "up_weight",
        torch::empty({i, h}));
    down_weight = register_parameter(
        "down_weight",
        torch::empty({h, i}));
    torch::nn::init::normal_(gate_weight, 0.0, 0.02);
    torch::nn::init::normal_(up_weight, 0.0, 0.02);
    torch::nn::init::normal_(down_weight, 0.0, 0.02);
}

torch::Tensor LlamaMLPImpl::forward(torch::Tensor x)
{
    auto gate = gemm(
        x,
        gate_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
    auto up = gemm(
        x,
        up_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
    auto hidden = torch::silu(gate) * up;
    return gemm(
        hidden,
        down_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
}

// -- LlamaDecoderImpl -----------------------------------------------------

LlamaDecoderImpl::LlamaDecoderImpl(LlamaConfig const &cfg) :
    rms_eps(cfg.rms_norm_eps)
{
    input_norm_w = register_parameter(
        "input_norm_w",
        torch::ones({cfg.hidden_size}));
    attn = register_module("attention", LlamaAttention(cfg));
    post_attn_norm_w = register_parameter(
        "post_attn_norm_w",
        torch::ones({cfg.hidden_size}));
    mlp = register_module("mlp", LlamaMLP(cfg));
}

torch::Tensor LlamaDecoderImpl::forward(
    torch::Tensor x,
    torch::Tensor const &sin,
    torch::Tensor const &cos,
    torch::Tensor const &mask)
{
    auto attn_out = attn->forward(
        rms_norm(x, input_norm_w, rms_eps),
        sin,
        cos,
        mask);
    auto post = x + attn_out;
    auto mlp_out = mlp->forward(
        rms_norm(post, post_attn_norm_w, rms_eps));
    return post + mlp_out;
}

// -- LlamaCausalImpl ------------------------------------------------------

LlamaCausalImpl::LlamaCausalImpl(LlamaConfig cfg) : config(std::move(cfg))
{
    if (config.hidden_size % config.num_attention_heads != 0)
    {
        throw std::invalid_argument(
            "LlamaCausal: hidden_size % n_heads != 0");
    }
    embed_tokens = register_module(
        "embed_tokens",
        torch::nn::Embedding(config.vocab_size, config.hidden_size));
    torch::nn::ModuleList list;
    for (int64_t i = 0; i < config.num_hidden_layers; ++i)
    {
        list->push_back(LlamaDecoder(config));
    }
    layers = register_module("layers", list);
    weight_rms = register_parameter(
        "weight_rms",
        torch::ones({config.hidden_size}));
    lm_weight = register_parameter(
        "lm_weight",
        torch::empty({config.vocab_size, config.hidden_size}));
    torch::nn::init::normal_(lm_weight, 0.0, 0.02);
}

void LlamaCausalImpl::warm_rope_cache(
    int64_t batch,
    int64_t seq,
    torch::Device device)
{
    int64_t const head_dim =
        config.hidden_size / config.num_attention_heads;
    torch::Tensor sin_h;
    torch::Tensor cos_h;
    rope_sin_cos_host(
        batch,
        seq,
        head_dim,
        config.rope_theta,
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

torch::Tensor LlamaCausalImpl::forward(torch::Tensor input_ids)
{
    int64_t const b = input_ids.size(0);
    int64_t const s = input_ids.size(1);
    if (!rope_sin_.defined() || rope_cache_batch_ != b
        || rope_cache_seq_ != s
        || rope_sin_.device() != input_ids.device())
    {
        warm_rope_cache(b, s, input_ids.device());
    }
    auto x = embed_tokens->forward(input_ids);
    for (auto &module : *layers)
    {
        x = module->as<LlamaDecoderImpl>()->forward(
            x,
            rope_sin_,
            rope_cos_,
            cached_mask_);
    }
    x = rms_norm(x, weight_rms, config.rms_norm_eps);
    return gemm(
        x,
        lm_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
}

} // namespace torch_nntile::models
