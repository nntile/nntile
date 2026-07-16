/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/gpt_neo.cpp
 * GPT-Neo causal LM — port of deleted ``nntile::model::gptneo``.
 */

#include <torch_nntile/models/gpt_neo.hh>

#include "nntile_add_fiber.h"
#include "nntile_gemm.h"
#include "nntile_sdpa.h"
#include "nntile_transpose.h"

#include <cmath>
#include <stdexcept>
#include <string>

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

torch::Tensor local_causal_mask_host(int64_t seq, int64_t window)
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
    return ((k <= q) & ((q - k) < window)).to(opts);
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

// ── GptNeoAttentionImpl ───────────────────────────────────────────────────

GptNeoAttentionImpl::GptNeoAttentionImpl(
    GptNeoConfig const &cfg,
    bool local_attn) :
    n_heads(cfg.num_attention_heads),
    head_size(cfg.hidden_size / cfg.num_attention_heads),
    hidden(cfg.hidden_size),
    local(local_attn),
    window_size(cfg.window_size)
{
    if (cfg.hidden_size % cfg.num_attention_heads != 0)
    {
        throw std::invalid_argument(
            "GptNeoAttention: hidden_size % n_heads != 0");
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
    o_bias = register_parameter("o_bias", torch::zeros({h}));
    torch::nn::init::normal_(q_weight, 0.0, 0.02);
    torch::nn::init::normal_(k_weight, 0.0, 0.02);
    torch::nn::init::normal_(v_weight, 0.0, 0.02);
    torch::nn::init::normal_(o_weight, 0.0, 0.02);
}

torch::Tensor GptNeoAttentionImpl::forward(
    torch::Tensor x,
    torch::Tensor const &global_mask,
    torch::Tensor const &local_mask)
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
    auto const &mask = local ? local_mask : global_mask;
    auto attn = sdpa_kernel(
        q,
        k,
        v,
        mask,
        /*batch_ndim=*/2);
    attn = model_transpose(attn, /*model_ndim=*/3);
    auto out = gemm(attn, o_weight, /*ndim=*/2, /*batch_ndim=*/0);
    return add_fiber(
        o_bias,
        out,
        /*axis=*/out.dim() - 1,
        /*batch_ndim=*/0);
}

// ── GptNeoMLPImpl ─────────────────────────────────────────────────────────

GptNeoMLPImpl::GptNeoMLPImpl(GptNeoConfig const &cfg)
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

torch::Tensor GptNeoMLPImpl::forward(torch::Tensor x)
{
    x = linear_gemm(x, fc1_weight, fc1_bias);
    x = torch::gelu(x, "tanh");
    return linear_gemm(x, fc2_weight, fc2_bias);
}

// ── GptNeoDecoderImpl ─────────────────────────────────────────────────────

GptNeoDecoderImpl::GptNeoDecoderImpl(
    GptNeoConfig const &cfg,
    bool local_attn)
{
    input_norm = register_module(
        "input_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({cfg.hidden_size})
                .eps(cfg.layer_norm_eps)));
    attn = register_module(
        "self_attn",
        GptNeoAttention(cfg, local_attn));
    post_attn_norm = register_module(
        "post_attn_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({cfg.hidden_size})
                .eps(cfg.layer_norm_eps)));
    mlp = register_module("mlp", GptNeoMLP(cfg));
}

torch::Tensor GptNeoDecoderImpl::forward(
    torch::Tensor x,
    torch::Tensor const &global_mask,
    torch::Tensor const &local_mask)
{
    auto attn_out = attn->forward(
        input_norm->forward(x),
        global_mask,
        local_mask);
    auto post = x + attn_out;
    auto mlp_out = mlp->forward(post_attn_norm->forward(post));
    return post + mlp_out;
}

// ── GptNeoCausalImpl ──────────────────────────────────────────────────────

GptNeoCausalImpl::GptNeoCausalImpl(GptNeoConfig cfg) :
    config(std::move(cfg))
{
    if (config.hidden_size % config.num_attention_heads != 0)
    {
        throw std::invalid_argument(
            "GptNeoCausal: hidden_size % n_heads != 0");
    }
    wte = register_module(
        "wte",
        torch::nn::Embedding(config.vocab_size, config.hidden_size));
    wpe = register_module(
        "wpe",
        torch::nn::Embedding(
            config.max_position_embeddings,
            config.hidden_size));
    torch::nn::ModuleList list;
    for (int64_t i = 0; i < config.num_hidden_layers; ++i)
    {
        list->push_back(
            GptNeoDecoder(config, config.is_local_layer(i)));
    }
    blocks = register_module("blocks", list);
    ln_f = register_module(
        "ln_f",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.hidden_size})
                .eps(config.layer_norm_eps)));
    lm_weight = register_parameter(
        "lm_weight",
        torch::empty({config.vocab_size, config.hidden_size}));
    torch::nn::init::normal_(lm_weight, 0.0, 0.02);
}

void GptNeoCausalImpl::warm_position_cache(
    int64_t batch,
    int64_t seq,
    torch::Device device)
{
    auto pos = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    pos = pos.unsqueeze(0).expand({batch, seq}).contiguous();
    auto gmask = causal_mask_host(seq);
    auto lmask = local_causal_mask_host(seq, config.window_size);
    if (!device.is_cpu())
    {
        pos = pos.to(device);
        gmask = gmask.to(device);
        lmask = lmask.to(device);
    }
    cached_pos_ = pos;
    cached_global_mask_ = gmask;
    cached_local_mask_ = lmask;
    cache_batch_ = batch;
    cache_seq_ = seq;
}

torch::Tensor GptNeoCausalImpl::forward(torch::Tensor input_ids)
{
    int64_t const b = input_ids.size(0);
    int64_t const s = input_ids.size(1);
    if (!cached_pos_.defined() || cache_batch_ != b
        || cache_seq_ != s
        || cached_pos_.device() != input_ids.device())
    {
        warm_position_cache(b, s, input_ids.device());
    }
    auto x = wte->forward(input_ids) + wpe->forward(cached_pos_);
    for (auto &module : *blocks)
    {
        x = module->as<GptNeoDecoderImpl>()->forward(
            x,
            cached_global_mask_,
            cached_local_mask_);
    }
    x = ln_f->forward(x);
    return gemm(
        x,
        lm_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
}

} // namespace torch_nntile::models
