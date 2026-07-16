/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/gpt2.cpp
 * GPT-2 causal LM - port of deleted ``nntile::model::gpt2`` (not HF ATen).
 *
 * Attention matches ``gpt2_attention.cc`` / Python ``gpt2_minimal``:
 * ``gemm(ndim=1)`` -> ``model_transpose(1)`` -> ``add_fiber`` ->
 * ``sdpa_kernel`` -> ``model_transpose(3)`` -> ``gemm(ndim=2)``.
 *
 * Use cyclic ``model_transpose`` only - never the HF pairwise axis-swap
 * bridge (slow; reserved for ATen HF layouts).
 */

#include <torch_nntile/models/gpt2.hh>

#include "nntile_add_fiber.h"
#include "nntile_gemm.h"
#include "nntile_sdpa.h"
#include "nntile_model_transpose.h"

#include <cmath>
#include <stdexcept>

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

} // namespace

Gpt2AttentionImpl::Gpt2AttentionImpl(Gpt2Config const &cfg) :
    n_head(cfg.n_head),
    head_size(cfg.head_size()),
    hidden(cfg.n_embd)
{
    if (cfg.n_embd % cfg.n_head != 0)
    {
        throw std::invalid_argument(
            "Gpt2Attention: n_embd must be divisible by n_head");
    }
    int64_t const hs = head_size;
    int64_t const nh = n_head;
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
    q_bias = register_parameter("q_bias", torch::zeros({nh, hs}));
    k_bias = register_parameter("k_bias", torch::zeros({nh, hs}));
    v_bias = register_parameter("v_bias", torch::zeros({nh, hs}));
    o_bias = register_parameter("o_bias", torch::zeros({h}));
    torch::nn::init::normal_(q_weight, 0.0, 0.02);
    torch::nn::init::normal_(k_weight, 0.0, 0.02);
    torch::nn::init::normal_(v_weight, 0.0, 0.02);
    torch::nn::init::normal_(o_weight, 0.0, 0.02);
}

torch::Tensor Gpt2AttentionImpl::forward(
    torch::Tensor x,
    torch::Tensor const &causal_mask)
{
    // Mirror ``nntile/src/model/gpt2/gpt2_attention.cc``.
    auto q = gemm(x, q_weight, /*ndim=*/1, /*batch_ndim=*/0);
    q = model_transpose(q, /*model_ndim=*/1);
    q = add_fiber(q_bias, q, /*axis=*/3, /*batch_ndim=*/1);

    auto k = gemm(x, k_weight, /*ndim=*/1, /*batch_ndim=*/0);
    k = model_transpose(k, /*model_ndim=*/1);
    k = add_fiber(k_bias, k, /*axis=*/3, /*batch_ndim=*/1);

    auto v = gemm(x, v_weight, /*ndim=*/1, /*batch_ndim=*/0);
    v = model_transpose(v, /*model_ndim=*/1);
    v = add_fiber(v_bias, v, /*axis=*/3, /*batch_ndim=*/1);

    auto attn = sdpa_kernel(
        q,
        k,
        v,
        causal_mask,
        /*batch_ndim=*/2);
    attn = model_transpose(attn, /*model_ndim=*/3);
    auto out = gemm(attn, o_weight, /*ndim=*/2, /*batch_ndim=*/0);
    return add_fiber(
        o_bias,
        out,
        /*axis=*/out.dim() - 1,
        /*batch_ndim=*/0);
}

Gpt2MLPImpl::Gpt2MLPImpl(Gpt2Config const &cfg)
{
    int64_t const h = cfg.n_embd;
    int64_t const inner = cfg.intermediate_size();
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

torch::Tensor Gpt2MLPImpl::forward(torch::Tensor x)
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
    x = torch::gelu(x, "tanh");
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

Gpt2BlockImpl::Gpt2BlockImpl(Gpt2Config const &cfg)
{
    ln_1 = register_module(
        "ln_1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({cfg.n_embd})
                .eps(cfg.layer_norm_epsilon)));
    attn = register_module("attn", Gpt2Attention(cfg));
    ln_2 = register_module(
        "ln_2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({cfg.n_embd})
                .eps(cfg.layer_norm_epsilon)));
    mlp = register_module("mlp", Gpt2MLP(cfg));
}

torch::Tensor Gpt2BlockImpl::forward(
    torch::Tensor x,
    torch::Tensor const &causal_mask)
{
    auto residual = x;
    x = ln_1->forward(x);
    x = attn->forward(x, causal_mask);
    x = residual + x;
    residual = x;
    x = ln_2->forward(x);
    x = mlp->forward(x);
    return residual + x;
}

Gpt2CausalImpl::Gpt2CausalImpl(Gpt2Config cfg) : config(std::move(cfg))
{
    if (config.n_embd % config.n_head != 0)
    {
        throw std::invalid_argument(
            "Gpt2Causal: n_embd must be divisible by n_head");
    }
    wte = register_module(
        "wte",
        torch::nn::Embedding(config.vocab_size, config.n_embd));
    wpe = register_module(
        "wpe",
        torch::nn::Embedding(config.n_positions, config.n_embd));
    torch::nn::ModuleList list;
    for (int64_t i = 0; i < config.n_layer; ++i)
    {
        list->push_back(Gpt2Block(config));
    }
    blocks = register_module("blocks", list);
    ln_f = register_module(
        "ln_f",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.n_embd})
                .eps(config.layer_norm_epsilon)));
    lm_weight = register_parameter(
        "lm_weight",
        torch::empty({config.vocab_size, config.n_embd}));
    torch::nn::init::normal_(lm_weight, 0.0, 0.02);
}

void Gpt2CausalImpl::warm_sequence_cache(
    int64_t batch,
    int64_t seq,
    torch::Device device)
{
    auto pos = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    pos = pos.unsqueeze(0).expand({batch, seq}).contiguous();
    auto mask = causal_mask_host(seq);
    if (!device.is_cpu())
    {
        pos = pos.to(device);
        mask = mask.to(device);
    }
    cached_pos_ = pos;
    cached_mask_ = mask;
    cache_batch_ = batch;
    cache_seq_ = seq;
}

torch::Tensor Gpt2CausalImpl::forward(torch::Tensor input_ids)
{
    int64_t const b = input_ids.size(0);
    int64_t const s = input_ids.size(1);
    if (!cached_pos_.defined() || cache_batch_ != b ||
        cache_seq_ != s ||
        cached_pos_.device() != input_ids.device())
    {
        warm_sequence_cache(b, s, input_ids.device());
    }
    auto x = wte->forward(input_ids) + wpe->forward(cached_pos_);
    for (auto &module : *blocks)
    {
        x = module->as<Gpt2BlockImpl>()->forward(x, cached_mask_);
    }
    x = ln_f->forward(x);
    // Untied LM head: gemm with ``trans_b`` (no bias).
    return gemm(
        x,
        lm_weight,
        /*ndim=*/1,
        /*batch_ndim=*/0,
        /*trans_a=*/false,
        /*trans_b=*/true);
}

} // namespace torch_nntile::models
