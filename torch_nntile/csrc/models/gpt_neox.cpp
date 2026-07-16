/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/gpt_neox.cpp
 * GPT-NeoX — LibTorch port of deleted NNGraph ``nntile::model::gptneox``.
 */

#include <torch_nntile/models/gpt_neox.hh>

#include "nntile_rope.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace torch_nntile::models
{

namespace
{

//! Host RoPE tables for partial rotary dim (NNGraph rope_sin_cos).
void rope_sin_cos_host(
    int64_t batch,
    int64_t seq,
    int64_t rotary_ndims,
    double rope_theta,
    torch::Tensor& sin_out,
    torch::Tensor& cos_out)
{
    if (rotary_ndims <= 0)
    {
        sin_out = torch::Tensor();
        cos_out = torch::Tensor();
        return;
    }
    if (rotary_ndims % 2 != 0)
    {
        throw std::invalid_argument(
            "rope: rotary_ndims must be even");
    }
    int64_t half = rotary_ndims / 2;
    std::vector<float> inv(static_cast<std::size_t>(half));
    for (int64_t i = 0; i < half; ++i)
    {
        double idx = static_cast<double>(2 * i);
        inv[static_cast<std::size_t>(i)] = static_cast<float>(
            1.0 / std::pow(
                rope_theta,
                idx / static_cast<double>(rotary_ndims)));
    }
    sin_out = torch::empty(
        {batch, seq, half},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    cos_out = torch::empty_like(sin_out);
    auto sin_a = sin_out.accessor<float, 3>();
    auto cos_a = cos_out.accessor<float, 3>();
    for (int64_t b = 0; b < batch; ++b)
    {
        for (int64_t s = 0; s < seq; ++s)
        {
            for (int64_t h = 0; h < half; ++h)
            {
                double angle = static_cast<double>(s) *
                    static_cast<double>(inv[static_cast<std::size_t>(h)]);
                sin_a[b][s][h] = static_cast<float>(std::sin(angle));
                cos_a[b][s][h] = static_cast<float>(std::cos(angle));
            }
        }
    }
}

//! Partial RoPE via narrow + interleaved ``rope_forward`` + cat.
torch::Tensor apply_partial_rope(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int64_t rotary_ndims,
    int64_t head_dim)
{
    if (rotary_ndims <= 0)
    {
        return x;
    }
    auto x_rot = x.narrow(/*dim=*/-1, /*start=*/0, /*length=*/rotary_ndims);
    auto x_pass = x.narrow(
        /*dim=*/-1,
        /*start=*/rotary_ndims,
        /*length=*/head_dim - rotary_ndims);
    int64_t n_heads = x.size(1);
    if (sin.dim() == 3)
    {
        int64_t b = sin.size(0);
        int64_t s = sin.size(1);
        int64_t half = sin.size(2);
        sin = sin.view({b, 1, s, half}).repeat({1, n_heads, 1, 1});
        cos = cos.view({b, 1, s, half}).repeat({1, n_heads, 1, 1});
    }
    x_rot = torch_nntile::rope_forward(sin, cos, x_rot);
    return torch::cat({x_rot, x_pass}, /*dim=*/-1);
}

struct GptNeoXLayerImpl : torch::nn::Module
{
    torch::nn::LayerNorm input_layernorm{nullptr};
    torch::nn::Linear query_key_value{nullptr};
    torch::nn::Linear dense{nullptr};
    torch::nn::LayerNorm post_attention_layernorm{nullptr};
    torch::nn::Linear dense_h_to_4h{nullptr};
    torch::nn::Linear dense_4h_to_h{nullptr};
    int64_t n_head = 0;
    int64_t hidden = 0;
    int64_t head_dim = 0;
    int64_t rotary_ndims = 0;
    bool use_parallel_residual = true;

    explicit GptNeoXLayerImpl(GptNeoXConfig const& cfg)
    {
        n_head = cfg.num_attention_heads;
        hidden = cfg.hidden_size;
        head_dim = hidden / n_head;
        rotary_ndims = static_cast<int64_t>(head_dim * cfg.rotary_pct);
        // Match Python / NNGraph: even rotary width.
        if (rotary_ndims % 2 != 0)
        {
            rotary_ndims -= 1;
        }
        use_parallel_residual = cfg.use_parallel_residual;
        if (hidden % n_head != 0)
        {
            throw std::invalid_argument(
                "GptNeoXLayer: hidden_size must be divisible by heads");
        }
        input_layernorm = register_module(
            "input_layernorm",
            torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({hidden})
                    .eps(cfg.layer_norm_eps)));
        query_key_value = register_module(
            "query_key_value",
            torch::nn::Linear(
                torch::nn::LinearOptions(hidden, 3 * hidden)
                    .bias(cfg.attention_bias)));
        dense = register_module(
            "dense",
            torch::nn::Linear(
                torch::nn::LinearOptions(hidden, hidden)
                    .bias(cfg.attention_bias)));
        post_attention_layernorm = register_module(
            "post_attention_layernorm",
            torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({hidden})
                    .eps(cfg.layer_norm_eps)));
        dense_h_to_4h = register_module(
            "dense_h_to_4h",
            torch::nn::Linear(hidden, cfg.intermediate_size));
        dense_4h_to_h = register_module(
            "dense_4h_to_h",
            torch::nn::Linear(cfg.intermediate_size, hidden));
    }

    torch::Tensor attention(
        torch::Tensor x,
        torch::Tensor sin,
        torch::Tensor cos)
    {
        int64_t b = x.size(0);
        int64_t s = x.size(1);
        auto qkv = query_key_value->forward(x)
            .view({b, s, n_head, 3 * head_dim});
        auto chunks = qkv.split(head_dim, /*dim=*/-1);
        auto q = chunks[0].transpose(1, 2);
        auto k = chunks[1].transpose(1, 2);
        auto v = chunks[2].transpose(1, 2);
        q = apply_partial_rope(q, sin, cos, rotary_ndims, head_dim);
        k = apply_partial_rope(k, sin, cos, rotary_ndims, head_dim);
        auto attn = at::scaled_dot_product_attention(
            q,
            k,
            v,
            /*attn_mask=*/c10::nullopt,
            /*dropout_p=*/0.0,
            /*is_causal=*/true);
        attn = attn.transpose(1, 2).contiguous().view({b, s, hidden});
        return dense->forward(attn);
    }

    torch::Tensor mlp(torch::Tensor x)
    {
        return dense_4h_to_h->forward(
            torch::gelu(dense_h_to_4h->forward(x)));
    }

    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor sin,
        torch::Tensor cos)
    {
        if (use_parallel_residual)
        {
            auto normed = input_layernorm->forward(x);
            return x + attention(normed, sin, cos) +
                mlp(post_attention_layernorm->forward(x));
        }
        auto h = x + attention(input_layernorm->forward(x), sin, cos);
        return h + mlp(post_attention_layernorm->forward(h));
    }
};

TORCH_MODULE(GptNeoXLayer);

} // namespace

GptNeoXCausalImpl::GptNeoXCausalImpl(GptNeoXConfig cfg) :
    config(std::move(cfg))
{
    embed_in = register_module(
        "embed_in",
        torch::nn::Embedding(config.vocab_size, config.hidden_size));
    torch::nn::ModuleList list;
    for (int64_t i = 0; i < config.num_hidden_layers; ++i)
    {
        list->push_back(GptNeoXLayer(config));
    }
    layers = register_module("layers", list);
    final_layer_norm = register_module(
        "final_layer_norm",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.hidden_size})
                .eps(config.layer_norm_eps)));
    embed_out = register_module(
        "embed_out",
        torch::nn::Linear(
            torch::nn::LinearOptions(
                config.hidden_size,
                config.vocab_size)
                .bias(false)));
}

void GptNeoXCausalImpl::warm_rope_cache(
    int64_t batch,
    int64_t seq,
    torch::Device device)
{
    int64_t head_dim = config.hidden_size / config.num_attention_heads;
    int64_t rotary_ndims =
        static_cast<int64_t>(head_dim * config.rotary_pct);
    if (rotary_ndims % 2 != 0)
    {
        rotary_ndims -= 1;
    }
    if (rotary_ndims <= 0)
    {
        return;
    }
    if (rope_sin_.defined() && rope_cache_batch_ == batch &&
        rope_cache_seq_ == seq && rope_sin_.device() == device)
    {
        return;
    }
    torch::Tensor sin_h;
    torch::Tensor cos_h;
    rope_sin_cos_host(
        batch,
        seq,
        rotary_ndims,
        config.rotary_emb_base,
        sin_h,
        cos_h);
    if (!device.is_cpu())
    {
        sin_h = sin_h.to(device);
        cos_h = cos_h.to(device);
    }
    torch::NoGradGuard guard;
    rope_sin_ = sin_h;
    rope_cos_ = cos_h;
    rope_cache_batch_ = batch;
    rope_cache_seq_ = seq;
}

torch::Tensor GptNeoXCausalImpl::forward(torch::Tensor input_ids)
{
    int64_t b = input_ids.size(0);
    int64_t s = input_ids.size(1);
    int64_t head_dim = config.hidden_size / config.num_attention_heads;
    int64_t rotary_ndims =
        static_cast<int64_t>(head_dim * config.rotary_pct);
    if (rotary_ndims % 2 != 0)
    {
        rotary_ndims -= 1;
    }
    if (rotary_ndims > 0 &&
        (!rope_sin_.defined() || rope_cache_batch_ != b ||
            rope_cache_seq_ != s ||
            rope_sin_.device() != input_ids.device()))
    {
        warm_rope_cache(b, s, input_ids.device());
    }
    auto x = embed_in->forward(input_ids);
    for (auto& module : *layers)
    {
        x = module->as<GptNeoXLayerImpl>()->forward(
            x,
            rope_sin_,
            rope_cos_);
    }
    x = final_layer_norm->forward(x);
    return embed_out->forward(x);
}

} // namespace torch_nntile::models
