/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/llama.cpp
 * Llama causal LM — LibTorch port of deleted NNGraph llama attention/rope.
 */

#include <torch_nntile/models/llama.hh>

#include "nntile_rope.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace torch_nntile::models
{

namespace
{

torch::Tensor rms_norm(
    torch::Tensor x,
    torch::Tensor weight,
    double eps)
{
    auto var = x.pow(2).mean(-1, /*keepdim=*/true);
    auto y = x * torch::rsqrt(var + eps);
    return y * weight;
}

//! Host RoPE tables (NNGraph ``rope_sin_cos_from_position_ids``), then upload.
void rope_sin_cos_host(
    int64_t batch,
    int64_t seq,
    int64_t head_dim,
    double rope_theta,
    torch::Tensor& sin_out,
    torch::Tensor& cos_out)
{
    if (head_dim % 2 != 0)
    {
        throw std::invalid_argument("rope: head_dim must be even");
    }
    int64_t half = head_dim / 2;
    std::vector<float> inv(static_cast<std::size_t>(half));
    for (int64_t i = 0; i < half; ++i)
    {
        double idx = static_cast<double>(2 * i);
        inv[static_cast<std::size_t>(i)] = static_cast<float>(
            1.0 / std::pow(rope_theta, idx / static_cast<double>(head_dim)));
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

torch::Tensor repeat_kv(torch::Tensor x, int64_t n_rep)
{
    // NNGraph ``scale_slice`` GQA expand → aten::repeat on nntile.
    if (n_rep == 1)
    {
        return x;
    }
    int64_t b = x.size(0);
    int64_t h = x.size(1);
    int64_t s = x.size(2);
    int64_t d = x.size(3);
    return x.view({b, h, 1, s, d})
        .repeat({1, 1, n_rep, 1, 1})
        .view({b, h * n_rep, s, d});
}

torch::Tensor apply_rope_nngraph(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos)
{
    // x: [B,H,S,D]; sin/cos: [B,S,D/2] — expand heads via repeat.
    int64_t n_heads = x.size(1);
    if (sin.dim() == 3)
    {
        int64_t b = sin.size(0);
        int64_t s = sin.size(1);
        int64_t half = sin.size(2);
        sin = sin.view({b, 1, s, half}).repeat({1, n_heads, 1, 1});
        cos = cos.view({b, 1, s, half}).repeat({1, n_heads, 1, 1});
    }
    // Prefer nntile rope kernel (same as deleted NNGraph ``rope`` op).
    return torch_nntile::rope_forward(sin, cos, x);
}

struct LlamaDecoderImpl : torch::nn::Module
{
    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear o_proj{nullptr};
    torch::nn::Linear gate_proj{nullptr};
    torch::nn::Linear up_proj{nullptr};
    torch::nn::Linear down_proj{nullptr};
    torch::Tensor attn_norm_w;
    torch::Tensor mlp_norm_w;
    int64_t n_heads = 0;
    int64_t n_kv = 0;
    int64_t head_dim = 0;
    double eps = 1e-6;

    explicit LlamaDecoderImpl(LlamaConfig const& cfg)
    {
        n_heads = cfg.num_attention_heads;
        n_kv = cfg.num_key_value_heads;
        head_dim = cfg.hidden_size / cfg.num_attention_heads;
        eps = cfg.rms_norm_eps;
        q_proj = register_module(
            "q_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(cfg.hidden_size, cfg.hidden_size)
                    .bias(false)));
        k_proj = register_module(
            "k_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    cfg.hidden_size, n_kv * head_dim)
                    .bias(false)));
        v_proj = register_module(
            "v_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    cfg.hidden_size, n_kv * head_dim)
                    .bias(false)));
        o_proj = register_module(
            "o_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(cfg.hidden_size, cfg.hidden_size)
                    .bias(false)));
        gate_proj = register_module(
            "gate_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    cfg.hidden_size, cfg.intermediate_size)
                    .bias(false)));
        up_proj = register_module(
            "up_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    cfg.hidden_size, cfg.intermediate_size)
                    .bias(false)));
        down_proj = register_module(
            "down_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    cfg.intermediate_size, cfg.hidden_size)
                    .bias(false)));
        attn_norm_w = register_parameter(
            "attn_norm_w", torch::ones({cfg.hidden_size}));
        mlp_norm_w = register_parameter(
            "mlp_norm_w", torch::ones({cfg.hidden_size}));
    }

    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor sin,
        torch::Tensor cos)
    {
        auto h = rms_norm(x, attn_norm_w, eps);
        int64_t b = h.size(0);
        int64_t s = h.size(1);
        auto q = q_proj->forward(h)
            .view({b, s, n_heads, head_dim})
            .transpose(1, 2);
        auto k = k_proj->forward(h)
            .view({b, s, n_kv, head_dim})
            .transpose(1, 2);
        auto v = v_proj->forward(h)
            .view({b, s, n_kv, head_dim})
            .transpose(1, 2);
        q = apply_rope_nngraph(q, sin, cos);
        k = apply_rope_nngraph(k, sin, cos);
        if (n_kv != n_heads)
        {
            int64_t rep = n_heads / n_kv;
            k = repeat_kv(k, rep);
            v = repeat_kv(v, rep);
        }
        auto attn = at::scaled_dot_product_attention(
            q,
            k,
            v,
            /*attn_mask=*/c10::nullopt,
            /*dropout_p=*/0.0,
            /*is_causal=*/true);
        attn = attn.transpose(1, 2).contiguous().view(
            {b, s, n_heads * head_dim});
        x = x + o_proj->forward(attn);
        auto m = rms_norm(x, mlp_norm_w, eps);
        auto gate = torch::silu(gate_proj->forward(m));
        m = down_proj->forward(gate * up_proj->forward(m));
        return x + m;
    }
};

TORCH_MODULE(LlamaDecoder);

} // namespace

LlamaCausalImpl::LlamaCausalImpl(LlamaConfig cfg) :
    config(std::move(cfg))
{
    if (config.hidden_size % config.num_attention_heads != 0)
    {
        throw std::invalid_argument(
            "LlamaCausal: hidden_size must be divisible by "
            "num_attention_heads");
    }
    if (config.num_attention_heads % config.num_key_value_heads != 0)
    {
        throw std::invalid_argument(
            "LlamaCausal: num_attention_heads must be divisible by "
            "num_key_value_heads");
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
    lm_head = register_module(
        "lm_head",
        torch::nn::Linear(
            torch::nn::LinearOptions(config.hidden_size, config.vocab_size)
                .bias(false)));
    // RoPE caches are plain Tensors (not buffers): host tables uploaded once
    // (NNGraph bind_data). register_buffer + .to(nntile) breaks set_data.
}

void LlamaCausalImpl::warm_rope_cache(
    int64_t batch,
    int64_t seq,
    torch::Device device)
{
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
        config.hidden_size / config.num_attention_heads,
        config.rope_theta,
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

torch::Tensor LlamaCausalImpl::forward(torch::Tensor input_ids)
{
    int64_t b = input_ids.size(0);
    int64_t s = input_ids.size(1);
    if (!rope_sin_.defined() || rope_cache_batch_ != b ||
        rope_cache_seq_ != s || rope_sin_.device() != input_ids.device())
    {
        // One-shot table prep (NNGraph bind_data); reused across steps.
        warm_rope_cache(b, s, input_ids.device());
    }
    auto h = embed_tokens->forward(input_ids);
    for (auto& module : *layers)
    {
        h = module->as<LlamaDecoderImpl>()->forward(h, rope_sin_, rope_cos_);
    }
    h = rms_norm(h, weight_rms, config.rms_norm_eps);
    return lm_head->forward(h);
}

} // namespace torch_nntile::models
