/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/llama.cpp
 */

#include <torch_nntile/models/llama.hh>

#include <cmath>
#include <stdexcept>

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

torch::Tensor apply_rope(
    torch::Tensor x,
    double theta)
{
    // x: [B, H, S, D]; rotate pairs in last dim.
    int64_t d = x.size(-1);
    if (d % 2 != 0)
    {
        throw std::invalid_argument("apply_rope: head_dim must be even");
    }
    int64_t s = x.size(-2);
    int64_t half = d / 2;
    auto freqs = torch::arange(
        half,
        torch::TensorOptions()
            .dtype(x.dtype())
            .device(x.device()));
    freqs = 1.0 / torch::pow(theta, freqs * (2.0 / static_cast<double>(d)));
    auto t = torch::arange(
        s,
        torch::TensorOptions()
            .dtype(x.dtype())
            .device(x.device()));
    auto angles = torch::outer(t, freqs);
    auto cos = angles.cos().unsqueeze(0).unsqueeze(0);
    auto sin = angles.sin().unsqueeze(0).unsqueeze(0);
    auto x1 = x.slice(-1, 0, half);
    auto x2 = x.slice(-1, half, d);
    auto out1 = x1 * cos - x2 * sin;
    auto out2 = x1 * sin + x2 * cos;
    return torch::cat({out1, out2}, -1);
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
    double rope_theta = 10000.0;

    explicit LlamaDecoderImpl(LlamaConfig const& cfg)
    {
        n_heads = cfg.num_attention_heads;
        n_kv = cfg.num_key_value_heads;
        head_dim = cfg.hidden_size / cfg.num_attention_heads;
        eps = cfg.rms_norm_eps;
        rope_theta = cfg.rope_theta;
        q_proj = register_module(
            "q_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(cfg.hidden_size, cfg.hidden_size)
                    .bias(false)));
        k_proj = register_module(
            "k_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    cfg.hidden_size,
                    n_kv * head_dim)
                    .bias(false)));
        v_proj = register_module(
            "v_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    cfg.hidden_size,
                    n_kv * head_dim)
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
                    cfg.hidden_size,
                    cfg.intermediate_size)
                    .bias(false)));
        up_proj = register_module(
            "up_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    cfg.hidden_size,
                    cfg.intermediate_size)
                    .bias(false)));
        down_proj = register_module(
            "down_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(
                    cfg.intermediate_size,
                    cfg.hidden_size)
                    .bias(false)));
        attn_norm_w = register_parameter(
            "attn_norm_w",
            torch::ones({cfg.hidden_size}));
        mlp_norm_w = register_parameter(
            "mlp_norm_w",
            torch::ones({cfg.hidden_size}));
    }

    torch::Tensor forward(torch::Tensor x)
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
        q = apply_rope(q, rope_theta);
        k = apply_rope(k, rope_theta);
        if (n_kv != n_heads)
        {
            int64_t rep = n_heads / n_kv;
            k = k.repeat_interleave(rep, /*dim=*/1);
            v = v.repeat_interleave(rep, /*dim=*/1);
        }
        auto attn = torch::nn::functional::scaled_dot_product_attention(
            q,
            k,
            v,
            torch::nn::functional::ScaledDotProductAttentionFuncOptions()
                .is_causal(true));
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
}

torch::Tensor LlamaCausalImpl::forward(torch::Tensor input_ids)
{
    auto h = embed_tokens->forward(input_ids);
    for (auto& module : *layers)
    {
        h = module->as<LlamaDecoderImpl>()->forward(h);
    }
    h = rms_norm(h, weight_rms, config.rms_norm_eps);
    return lm_head->forward(h);
}

} // namespace torch_nntile::models
