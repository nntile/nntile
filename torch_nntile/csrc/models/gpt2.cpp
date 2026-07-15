/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/gpt2.cpp
 */

#include <torch_nntile/models/gpt2.hh>

#include <cmath>
#include <stdexcept>

namespace torch_nntile::models
{

namespace
{

torch::Tensor causal_mask(int64_t seq, torch::Device device)
{
    auto opts = torch::TensorOptions()
        .dtype(torch::kBool)
        .device(device);
    auto q = torch::arange(seq, opts.dtype(torch::kLong).device(device));
    auto k = torch::arange(seq, opts.dtype(torch::kLong).device(device));
    return k.unsqueeze(0) <= q.unsqueeze(1);
}

} // namespace

Gpt2BlockImpl::Gpt2BlockImpl(Gpt2Config const& cfg) :
    n_head(cfg.n_head),
    n_embd(cfg.n_embd)
{
    if (cfg.n_embd % cfg.n_head != 0)
    {
        throw std::invalid_argument(
            "Gpt2Block: n_embd must be divisible by n_head");
    }
    ln_1 = register_module(
        "ln_1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({cfg.n_embd})
                .eps(cfg.layer_norm_epsilon)));
    qkv = register_module(
        "qkv",
        torch::nn::Linear(cfg.n_embd, 3 * cfg.n_embd));
    c_proj = register_module(
        "c_proj",
        torch::nn::Linear(cfg.n_embd, cfg.n_embd));
    ln_2 = register_module(
        "ln_2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({cfg.n_embd})
                .eps(cfg.layer_norm_epsilon)));
    fc_in = register_module(
        "fc_in",
        torch::nn::Linear(cfg.n_embd, 4 * cfg.n_embd));
    fc_out = register_module(
        "fc_out",
        torch::nn::Linear(4 * cfg.n_embd, cfg.n_embd));
}

torch::Tensor Gpt2BlockImpl::forward(
    torch::Tensor x,
    torch::Tensor const& mask)
{
    auto h = ln_1->forward(x);
    auto qkv_out = qkv->forward(h);
    auto chunks = qkv_out.chunk(3, /*dim=*/-1);
    auto q = chunks[0];
    auto k = chunks[1];
    auto v = chunks[2];
    int64_t b = x.size(0);
    int64_t s = x.size(1);
    int64_t hs = n_embd / n_head;
    auto reshape_heads = [&](torch::Tensor t) {
        return t.view({b, s, n_head, hs}).transpose(1, 2);
    };
    q = reshape_heads(q);
    k = reshape_heads(k);
    v = reshape_heads(v);
    auto attn = torch::nn::functional::scaled_dot_product_attention(
        q,
        k,
        v,
        torch::nn::functional::ScaledDotProductAttentionFuncOptions()
            .attn_mask(mask)
            .is_causal(false));
    attn = attn.transpose(1, 2).contiguous().view({b, s, n_embd});
    x = x + c_proj->forward(attn);
    auto m = ln_2->forward(x);
    m = torch::gelu(fc_in->forward(m), "tanh");
    x = x + fc_out->forward(m);
    return x;
}

Gpt2CausalImpl::Gpt2CausalImpl(Gpt2Config cfg) : config(std::move(cfg))
{
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
    lm_head = register_module(
        "lm_head",
        torch::nn::Linear(
            torch::nn::LinearOptions(config.n_embd, config.vocab_size)
                .bias(false)));
}

torch::Tensor Gpt2CausalImpl::forward(torch::Tensor input_ids)
{
    int64_t b = input_ids.size(0);
    int64_t s = input_ids.size(1);
    auto pos = torch::arange(
        s,
        torch::TensorOptions()
            .dtype(torch::kLong)
            .device(input_ids.device()));
    auto h = wte->forward(input_ids) + wpe->forward(pos);
    auto mask = causal_mask(s, input_ids.device());
    for (auto& module : *blocks)
    {
        h = module->as<Gpt2BlockImpl>()->forward(h, mask);
        (void)b;
    }
    h = ln_f->forward(h);
    return lm_head->forward(h);
}

} // namespace torch_nntile::models
