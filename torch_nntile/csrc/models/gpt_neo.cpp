/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/gpt_neo.cpp
 * GPT-Neo — activations on nntile; position / local masks are CPU aux tables.
 */

#include <torch_nntile/models/gpt_neo.hh>

#include <cmath>
#include <stdexcept>

namespace torch_nntile::models
{

namespace
{

//! BOOL local-causal mask on CPU — nntile SDPA converts host masks.
torch::Tensor local_causal_mask_host(int64_t seq, int64_t window)
{
    auto opts = torch::TensorOptions()
        .dtype(torch::kLong)
        .device(torch::kCPU);
    auto q = torch::arange(seq, opts).unsqueeze(1);
    auto k = torch::arange(seq, opts).unsqueeze(0);
    return ((k <= q) & ((q - k) < window)).contiguous();
}

struct GptNeoBlockImpl : torch::nn::Module
{
    torch::nn::LayerNorm ln_1{nullptr};
    torch::nn::Linear q_proj{nullptr};
    torch::nn::Linear k_proj{nullptr};
    torch::nn::Linear v_proj{nullptr};
    torch::nn::Linear out_proj{nullptr};
    torch::nn::LayerNorm ln_2{nullptr};
    torch::nn::Linear c_fc{nullptr};
    torch::nn::Linear c_proj{nullptr};
    int64_t n_head = 0;
    int64_t hidden = 0;
    int64_t head_dim = 0;
    bool local = false;
    int64_t window_size = 256;
    //! Cached host local mask (aux); not an activation.
    torch::Tensor cached_local_mask_;
    int64_t cached_mask_seq_ = -1;

    GptNeoBlockImpl(GptNeoConfig const& cfg, int64_t layer_id)
    {
        n_head = cfg.num_attention_heads;
        hidden = cfg.hidden_size;
        head_dim = hidden / n_head;
        local = cfg.is_local_layer(layer_id);
        window_size = cfg.window_size;
        if (hidden % n_head != 0)
        {
            throw std::invalid_argument(
                "GptNeoBlock: hidden_size must be divisible by heads");
        }
        ln_1 = register_module(
            "ln_1",
            torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({hidden})
                    .eps(cfg.layer_norm_eps)));
        q_proj = register_module(
            "q_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(hidden, hidden).bias(false)));
        k_proj = register_module(
            "k_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(hidden, hidden).bias(false)));
        v_proj = register_module(
            "v_proj",
            torch::nn::Linear(
                torch::nn::LinearOptions(hidden, hidden).bias(false)));
        out_proj = register_module(
            "out_proj",
            torch::nn::Linear(hidden, hidden));
        ln_2 = register_module(
            "ln_2",
            torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({hidden})
                    .eps(cfg.layer_norm_eps)));
        c_fc = register_module(
            "c_fc",
            torch::nn::Linear(hidden, cfg.intermediate_size));
        c_proj = register_module(
            "c_proj",
            torch::nn::Linear(cfg.intermediate_size, hidden));
    }

    torch::Tensor local_mask(int64_t seq)
    {
        if (cached_local_mask_.defined() && cached_mask_seq_ == seq)
        {
            return cached_local_mask_;
        }
        cached_local_mask_ = local_causal_mask_host(seq, window_size);
        cached_mask_seq_ = seq;
        return cached_local_mask_;
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto h = ln_1->forward(x);
        int64_t b = x.size(0);
        int64_t s = x.size(1);
        auto reshape = [&](torch::Tensor t) {
            return t.view({b, s, n_head, head_dim}).transpose(1, 2);
        };
        auto q = reshape(q_proj->forward(h));
        auto k = reshape(k_proj->forward(h));
        auto v = reshape(v_proj->forward(h));
        // HF GPT-Neo scores are unscaled; cancel SDPA 1/sqrt(d) on-device.
        q = at::mul(q, std::sqrt(static_cast<double>(head_dim)));
        c10::optional<at::Tensor> attn_mask = c10::nullopt;
        bool is_causal = false;
        if (local)
        {
            attn_mask = local_mask(s);
        }
        else
        {
            is_causal = true;
        }
        auto attn = at::scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            /*dropout_p=*/0.0,
            is_causal);
        attn = attn.transpose(1, 2).contiguous().view({b, s, hidden});
        x = x + out_proj->forward(attn);
        auto m = ln_2->forward(x);
        m = torch::gelu(c_fc->forward(m), "tanh");
        return x + c_proj->forward(m);
    }
};

TORCH_MODULE(GptNeoBlock);

} // namespace

GptNeoCausalImpl::GptNeoCausalImpl(GptNeoConfig cfg) :
    config(std::move(cfg))
{
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
        list->push_back(GptNeoBlock(config, i));
    }
    blocks = register_module("blocks", list);
    ln_f = register_module(
        "ln_f",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.hidden_size})
                .eps(config.layer_norm_eps)));
    lm_head = register_module(
        "lm_head",
        torch::nn::Linear(
            torch::nn::LinearOptions(
                config.hidden_size,
                config.vocab_size)
                .bias(false)));
}

void GptNeoCausalImpl::warm_position_cache(
    int64_t batch,
    int64_t seq,
    torch::Device device)
{
    if (cached_pos_.defined() && cache_batch_ == batch &&
        cache_seq_ == seq && cached_pos_.device() == device)
    {
        return;
    }
    auto pos = torch::arange(
        seq,
        torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
    pos = pos.unsqueeze(0).expand({batch, seq}).contiguous();
    if (!device.is_cpu())
    {
        pos = pos.to(device);
    }
    torch::NoGradGuard guard;
    cached_pos_ = pos;
    cache_batch_ = batch;
    cache_seq_ = seq;
}

torch::Tensor GptNeoCausalImpl::forward(torch::Tensor input_ids)
{
    int64_t b = input_ids.size(0);
    int64_t s = input_ids.size(1);
    if (!cached_pos_.defined() || cache_batch_ != b || cache_seq_ != s ||
        cached_pos_.device() != input_ids.device())
    {
        warm_position_cache(b, s, input_ids.device());
    }
    auto x = wte->forward(input_ids) + wpe->forward(cached_pos_);
    for (auto& module : *blocks)
    {
        x = module->as<GptNeoBlockImpl>()->forward(x);
    }
    x = ln_f->forward(x);
    return lm_head->forward(x);
}

} // namespace torch_nntile::models
