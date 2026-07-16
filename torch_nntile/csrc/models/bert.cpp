/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/bert.cpp
 * BERT MLM — port of deleted ``nntile::model::bert`` (not HF ATen).
 */

#include <torch_nntile/models/bert.hh>

#include "nntile_add_fiber.h"
#include "nntile_gemm.h"
#include "nntile_sdpa.h"
#include "nntile_transpose.h"

#include <stdexcept>

namespace torch_nntile::models
{

namespace
{

torch::Tensor apply_bert_gelu(torch::Tensor x, bool tanh_approx)
{
    if (tanh_approx)
    {
        return torch::gelu(x, "tanh");
    }
    return torch::gelu(x);
}

bool is_gelu_tanh(std::string const &act)
{
    return act == "gelu_pytorch_tanh" || act == "gelutanh" ||
        act == "gelu_new";
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

torch::Tensor bert_position_ids_from_input_ids(
    torch::Tensor const &input_ids,
    int64_t pad_token_id)
{
    int64_t b = input_ids.size(0);
    int64_t s = input_ids.size(1);
    if (pad_token_id < 0)
    {
        auto pos = torch::arange(
            s,
            torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
        pos = pos.unsqueeze(0).expand({b, s}).contiguous();
        if (!input_ids.device().is_cpu())
        {
            pos = pos.to(input_ids.device());
        }
        return pos;
    }
    auto ids_cpu = input_ids.device().is_cpu() ?
        input_ids :
        input_ids.to(torch::kCPU);
    auto mask = ids_cpu.ne(pad_token_id).to(torch::kLong);
    auto position_ids = mask.cumsum(/*dim=*/1) * mask + pad_token_id;
    if (!input_ids.device().is_cpu())
    {
        position_ids = position_ids.contiguous().to(input_ids.device());
    }
    return position_ids;
}

BertSelfAttentionImpl::BertSelfAttentionImpl(BertConfig const &cfg) :
    n_heads(cfg.num_attention_heads),
    head_size(cfg.head_dim()),
    hidden(cfg.hidden_size)
{
    if (hidden % n_heads != 0)
    {
        throw std::invalid_argument(
            "BertSelfAttention: hidden_size must be divisible by heads");
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
    q_bias = register_parameter("q_bias", torch::zeros({nh, hs}));
    k_bias = register_parameter("k_bias", torch::zeros({nh, hs}));
    v_bias = register_parameter("v_bias", torch::zeros({nh, hs}));
    torch::nn::init::normal_(q_weight, 0.0, 0.02);
    torch::nn::init::normal_(k_weight, 0.0, 0.02);
    torch::nn::init::normal_(v_weight, 0.0, 0.02);
}

torch::Tensor BertSelfAttentionImpl::forward(torch::Tensor x)
{
    // Mirror ``bert_self_attention.cc`` (no mask in smoke path).
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
        /*mask=*/std::nullopt,
        /*batch_ndim=*/2);
    return model_transpose(attn, /*model_ndim=*/3);
}

BertSelfOutputImpl::BertSelfOutputImpl(BertConfig const &cfg)
{
    int64_t const hs = cfg.head_dim();
    int64_t const nh = cfg.num_attention_heads;
    int64_t const h = cfg.hidden_size;
    dense_weight = register_parameter(
        "dense_weight",
        torch::empty({hs, nh, h}));
    dense_bias = register_parameter("dense_bias", torch::zeros({h}));
    ln = register_module(
        "ln",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({h}).eps(cfg.layer_norm_eps)));
    torch::nn::init::normal_(dense_weight, 0.0, 0.02);
}

torch::Tensor BertSelfOutputImpl::forward(
    torch::Tensor attn_heads,
    torch::Tensor residual)
{
    // ``bert_self_output.cc``: gemm(ndim=2) + add_fiber + add + LN.
    auto dense_out = gemm(
        attn_heads,
        dense_weight,
        /*ndim=*/2,
        /*batch_ndim=*/0);
    dense_out = add_fiber(
        dense_bias,
        dense_out,
        /*axis=*/dense_out.dim() - 1,
        /*batch_ndim=*/0);
    return ln->forward(residual + dense_out);
}

BertAttentionImpl::BertAttentionImpl(BertConfig const &cfg)
{
    self = register_module("self", BertSelfAttention(cfg));
    output = register_module("output", BertSelfOutput(cfg));
}

torch::Tensor BertAttentionImpl::forward(torch::Tensor x)
{
    auto heads = self->forward(x);
    return output->forward(heads, x);
}

BertLayerImpl::BertLayerImpl(BertConfig const &cfg)
{
    gelu_tanh = is_gelu_tanh(cfg.hidden_act);
    attention = register_module("attention", BertAttention(cfg));
    int64_t const h = cfg.hidden_size;
    int64_t const mid = cfg.intermediate_size;
    inter_weight = register_parameter(
        "inter_weight",
        torch::empty({mid, h}));
    inter_bias = register_parameter("inter_bias", torch::zeros({mid}));
    out_weight = register_parameter(
        "out_weight",
        torch::empty({h, mid}));
    out_bias = register_parameter("out_bias", torch::zeros({h}));
    out_ln = register_module(
        "out_ln",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({h}).eps(cfg.layer_norm_eps)));
    torch::nn::init::normal_(inter_weight, 0.0, 0.02);
    torch::nn::init::normal_(out_weight, 0.0, 0.02);
}

torch::Tensor BertLayerImpl::forward(torch::Tensor x)
{
    // ``bert_layer.cc``: attention → intermediate → output(+residual).
    auto attn_out = attention->forward(x);
    auto mid = linear_gemm(attn_out, inter_weight, inter_bias);
    mid = apply_bert_gelu(mid, gelu_tanh);
    auto proj = linear_gemm(mid, out_weight, out_bias);
    return out_ln->forward(attn_out + proj);
}

BertMlmImpl::BertMlmImpl(BertConfig cfg) : config(std::move(cfg))
{
    gelu_tanh = is_gelu_tanh(config.hidden_act);
    word_embeddings = register_module(
        "word_embeddings",
        torch::nn::Embedding(config.vocab_size, config.hidden_size));
    position_embeddings = register_module(
        "position_embeddings",
        torch::nn::Embedding(
            config.max_position_embeddings,
            config.hidden_size));
    token_type_embeddings = register_module(
        "token_type_embeddings",
        torch::nn::Embedding(config.type_vocab_size, config.hidden_size));
    emb_ln = register_module(
        "emb_ln",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.hidden_size})
                .eps(config.layer_norm_eps)));
    torch::nn::ModuleList list;
    for (int64_t i = 0; i < config.num_hidden_layers; ++i)
    {
        list->push_back(BertLayer(config));
    }
    layers = register_module("layers", list);
    int64_t const h = config.hidden_size;
    transform_weight = register_parameter(
        "transform_weight",
        torch::empty({h, h}));
    transform_bias = register_parameter(
        "transform_bias",
        torch::zeros({h}));
    transform_ln = register_module(
        "transform_ln",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({h}).eps(config.layer_norm_eps)));
    decoder_weight = register_parameter(
        "decoder_weight",
        torch::empty({config.vocab_size, h}));
    decoder_bias = register_parameter(
        "decoder_bias",
        torch::zeros({config.vocab_size}));
    torch::nn::init::normal_(transform_weight, 0.0, 0.02);
    torch::nn::init::normal_(decoder_weight, 0.0, 0.02);
}

torch::Tensor BertMlmImpl::forward(
    torch::Tensor input_ids,
    torch::Tensor token_type_ids)
{
    int64_t b = input_ids.size(0);
    int64_t s = input_ids.size(1);
    torch::Tensor pos;
    if (config.pad_token_id < 0 && cached_pos_.defined() &&
        cache_batch_ == b && cache_seq_ == s &&
        cached_pos_.device() == input_ids.device())
    {
        pos = cached_pos_;
    }
    else
    {
        pos = bert_position_ids_from_input_ids(
            input_ids,
            config.pad_token_id);
        if (config.pad_token_id < 0)
        {
            cached_pos_ = pos;
            cache_batch_ = b;
            cache_seq_ = s;
        }
    }
    auto h = word_embeddings->forward(input_ids) +
        position_embeddings->forward(pos) +
        token_type_embeddings->forward(token_type_ids);
    h = emb_ln->forward(h);
    for (auto &module : *layers)
    {
        h = module->as<BertLayerImpl>()->forward(h);
    }
    h = linear_gemm(h, transform_weight, transform_bias);
    h = apply_bert_gelu(h, gelu_tanh);
    h = transform_ln->forward(h);
    return linear_gemm(h, decoder_weight, decoder_bias);
}

} // namespace torch_nntile::models
