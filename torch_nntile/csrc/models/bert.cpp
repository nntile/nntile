/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/bert.cpp
 * BERT MLM — LibTorch port of deleted NNGraph ``nntile::model::bert``.
 */

#include <torch_nntile/models/bert.hh>

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

bool is_gelu_tanh(std::string const& act)
{
    return act == "gelu_pytorch_tanh" || act == "gelutanh" ||
        act == "gelu_new";
}

} // namespace

torch::Tensor bert_position_ids_from_input_ids(
    torch::Tensor const& input_ids,
    int64_t pad_token_id)
{
    int64_t b = input_ids.size(0);
    int64_t s = input_ids.size(1);
    if (pad_token_id < 0)
    {
        // Host arange then upload (nntile lacks aten::arange for long).
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
    // Match HF / Python RobertaEmbeddings: compute on CPU, then move.
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

BertLayerImpl::BertLayerImpl(BertConfig const& cfg)
{
    n_head = cfg.num_attention_heads;
    hidden = cfg.hidden_size;
    head_dim = hidden / n_head;
    gelu_tanh = is_gelu_tanh(cfg.hidden_act);
    if (hidden % n_head != 0)
    {
        throw std::invalid_argument(
            "BertLayer: hidden_size must be divisible by heads");
    }
    // Post-norm layout (NNGraph BertAttention / BertOutput).
    query = register_module("query", torch::nn::Linear(hidden, hidden));
    key = register_module("key", torch::nn::Linear(hidden, hidden));
    value = register_module("value", torch::nn::Linear(hidden, hidden));
    attn_dense = register_module(
        "attn_dense",
        torch::nn::Linear(hidden, hidden));
    attn_ln = register_module(
        "attn_ln",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden}).eps(cfg.layer_norm_eps)));
    intermediate = register_module(
        "intermediate",
        torch::nn::Linear(hidden, cfg.intermediate_size));
    output_dense = register_module(
        "output_dense",
        torch::nn::Linear(cfg.intermediate_size, hidden));
    output_ln = register_module(
        "output_ln",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden}).eps(cfg.layer_norm_eps)));
}

torch::Tensor BertLayerImpl::forward(torch::Tensor x)
{
    int64_t b = x.size(0);
    int64_t s = x.size(1);
    auto reshape = [&](torch::Tensor t) {
        return t.view({b, s, n_head, head_dim}).transpose(1, 2);
    };
    auto attn = at::scaled_dot_product_attention(
        reshape(query->forward(x)),
        reshape(key->forward(x)),
        reshape(value->forward(x)),
        /*attn_mask=*/c10::nullopt,
        /*dropout_p=*/0.0,
        /*is_causal=*/false);
    attn = attn.transpose(1, 2).contiguous().view({b, s, hidden});
    // BertSelfOutput: LayerNorm(dense(attn) + residual)
    x = attn_ln->forward(attn_dense->forward(attn) + x);
    auto mid = apply_bert_gelu(intermediate->forward(x), gelu_tanh);
    // BertOutput: LayerNorm(dense(ff) + residual)
    return output_ln->forward(output_dense->forward(mid) + x);
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
    // BertMlmHead: dense → act → LN → decoder (untied; migration debt).
    transform_dense = register_module(
        "transform_dense",
        torch::nn::Linear(config.hidden_size, config.hidden_size));
    transform_ln = register_module(
        "transform_ln",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.hidden_size})
                .eps(config.layer_norm_eps)));
    decoder = register_module(
        "decoder",
        torch::nn::Linear(config.hidden_size, config.vocab_size));
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
    for (auto& module : *layers)
    {
        h = module->as<BertLayerImpl>()->forward(h);
    }
    h = transform_ln->forward(
        apply_bert_gelu(transform_dense->forward(h), gelu_tanh));
    return decoder->forward(h);
}

} // namespace torch_nntile::models
