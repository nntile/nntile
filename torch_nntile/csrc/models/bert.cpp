/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/bert.cpp
 */

#include <torch_nntile/models/bert.hh>

#include <stdexcept>

namespace torch_nntile::models
{

BertLayerImpl::BertLayerImpl(BertConfig const& cfg)
{
    n_head = cfg.num_attention_heads;
    hidden = cfg.hidden_size;
    if (hidden % n_head != 0)
    {
        throw std::invalid_argument(
            "BertLayer: hidden_size must be divisible by heads");
    }
    ln1 = register_module(
        "ln1",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden}).eps(cfg.layer_norm_eps)));
    qkv = register_module("qkv", torch::nn::Linear(hidden, 3 * hidden));
    out = register_module("out", torch::nn::Linear(hidden, hidden));
    ln2 = register_module(
        "ln2",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden}).eps(cfg.layer_norm_eps)));
    ff_in = register_module(
        "ff_in",
        torch::nn::Linear(hidden, cfg.intermediate_size));
    ff_out = register_module(
        "ff_out",
        torch::nn::Linear(cfg.intermediate_size, hidden));
}

torch::Tensor BertLayerImpl::forward(torch::Tensor x)
{
    auto h = ln1->forward(x);
    auto packed = qkv->forward(h).chunk(3, -1);
    int64_t b = x.size(0);
    int64_t s = x.size(1);
    int64_t hs = hidden / n_head;
    auto reshape = [&](torch::Tensor t) {
        return t.view({b, s, n_head, hs}).transpose(1, 2);
    };
    auto attn = torch::nn::functional::scaled_dot_product_attention(
        reshape(packed[0]),
        reshape(packed[1]),
        reshape(packed[2]));
    attn = attn.transpose(1, 2).contiguous().view({b, s, hidden});
    x = x + out->forward(attn);
    auto m = ln2->forward(x);
    m = torch::gelu(ff_in->forward(m));
    return x + ff_out->forward(m);
}

torch::Tensor bert_position_ids_from_input_ids(
    torch::Tensor const& input_ids,
    int64_t pad_token_id)
{
    int64_t s = input_ids.size(1);
    if (pad_token_id < 0)
    {
        return torch::arange(
            s,
            torch::TensorOptions()
                .dtype(torch::kLong)
                .device(input_ids.device()));
    }
    // Match HF / Python RobertaEmbeddings: compute on CPU (nntile lacks
    // integer compare/cumsum ops), then move.
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

BertMlmImpl::BertMlmImpl(BertConfig cfg) : config(std::move(cfg))
{
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
    cls = register_module(
        "cls",
        torch::nn::Linear(config.hidden_size, config.vocab_size));
}

torch::Tensor BertMlmImpl::forward(
    torch::Tensor input_ids,
    torch::Tensor token_type_ids)
{
    auto pos = bert_position_ids_from_input_ids(
        input_ids,
        config.pad_token_id);
    auto h = word_embeddings->forward(input_ids) +
        position_embeddings->forward(pos) +
        token_type_embeddings->forward(token_type_ids);
    h = emb_ln->forward(h);
    for (auto& module : *layers)
    {
        h = module->as<BertLayerImpl>()->forward(h);
    }
    return cls->forward(h);
}

} // namespace torch_nntile::models
