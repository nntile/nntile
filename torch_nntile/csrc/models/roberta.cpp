/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/roberta.cpp
 */

#include <torch_nntile/models/roberta.hh>

namespace torch_nntile::models
{

RobertaMlmImpl::RobertaMlmImpl(RobertaConfig cfg) : config(std::move(cfg))
{
    BertConfig bert_cfg = config.to_bert_config();
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
        list->push_back(BertLayer(bert_cfg));
    }
    layers = register_module("layers", list);
    cls = register_module(
        "cls",
        torch::nn::Linear(config.hidden_size, config.vocab_size));
}

torch::Tensor RobertaMlmImpl::forward(
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
