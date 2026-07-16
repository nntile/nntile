/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/roberta.cpp
 * RoBERTa MLM — LibTorch port of deleted NNGraph ``nntile::model::roberta``.
 */

#include <torch_nntile/models/roberta.hh>

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

RobertaMlmImpl::RobertaMlmImpl(RobertaConfig cfg) : config(std::move(cfg))
{
    gelu_tanh = is_gelu_tanh(config.hidden_act);
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
    // RobertaLMHead: dense → act → LN → decoder (+ bias).
    lm_dense = register_module(
        "lm_dense",
        torch::nn::Linear(config.hidden_size, config.hidden_size));
    lm_ln = register_module(
        "lm_ln",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({config.hidden_size})
                .eps(config.layer_norm_eps)));
    lm_decoder = register_module(
        "lm_decoder",
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
    h = lm_ln->forward(apply_bert_gelu(lm_dense->forward(h), gelu_tanh));
    return lm_decoder->forward(h);
}

} // namespace torch_nntile::models
