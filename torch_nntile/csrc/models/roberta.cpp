/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/models/roberta.cpp
 * RoBERTa MLM — port of deleted ``nntile::model::roberta`` (NNGraph layout).
 */

#include <torch_nntile/models/roberta.hh>

#include "nntile_add_fiber.h"
#include "nntile_gemm.h"

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
    int64_t const h = config.hidden_size;
    lm_dense_weight = register_parameter(
        "lm_dense_weight",
        torch::empty({h, h}));
    lm_dense_bias = register_parameter(
        "lm_dense_bias",
        torch::zeros({h}));
    lm_ln = register_module(
        "lm_ln",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({h}).eps(config.layer_norm_eps)));
    lm_decoder_weight = register_parameter(
        "lm_decoder_weight",
        torch::empty({config.vocab_size, h}));
    lm_decoder_bias = register_parameter(
        "lm_decoder_bias",
        torch::zeros({config.vocab_size}));
    torch::nn::init::normal_(lm_dense_weight, 0.0, 0.02);
    torch::nn::init::normal_(lm_decoder_weight, 0.0, 0.02);
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
    for (auto &module : *layers)
    {
        h = module->as<BertLayerImpl>()->forward(h);
    }
    h = linear_gemm(h, lm_dense_weight, lm_dense_bias);
    h = apply_bert_gelu(h, gelu_tanh);
    h = lm_ln->forward(h);
    return linear_gemm(h, lm_decoder_weight, lm_decoder_bias);
}

} // namespace torch_nntile::models
