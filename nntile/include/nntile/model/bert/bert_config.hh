/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/bert/bert_config.hh
 * BERT model configuration.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <nntile/model/common.hh>
#include <nntile/module/activation.hh>

namespace nntile::model::bert
{

//! BERT model configuration (mirrors HuggingFace BertConfig)
struct BertConfig
{
    Index vocab_size = 30522;
    Index hidden_size = 768;
    Index intermediate_size = 3072;
    Index num_hidden_layers = 12;
    Index num_attention_heads = 12;
    Index max_position_embeddings = 512;
    Index type_vocab_size = 2;

    float layer_norm_eps = 1e-12f;

    std::string hidden_act = "gelu";

    std::string name = "bert";

    Index head_dim() const
    {
        return hidden_size / num_attention_heads;
    }

    void validate() const;
};

//! Map HuggingFace BertConfig::hidden_act to graph ActivationType.
inline module::ActivationType activation_type_from_hidden_act(
    const std::string& hidden_act)
{
    if(hidden_act == "gelu")
    {
        return module::ActivationType::GELU;
    }
    if(hidden_act == "gelu_pytorch_tanh" || hidden_act == "gelutanh")
    {
        return module::ActivationType::GELUTANH;
    }
    if(hidden_act == "relu")
    {
        return module::ActivationType::RELU;
    }
    if(hidden_act == "gelu_new")
    {
        return module::ActivationType::GELUTANH;
    }
    if(hidden_act == "silu" || hidden_act == "swish")
    {
        return module::ActivationType::SILU;
    }
    throw std::invalid_argument(
        "BertConfig: unsupported hidden_act '" + hidden_act + "'");
}

inline module::ActivationType activation_type_from_config(
    const BertConfig& config)
{
    return activation_type_from_hidden_act(config.hidden_act);
}

inline void BertConfig::validate() const
{
    if(hidden_size % num_attention_heads != 0)
    {
        throw std::invalid_argument(
            "BertConfig: hidden_size must be divisible by "
            "num_attention_heads");
    }
    (void)activation_type_from_hidden_act(hidden_act);
}

} // namespace nntile::model::bert
