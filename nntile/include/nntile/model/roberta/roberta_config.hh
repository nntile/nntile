/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file include/nntile/model/roberta/roberta_config.hh
 * RoBERTa model configuration.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <nntile/model/common.hh>
#include <nntile/model/bert/bert_config.hh>
#include <nntile/module/activation.hh>

namespace nntile::model::roberta
{

//! RoBERTa model configuration (mirrors HuggingFace RobertaConfig)
struct RobertaConfig
{
    Index vocab_size = 50265;
    Index hidden_size = 768;
    Index intermediate_size = 3072;
    Index num_hidden_layers = 12;
    Index num_attention_heads = 12;
    Index max_position_embeddings = 514;
    Index type_vocab_size = 1;
    Index pad_token_id = 1;

    float layer_norm_eps = 1e-5f;

    std::string hidden_act = "gelu";

    std::string name = "roberta";

    Index head_dim() const
    {
        return hidden_size / num_attention_heads;
    }

    void validate() const;
};

inline module::ActivationType activation_type_from_config(
    const RobertaConfig& config)
{
    return bert::activation_type_from_hidden_act(config.hidden_act);
}

inline bert::BertConfig to_bert_config(const RobertaConfig& config)
{
    bert::BertConfig bert_cfg;
    bert_cfg.vocab_size = config.vocab_size;
    bert_cfg.hidden_size = config.hidden_size;
    bert_cfg.intermediate_size = config.intermediate_size;
    bert_cfg.num_hidden_layers = config.num_hidden_layers;
    bert_cfg.num_attention_heads = config.num_attention_heads;
    bert_cfg.max_position_embeddings = config.max_position_embeddings;
    bert_cfg.type_vocab_size = config.type_vocab_size;
    bert_cfg.layer_norm_eps = config.layer_norm_eps;
    bert_cfg.hidden_act = config.hidden_act;
    bert_cfg.name = "bert";
    return bert_cfg;
}

inline void RobertaConfig::validate() const
{
    if(hidden_size % num_attention_heads != 0)
    {
        throw std::invalid_argument(
            "RobertaConfig: hidden_size must be divisible by "
            "num_attention_heads");
    }
    (void)bert::activation_type_from_hidden_act(hidden_act);
}

} // namespace nntile::model::roberta
