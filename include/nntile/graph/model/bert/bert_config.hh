/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/bert/bert_config.hh
 * BERT model configuration.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <nntile/base_types.hh>

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

    float layer_norm_eps = 1e-5f;

    std::string hidden_act = "gelu";

    std::string name = "bert";

    Index head_dim() const
    {
        return hidden_size / num_attention_heads;
    }

    void validate() const
    {
        if(hidden_size % num_attention_heads != 0)
        {
            throw std::invalid_argument(
                "BertConfig: hidden_size must be divisible by "
                "num_attention_heads");
        }
    }
};

} // namespace nntile::model::bert
