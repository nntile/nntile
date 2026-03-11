/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gpt2/gpt2_config.hh
 * GPT-2 model configuration.
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <cstdint>
#include <stdexcept>
#include <string>

// NNTile headers
#include <nntile/base_types.hh>

namespace nntile::model::gpt2
{

//! GPT-2 model configuration (mirrors HuggingFace GPT2Config)
struct Gpt2Config
{
    Index vocab_size = 50257;
    Index hidden_size = 768;
    Index intermediate_size = 3072;
    Index num_hidden_layers = 12;
    Index num_attention_heads = 12;
    Index max_position_embeddings = 1024;

    float layer_norm_eps = 1e-5f;

    int eos_token_id = 50256;
    int bos_token_id = 50256;

    std::string name = "gpt2";

    //! Compute head_dim from hidden_size and num_attention_heads
    Index head_dim() const
    {
        return hidden_size / num_attention_heads;
    }

    //! Validate configuration
    void validate() const
    {
        if(hidden_size % num_attention_heads != 0)
        {
            throw std::invalid_argument(
                "Gpt2Config: hidden_size must be divisible by num_attention_heads");
        }
    }
};

} // namespace nntile::model::gpt2
