/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/llama/llama_config.hh
 * Llama model configuration.
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

namespace nntile::model::llama
{

//! Llama model configuration (mirrors HuggingFace LlamaConfig)
struct LlamaConfig
{
    Index vocab_size = 32000;
    Index hidden_size = 4096;
    Index intermediate_size = 11008;
    Index num_hidden_layers = 32;
    Index num_attention_heads = 32;
    Index num_key_value_heads = 32;
    Index max_position_embeddings = 2048;
    Index head_dim = 128;  // hidden_size / num_attention_heads

    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;

    bool attention_bias = false;
    bool mlp_bias = false;

    int eos_token_id = 2;
    int bos_token_id = 1;

    std::string name = "llama";

    //! Compute head_dim from hidden_size and num_attention_heads
    void compute_head_dim()
    {
        if(num_attention_heads > 0 && hidden_size % num_attention_heads == 0)
        {
            head_dim = hidden_size / num_attention_heads;
        }
    }

    //! Validate configuration
    void validate() const
    {
        if(hidden_size % num_attention_heads != 0)
        {
            throw std::invalid_argument(
                "LlamaConfig: hidden_size must be divisible by num_attention_heads");
        }
        if(num_attention_heads % num_key_value_heads != 0)
        {
            throw std::invalid_argument(
                "LlamaConfig: num_attention_heads must be divisible by "
                "num_key_value_heads");
        }
        if(hidden_size / num_attention_heads != head_dim)
        {
            throw std::invalid_argument(
                "LlamaConfig: head_dim must equal hidden_size / num_attention_heads");
        }
    }
};

} // namespace nntile::model::llama
