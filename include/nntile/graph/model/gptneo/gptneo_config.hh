/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneo/gptneo_config.hh
 * GPT-Neo model configuration (mirrors HuggingFace GPTNeoConfig).
 *
 * @version 1.1.0
 * */

#pragma once

// Standard library headers
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// NNTile headers
#include <nntile/base_types.hh>

namespace nntile::model::gptneo
{

//! GPT-Neo model configuration (mirrors HuggingFace GPTNeoConfig)
struct GptneoConfig
{
    Index vocab_size = 50257;
    Index hidden_size = 2048;
    Index intermediate_size = 8192;  // 4 * hidden_size for 1.3B
    Index num_hidden_layers = 24;
    Index num_attention_heads = 16;
    Index max_position_embeddings = 2048;
    Index head_dim = 128;  // hidden_size / num_attention_heads
    Index window_size = 256;  // for local attention

    float layer_norm_eps = 1e-5f;

    int eos_token_id = 50256;
    int bos_token_id = 50256;

    std::string name = "gpt-neo";

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
                "GptneoConfig: hidden_size must be divisible by num_attention_heads");
        }
        if(hidden_size / num_attention_heads != head_dim)
        {
            throw std::invalid_argument(
                "GptneoConfig: head_dim must equal hidden_size / num_attention_heads");
        }
    }
};

} // namespace nntile::model::gptneo
