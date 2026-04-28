/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/t5/t5_config.hh
 * T5 model configuration (mirrors HuggingFace T5Config).
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

namespace nntile::model::t5
{

//! T5 model configuration (mirrors HuggingFace T5Config)
struct T5Config
{
    Index vocab_size = 32100;
    Index d_model = 512;
    Index d_kv = 64;       // key/value dimension per head
    Index d_ff = 1024;     // feed-forward intermediate size
    Index num_layers = 6;  // encoder layers
    Index num_decoder_layers = 6;
    Index num_heads = 8;
    Index relative_attention_num_buckets = 32;

    float layer_norm_epsilon = 1e-5f;
    float dropout_rate = 0.0f;

    std::string feed_forward_proj = "gated-gelu";  // gated-gelu for T5
    bool is_gated_act = true;

    int pad_token_id = 0;
    int eos_token_id = 1;
    int decoder_start_token_id = 0;

    std::string name = "t5";

    //! Compute head_dim from d_model and num_heads
    Index head_dim() const
    {
        return d_model / num_heads;
    }

    //! Validate configuration
    void validate() const
    {
        if(d_model % num_heads != 0)
        {
            throw std::invalid_argument(
                "T5Config: d_model must be divisible by num_heads");
        }
        if(d_kv * num_heads != d_model)
        {
            throw std::invalid_argument(
                "T5Config: d_kv * num_heads must equal d_model");
        }
    }
};

} // namespace nntile::model::t5
