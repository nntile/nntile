/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneox/gptneox_config.hh
 * GPT-NeoX model configuration (mirrors HuggingFace GPTNeoXConfig).
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

namespace nntile::model::gptneox
{

//! GPT-NeoX model configuration (mirrors HuggingFace GPTNeoXConfig)
struct GptneoxConfig
{
    Index vocab_size = 50280;
    Index hidden_size = 1024;
    Index intermediate_size = 4096;
    Index num_hidden_layers = 24;
    Index num_attention_heads = 16;
    Index max_position_embeddings = 2048;
    Index head_dim = 64;  // hidden_size / num_attention_heads

    float layer_norm_eps = 1e-5f;
    float rotary_pct = 0.25f;
    float rotary_emb_base = 10000.0f;

    bool use_parallel_residual = true;
    bool attention_bias = false;

    int eos_token_id = 50256;
    int bos_token_id = 50256;

    std::string name = "gpt-neox";

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
                "GptneoxConfig: hidden_size must be divisible by "
                "num_attention_heads");
        }
        if(hidden_size / num_attention_heads != head_dim)
        {
            throw std::invalid_argument(
                "GptneoxConfig: head_dim must equal hidden_size / "
                "num_attention_heads");
        }
    }
};

} // namespace nntile::model::gptneox
