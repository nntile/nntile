#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/nn_graph/ops/sdpa_causal_mask.cc
 * Causal attention mask buffer fill for ``sdpa_eager``.
 *
 * @version 1.1.0
 * */

#include "nntile/nn/ops/sdpa_causal_mask.hh"

#include <stdexcept>

namespace nntile
{

void sdpa_causal_mask_bool_fill(
    Index seq_len,
    std::uint8_t* out)
{
    if(out == nullptr)
    {
        throw std::invalid_argument(
            "sdpa_causal_mask_bool_fill: out is null");
    }
    if(seq_len <= 0)
    {
        throw std::invalid_argument(
            "sdpa_causal_mask_bool_fill: seq_len must be positive");
    }
    for(Index query = 0; query < seq_len; ++query)
    {
        for(Index key = 0; key < seq_len; ++key)
        {
            const bool allowed = key <= query;
            out[query * seq_len + key] =
                allowed ? static_cast<std::uint8_t>(1)
                        : static_cast<std::uint8_t>(0);
        }
    }
}

void sdpa_gptneo_local_mask_bool_fill(
    Index seq_len,
    Index window_size,
    std::uint8_t* out)
{
    if(out == nullptr)
    {
        throw std::invalid_argument(
            "sdpa_gptneo_local_mask_bool_fill: out is null");
    }
    if(seq_len <= 0)
    {
        throw std::invalid_argument(
            "sdpa_gptneo_local_mask_bool_fill: seq_len must be positive");
    }
    if(window_size <= 0)
    {
        throw std::invalid_argument(
            "sdpa_gptneo_local_mask_bool_fill: window_size must be positive");
    }
    for(Index query = 0; query < seq_len; ++query)
    {
        for(Index key = 0; key < seq_len; ++key)
        {
            const bool allowed =
                key <= query && (query - key) < window_size;
            out[query * seq_len + key] =
                allowed ? static_cast<std::uint8_t>(1)
                        : static_cast<std::uint8_t>(0);
        }
    }
}

} // namespace nntile
