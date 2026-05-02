/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/model/llama/llama_causal_mask.cc
 * Causal attention mask buffer fill for Llama.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/model/llama/llama_causal_mask.hh"

#include <stdexcept>

namespace nntile::model::llama
{

void sdpa_causal_mask_bool_fortran_fill(
    Index seq_len,
    std::uint8_t* out)
{
    if(out == nullptr)
    {
        throw std::invalid_argument(
            "sdpa_causal_mask_bool_fortran_fill: out is null");
    }
    if(seq_len <= 0)
    {
        throw std::invalid_argument(
            "sdpa_causal_mask_bool_fortran_fill: seq_len must be positive");
    }
    for(Index qq = 0; qq < seq_len; ++qq)
    {
        for(Index kk = 0; kk < seq_len; ++kk)
        {
            const bool block = kk > qq;
            out[kk + seq_len * qq] =
                block ? static_cast<std::uint8_t>(1)
                      : static_cast<std::uint8_t>(0);
        }
    }
}

} // namespace nntile::model::llama
