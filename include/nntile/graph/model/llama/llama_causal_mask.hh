/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/llama/llama_causal_mask.hh
 * Causal attention mask for Llama ``sdpa_eager`` (BOOL, ``seq``×``seq``).
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>

#include <nntile/base_types.hh>

namespace nntile::model::llama
{

//! Fill a BOOL mask buffer for ``sdpa_eager`` (shape ``(seq_len, seq_len)``,
//! Fortran / column-major layout, one byte per element: 0 = false, 1 =
//! true). Where the mask is **true**, ``mask_scalar`` writes ``-inf`` into
//! attention logits (blocked). Causal LM: block keys **after** the query
//! position, i.e. ``mask[kk, qq] = (kk > qq)`` (same convention as Llama
//! graph tests: allowed logits have ``kk <= qq``).
void sdpa_causal_mask_bool_fortran_fill(
    Index seq_len,
    std::uint8_t* out);

} // namespace nntile::model::llama
