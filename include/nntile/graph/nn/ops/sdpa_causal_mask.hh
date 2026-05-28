/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/nn/ops/sdpa_causal_mask.hh
 * Causal attention mask buffer fill for ``sdpa_eager``.
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>

#include <nntile/graph/common.hh>

namespace nntile::graph
{

//! Fill a BOOL mask buffer for ``sdpa_eager`` (shape ``(seq_len, seq_len)``,
//! Fortran / column-major layout, one byte per element: 0 = false, 1 = true).
//! ``mask_scalar`` keeps logits where the mask is **true** and writes ``-inf``
//! where the mask is **false**. Causal LM: allow keys at or before the query
//! position, i.e. ``mask[kk, qq] = (kk <= qq)``.
void sdpa_causal_mask_bool_fortran_fill(
    Index seq_len,
    std::uint8_t* out);

//! GPT-Neo local (sliding-window) causal mask: allow ``kk`` when
//! ``kk <= qq`` and ``qq - kk < window_size`` (Fortran layout, 1 = allowed).
void sdpa_gptneo_local_mask_bool_fortran_fill(
    Index seq_len,
    Index window_size,
    std::uint8_t* out);

} // namespace nntile::graph
