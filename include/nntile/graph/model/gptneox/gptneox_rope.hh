/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/model/gptneox/gptneox_rope.hh
 * RoPE sin/cos from position ids (HuggingFace GPTNeoXRotaryEmbedding).
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>

#include <nntile/base_types.hh>
#include <nntile/graph/model/gptneox/gptneox_config.hh>

namespace nntile::model::gptneox
{

//! Number of head dimensions that receive RoPE (even, from ``rotary_pct``).
Index gptneox_rope_dim(GptneoxConfig const& config);

//! Inverse frequencies for GPT-NeoX RoPE. ``out`` must hold ``rope_dim/2``
//! values.
void rope_inv_freq_gptneox(GptneoxConfig const& config, float* out);

//! Fill ``sin`` and ``cos`` for ``graph::rope`` in layout
//! ``(rope_dim/2, n_seq, n_batch)`` (Fortran order).
void rope_sin_cos_from_position_ids(
    GptneoxConfig const& config,
    std::int64_t const* position_ids,
    Index n_seq,
    Index n_batch,
    float* out_sin,
    float* out_cos);

} // namespace nntile::model::gptneox
