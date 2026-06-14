/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/llama/llama_rope.hh
 * RoPE sin/cos from position ids (HuggingFace ``LlamaRotaryEmbedding`` default).
 *
 * @version 1.1.0
 * */

#pragma once

#include <cstdint>

#include <nntile/model/common.hh>
#include <nntile/model/llama/llama_config.hh>

namespace nntile::model::llama
{

//! Inverse frequencies for default Llama RoPE (HF
//! ``_compute_default_rope_parameters`` / ``LlamaRotaryEmbedding`` with
//! ``rope_type == "default"``). ``out`` must hold ``config.head_dim / 2``
//! values.
void rope_inv_freq_default(LlamaConfig const& config, float* out);

//! Fill ``sin`` and ``cos`` for ``rope`` in layout ``(n_batch, n_seq,
//! head_dim/2)`` (C-order).
//!
//! ``position_ids`` matches NNTile ``input_ids``: logical shape
//! ``(n_batch, n_seq)``; element ``(b, s)`` at
//! ``position_ids[b * n_seq + s]``.
void rope_sin_cos_from_position_ids(
    LlamaConfig const& config,
    std::int64_t const* position_ids,
    Index n_seq,
    Index n_batch,
    float* out_sin,
    float* out_cos);

} // namespace nntile::model::llama
