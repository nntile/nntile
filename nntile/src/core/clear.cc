/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/clear.cc
 * Clear Tile<T>
 *
 * @version 2.0.0
 * */

#include "nntile/core/clear.hh"
#include "nntile/nnhaul/ops/clear.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

//! Asynchronously clear a tile
template<typename T>
void clear_async(int worker_hint, const Tile<T> &tile)
{
    nnhaul::clear.submit(worker_hint, tile);
}

//! Asynchronously clear a tile
template<typename T>
void clear(int worker_hint, const Tile<T> &tile)
{
    clear_async<T>(worker_hint, tile);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void clear_async<fp32_t>(int worker_hint, const Tile<fp32_t> &tile);

template
void clear_async<bf16_t>(int worker_hint, const Tile<bf16_t> &tile);

template
void clear_async<fp16_t>(int worker_hint, const Tile<fp16_t> &tile);

template
void clear_async<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &tile);

template
void clear_async<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &tile);

template
void clear_async<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &tile);

template
void clear_async<fp64_t>(int worker_hint, const Tile<fp64_t> &tile);

template
void clear_async<int64_t>(int worker_hint, const Tile<int64_t> &tile);

template
void clear_async<bool_t>(int worker_hint, const Tile<bool_t> &tile);

// Explicit instantiation
template
void clear<fp32_t>(int worker_hint, const Tile<fp32_t> &tile);

template
void clear<bf16_t>(int worker_hint, const Tile<bf16_t> &tile);

template
void clear<fp16_t>(int worker_hint, const Tile<fp16_t> &tile);

template
void clear<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &tile);

template
void clear<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &tile);

template
void clear<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &tile);

template
void clear<fp64_t>(int worker_hint, const Tile<fp64_t> &tile);

template
void clear<int64_t>(int worker_hint, const Tile<int64_t> &tile);

template
void clear<bool_t>(int worker_hint, const Tile<bool_t> &tile);

} // namespace nntile::core
