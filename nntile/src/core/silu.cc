/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems with GPU acceleration.
 *
 * @file src/core/silu.cc
 * SiLU operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/silu.hh"
#include "nntile/nnhaul/ops/silu.hh"
#include "nntile/nnhaul/sync_defer.hh"

namespace nntile::core
{

template<typename T>
void silu_async(int worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    // Check shapes
    if(src.shape != dst.shape)
    {
        throw std::runtime_error("src.shape != dst.shape");
    }
    // Submit silu
    nnhaul::silu.submit<std::tuple<T>>(worker_hint, src.nelems, src, dst);
}

template<typename T>
void silu(int worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    silu_async<T>(worker_hint, src, dst);
    nntile::nnhaul_task_wait_for_all_unless_deferred();
}

// Explicit instantiation
template
void silu_async<fp32_t>(int worker_hint, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void silu_async<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void silu_async<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void silu_async<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void silu_async<fp64_t>(int worker_hint, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void silu_async<bf16_t>(int worker_hint, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void silu_async<fp16_t>(int worker_hint, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst);

// Explicit instantiation
template
void silu<fp32_t>(int worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void silu<fp32_fast_tf32_t>(int worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void silu<fp32_fast_fp16_t>(int worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void silu<fp32_fast_bf16_t>(int worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void silu<fp64_t>(int worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void silu<bf16_t>(int worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void silu<fp16_t>(int worker_hint, const Tile<fp16_t> &src, const Tile<fp16_t> &dst);

} // namespace nntile::core
