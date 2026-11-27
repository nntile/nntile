/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/scale.cc
 * Scale operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/scale.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/scale.hh"
#else
#include "nntile/starpu/scale.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Tile-wise scale operation
template<typename T>
void scale_async(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
{
    // Check dimensions
    if(dst.ndim != src.ndim)
    {
        throw std::runtime_error("dst.ndim != src.ndim");
    }
    // Check shapes of tiles
    for(Index i = 0; i < dst.ndim; ++i)
    {
        if(dst.shape[i] != src.shape[i])
        {
            throw std::runtime_error("dst.shape[i] != src.shape[i]");
        }
    }
    #ifdef NNTILE_USE_NNHAUL
    int mpi_rank = 0;
    #else
    int mpi_rank = starpu_mpi_world_rank();
    #endif
    #ifdef NNTILE_USE_NNHAUL
    int dst_rank = 0;
    #else
    int dst_rank = dst.mpi_get_rank();
    #endif
    #ifndef NNTILE_USE_NNHAUL
    src.mpi_transfer(dst_rank, mpi_rank);
    #endif
    if(mpi_rank == dst_rank)
    {
        // Insert corresponding task
        #ifdef NNTILE_USE_NNHAUL
        haul::scale.submit<std::tuple<T>>(starpu_worker_hint, src.nelems, alpha, src, dst);
        #else
        starpu::scale.submit<std::tuple<T>>(starpu_worker_hint, src.nelems, alpha, src, dst);
        #endif

    }
}

//! Tile-wise scale operation
template<typename T>
void scale(int starpu_worker_hint, Scalar alpha, const Tile<T> &src, const Tile<T> &dst)
{
    scale_async<T>(starpu_worker_hint, alpha, src, dst);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation of template
template
void scale_async<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scale_async<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void scale_async<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void scale_async<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void scale_async<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void scale_async<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void scale_async<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst);

// Explicit instantiation of template
template
void scale<fp32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_t> &src,
        const Tile<fp32_t> &dst);

template
void scale<fp32_fast_tf32_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void scale<fp32_fast_fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void scale<fp32_fast_bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

template
void scale<fp64_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp64_t> &src,
        const Tile<fp64_t> &dst);

template
void scale<bf16_t>(int starpu_worker_hint, Scalar alpha, const Tile<bf16_t> &src,
        const Tile<bf16_t> &dst);

template
void scale<fp16_t>(int starpu_worker_hint, Scalar alpha, const Tile<fp16_t> &src,
        const Tile<fp16_t> &dst);

} // namespace nntile::core
