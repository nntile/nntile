/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/core/sqrt.cc
 * Sqrt operation for Tile<T>
 *
 * @version 1.1.0
 * */

#include "nntile/core/sqrt.hh"
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/ops/sqrt.hh"
#else
#include "nntile/starpu/sqrt.hh"
#endif
#ifdef NNTILE_USE_NNHAUL
#include "nntile/nnhaul/sync_defer.hh"
#else
#include "nntile/starpu/config.hh"
#endif

namespace nntile::core
{

//! Asynchronous tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt_async(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
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
        // Submit task without any arguments checked
        #ifdef NNTILE_USE_NNHAUL
        haul::sqrt.submit<std::tuple<T>>(starpu_worker_hint, src.nelems, src, dst);
        #else
        starpu::sqrt.submit<std::tuple<T>>(starpu_worker_hint, src.nelems, src, dst);
        #endif

    }
}

//! Blocking version of tile-wise sqrt operation
/*! @param[inout] A: Tile for the element-wise sqrt operation
 * */
template<typename T>
void sqrt(int starpu_worker_hint, const Tile<T> &src, const Tile<T> &dst)
{
    sqrt_async<T>(starpu_worker_hint, src, dst);
    #ifdef NNTILE_USE_NNHAUL
    nntile::nnhaul_task_wait_for_all_unless_deferred();
#else
    nntile::starpu_task_wait_for_all_unless_deferred();
#endif
}

// Explicit instantiation
template
void sqrt_async<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void sqrt_async<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void sqrt_async<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void sqrt_async<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void sqrt_async<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void sqrt_async<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

// Explicit instantiation
template
void sqrt<fp32_t>(int starpu_worker_hint, const Tile<fp32_t> &src, const Tile<fp32_t> &dst);

template
void sqrt<fp64_t>(int starpu_worker_hint, const Tile<fp64_t> &src, const Tile<fp64_t> &dst);

template
void sqrt<bf16_t>(int starpu_worker_hint, const Tile<bf16_t> &src, const Tile<bf16_t> &dst);

template
void sqrt<fp32_fast_tf32_t>(int starpu_worker_hint, const Tile<fp32_fast_tf32_t> &src,
        const Tile<fp32_fast_tf32_t> &dst);

template
void sqrt<fp32_fast_fp16_t>(int starpu_worker_hint, const Tile<fp32_fast_fp16_t> &src,
        const Tile<fp32_fast_fp16_t> &dst);

template
void sqrt<fp32_fast_bf16_t>(int starpu_worker_hint, const Tile<fp32_fast_bf16_t> &src,
        const Tile<fp32_fast_bf16_t> &dst);

} // namespace nntile::core
